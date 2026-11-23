"""Rendezvous state machine: membership, generations, leases, fenced key-value store.

Pure logic with an injectable clock, so every rule is unit-tested without sockets.
The HTTP server (server.py) wraps one instance and serialises access with a lock.

Life of a job:
  OPEN     nodes call join(); once min_nodes are waiting a last-call timer starts;
           at max_nodes, or when the timer ends, the round CLOSES.
  RUNNING  generation g has a fixed member list; ranks are assigned in node-id order.
           Any membership change (lease expired, worker failed, node left, a new node
           arrives while there is room, a straggler is evicted) re-OPENS the round.
  DONE     a member reported that training finished.
Every key-value write carries the writer's generation; writes from an old
generation are refused (fencing), so a frozen or partitioned worker cannot
corrupt the new group's state.
"""
import time

OPEN, RUNNING, DONE = "open", "running", "done"


class StaleGeneration(Exception):
    """Raised when a caller presents a generation that is no longer active."""


class UnknownNode(Exception):
    """Raised when a node without a live lease heartbeats; it must rejoin."""


class Rendezvous:
    def __init__(self, min_nodes=1, max_nodes=4, lease_ttl=2.0, last_call=0.5,
                 clock=time.monotonic, log=None, evict_lag=None, evict_after=20, ban_s=60.0):
        assert 1 <= min_nodes <= max_nodes
        self.min_nodes, self.max_nodes = min_nodes, max_nodes
        self.lease_ttl, self.last_call = lease_ttl, last_call
        self.clock, self.log = clock, (log or (lambda ev, **kw: None))
        self.state, self.gen = OPEN, 0
        self.leases = {}          # node -> lease deadline (clock time)
        self.nproc = {}           # node -> local worker count
        self.waiting = []         # nodes that joined the open round, in arrival order
        self.members = []         # nodes of the active generation, sorted
        self.offsets = {}         # node -> first global rank
        self.world = 0
        self.min_reached_at = None
        self.kv = {}
        self.rejected_writes = 0
        self.stragglers = StragglerMonitor(evict_lag, evict_after)
        self.ban_s, self.banned = ban_s, {}   # evicted stragglers stay out for ban_s seconds

    # ---------------------------------------------------------------- membership
    def join(self, node, nproc=1):
        self.expire()
        self.leases[node] = self.clock() + self.lease_ttl
        self.nproc[node] = nproc
        if self.state == DONE or self.banned.get(node, -1) > self.clock():
            return
        if self.state == RUNNING:
            if node in self.members:          # a member rejoining means its workers restarted
                self.open_round(f"{node} rejoined")
            elif len(self.members) < self.max_nodes:
                self.open_round(f"scale-up: {node} arrived")
            # else: the node waits as a hot spare for the next round
        if node not in self.waiting:
            self.waiting.append(node)
            self.log("node_joined", node=node, waiting=len(self.waiting))
        self.maybe_close()

    def heartbeat(self, node, gen=None):
        """Renew the lease. Returns what the agent must know: the active generation and state."""
        self.expire()
        if node not in self.leases:
            raise UnknownNode(node)
        self.leases[node] = self.clock() + self.lease_ttl
        self.maybe_close()
        return {"gen": self.gen, "state": self.state, "member": node in self.members,
                "waiting": len(self.waiting)}

    def touch(self, node):
        """A waiting node (e.g. a hot spare in a long poll) keeps its lease alive without changing the round."""
        self.expire()
        if node in self.leases:
            self.leases[node] = self.clock() + self.lease_ttl
        return node in self.leases

    def leave(self, node, reason="left"):
        self.leases.pop(node, None)
        if node in self.waiting:
            self.waiting.remove(node)
        if self.state == RUNNING and node in self.members:
            self.open_round(f"{node} {reason}")

    def fail(self, node, gen, reason="worker failed"):
        """An agent reports that a local worker died. Only the active generation can trigger a restart."""
        if self.state == RUNNING and gen == self.gen and node in self.members:
            self.open_round(f"{node}: {reason}")

    def complete(self, node, gen):
        if self.state == RUNNING and gen == self.gen and node in self.members:
            self.state = DONE
            self.log("job_done", gen=gen)

    def expire(self):
        now = self.clock()
        for node, deadline in list(self.leases.items()):
            if deadline < now:
                del self.leases[node]
                if node in self.waiting:
                    self.waiting.remove(node)
                lost = self.state == RUNNING and node in self.members
                self.log("node_lost", node=node, member=lost, late_s=round(now - deadline, 3))
                if lost:
                    self.open_round(f"lease of {node} expired")

    def open_round(self, reason):
        self.log("round_open", old_gen=self.gen, reason=reason)
        self.state = OPEN
        self.members, self.offsets, self.world = [], {}, 0
        self.waiting = [n for n in self.waiting if n in self.leases]
        self.min_reached_at = None
        self.kv.clear()                    # keys of the old generation can never be used again

    def maybe_close(self):
        if self.state != OPEN:
            return
        if len(self.waiting) < self.min_nodes:
            self.min_reached_at = None
            return
        now = self.clock()
        if self.min_reached_at is None:
            self.min_reached_at = now
        if len(self.waiting) >= self.max_nodes or now - self.min_reached_at >= self.last_call:
            self.close_round()

    def close_round(self):
        chosen = sorted(self.waiting[:self.max_nodes])
        self.waiting = self.waiting[self.max_nodes:]      # extra nodes stay as spares
        self.gen += 1
        self.members, self.offsets, r = chosen, {}, 0
        for n in chosen:
            self.offsets[n] = r
            r += self.nproc[n]
        self.world, self.state = r, RUNNING
        self.stragglers.reset(self.world)
        self.log("round_closed", gen=self.gen, members=chosen, world=self.world)

    def assignment(self, node):
        """Where this node stands: its ranks if it is a member of the active generation."""
        self.expire()
        self.maybe_close()
        if self.state == DONE:
            return {"state": DONE, "gen": self.gen}
        if self.state == RUNNING and node in self.members:
            return {"state": RUNNING, "gen": self.gen, "world": self.world, "rank_offset": self.offsets[node],
                    "node_rank": self.members.index(node), "members": self.members}
        return {"state": self.state, "gen": self.gen, "waiting": node in self.waiting}

    # ---------------------------------------------------------------- fenced key-value store
    def check_gen(self, gen):
        if self.state != RUNNING or gen != self.gen:
            self.rejected_writes += 1
            self.log("fenced_request", gen=gen, active=self.gen, state=self.state)
            raise StaleGeneration(f"generation {gen} is not active (active={self.gen}, state={self.state})")

    def kv_set(self, gen, items, rank=None, step=None, delete=()):
        self.expire()
        self.check_gen(gen)
        for k in delete:
            self.kv.pop(k, None)
        self.kv.update(items)
        if rank is not None and step is not None:
            node = self.stragglers.record(step, rank, self.clock())
            if node is not None:
                self.evict_rank(node)

    def kv_get(self, gen, keys):
        """Returns the values if all keys exist, else None. Fenced like writes."""
        self.expire()
        self.check_gen(gen)
        if all(k in self.kv for k in keys):
            return {k: self.kv[k] for k in keys}
        return None

    def evict_rank(self, rank):
        for n in self.members:
            if self.offsets[n] <= rank < self.offsets[n] + self.nproc[n]:
                self.log("straggler_evicted", node=n, rank=rank)
                self.leases.pop(n, None)        # its agent sees UnknownNode and stops its workers
                self.banned[n] = self.clock() + self.ban_s
                if n in self.waiting:
                    self.waiting.remove(n)
                self.open_round(f"straggler {n} evicted")
                return

    def status(self):
        self.expire()
        return {"state": self.state, "gen": self.gen, "members": self.members, "world": self.world,
                "waiting": self.waiting, "alive": sorted(self.leases), "rejected_writes": self.rejected_writes,
                "lag_ewma_ms": {r: round(v * 1000, 2) for r, v in self.stragglers.ewma.items()}}


class StragglerMonitor:
    """Arrival lag of each rank at the gradient barrier: lag = arrival - first arrival.

    A rank whose smoothed lag stays above evict_lag seconds for evict_after
    consecutive steps is reported (and evicted when a policy is set)."""

    def __init__(self, evict_lag=None, evict_after=20, alpha=0.2):
        self.evict_lag, self.evict_after, self.alpha = evict_lag, evict_after, alpha
        self.reset(0)

    def reset(self, world):
        self.world, self.arrivals, self.ewma, self.over = world, {}, {}, {}

    def record(self, step, rank, t):
        a = self.arrivals.setdefault(step, {})
        a[rank] = t
        if len(a) < self.world:
            return None
        first = min(a.values())
        del self.arrivals[step]
        worst = None
        for r, tr in a.items():
            self.ewma[r] = self.alpha * (tr - first) + (1 - self.alpha) * self.ewma.get(r, 0.0)
            if self.evict_lag is not None and self.ewma[r] > self.evict_lag:
                self.over[r] = self.over.get(r, 0) + 1
                if self.over[r] >= self.evict_after:
                    worst = r
            else:
                self.over[r] = 0
        return worst
