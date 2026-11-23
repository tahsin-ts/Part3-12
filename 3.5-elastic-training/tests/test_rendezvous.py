import pytest
from conftest import FakeClock

from elastic.rendezvous import OPEN, RUNNING, DONE, Rendezvous, StaleGeneration, UnknownNode


def make(**kw):
    clk = FakeClock()
    events = []
    r = Rendezvous(clock=clk, log=lambda ev, **k: events.append((ev, k)), **kw)
    return r, clk, events


def test_closes_at_max_immediately_with_ranks_in_node_order():
    r, clk, _ = make(min_nodes=2, max_nodes=3)
    for n in ("n3", "n1", "n2"):
        r.join(n, nproc=2)
    assert r.state == RUNNING and r.gen == 1 and r.world == 6
    assert [r.assignment(n)["rank_offset"] for n in ("n1", "n2", "n3")] == [0, 2, 4]


def test_min_nodes_then_last_call_timer():
    r, clk, _ = make(min_nodes=2, max_nodes=4, last_call=0.5)
    r.join("a")
    assert r.state == OPEN
    r.join("b")                       # min reached: wait for stragglers
    assert r.state == OPEN
    clk.advance(0.3); r.join("c")
    assert r.state == OPEN
    clk.advance(0.3); r.maybe_close()
    assert r.state == RUNNING and r.members == ["a", "b", "c"]


def test_lease_expiry_reopens_round_and_clears_old_keys():
    r, clk, ev = make(min_nodes=1, max_nodes=2, lease_ttl=1.0)
    r.join("a"); r.join("b")
    r.kv_set(1, {"x": "1"})
    clk.advance(0.8); r.heartbeat("a")
    clk.advance(0.5)                  # b silent for 1.3 s > ttl
    r.expire()
    assert r.state == OPEN and r.kv == {}
    assert ("node_lost", {"node": "b", "member": True, "late_s": 0.3}) in ev
    r.join("a"); clk.advance(1.0); r.maybe_close()
    assert r.gen == 2 and r.members == ["a"]


def test_stale_generation_writes_and_reads_are_fenced():
    r, clk, _ = make(min_nodes=2, max_nodes=2)
    r.join("a"); r.join("b")
    r.fail("b", 1, "crash")
    with pytest.raises(StaleGeneration):
        r.kv_set(1, {"k": "v"})
    with pytest.raises(StaleGeneration):
        r.kv_get(1, ["k"])
    assert r.rejected_writes == 2


def test_fail_from_old_generation_is_ignored():
    r, clk, _ = make(min_nodes=1, max_nodes=2)
    r.join("a"); r.join("b")
    r.fail("a", 0, "late report")      # generation 0 never ran: no effect
    assert r.state == RUNNING and r.gen == 1


def test_unknown_node_heartbeat_must_rejoin():
    r, clk, _ = make(lease_ttl=1.0)
    r.join("a")
    clk.advance(2.0)
    with pytest.raises(UnknownNode):
        r.heartbeat("a")


def test_spare_waits_at_max_and_takes_over():
    r, clk, _ = make(min_nodes=2, max_nodes=2, last_call=0.0)
    r.join("a"); r.join("b"); r.join("c")
    assert r.members == ["a", "b"] and r.waiting == ["c"] and r.gen == 1
    r.leave("b")
    r.join("a")
    assert r.members == ["a", "c"] and r.gen == 2


def test_waiting_spare_keeps_its_lease_by_touching():
    r, clk, _ = make(min_nodes=2, max_nodes=2, last_call=0.0, lease_ttl=2.0)
    r.join("a"); r.join("b"); r.join("c")
    for _ in range(5):                 # 5 s of long polls; members heartbeat, the spare only waits
        clk.advance(1.0)
        r.heartbeat("a"); r.heartbeat("b"); r.touch("c")
    assert r.waiting == ["c"] and "c" in r.leases and r.gen == 1
    for _ in range(3):                 # a spare that stops polling is forgotten like anyone else
        clk.advance(1.0)
        r.heartbeat("a"); r.heartbeat("b")
    assert r.waiting == [] and r.gen == 1


def test_new_node_with_room_triggers_scale_up():
    r, clk, _ = make(min_nodes=1, max_nodes=3, last_call=0.5)
    r.join("a")
    clk.advance(0.6); r.maybe_close()
    assert r.gen == 1 and r.world == 1
    r.join("b")                        # room for one more: re-rendezvous
    assert r.state == OPEN
    r.join("a")
    clk.advance(0.6); r.maybe_close()
    assert r.gen == 2 and r.members == ["a", "b"]


def test_complete_marks_done():
    r, clk, _ = make(min_nodes=1, max_nodes=1)
    r.join("a")
    r.complete("a", 1)
    assert r.state == DONE and r.assignment("a")["state"] == DONE


def test_straggler_is_evicted_and_banned():
    r, clk, ev = make(min_nodes=3, max_nodes=3, last_call=0.0, evict_lag=0.05, evict_after=3)
    for n in ("a", "b", "c"):
        r.join(n)
    for step in range(3):
        for rank in (0, 1):
            r.kv_set(1, {}, rank=rank, step=step)
        clk.advance(0.5)               # rank 2 (node c) arrives 0.5 s late every step
        r.kv_set(1, {}, rank=2, step=step)
    assert ("straggler_evicted", {"node": "c", "rank": 2}) in ev
    r.join("c")
    assert "c" not in r.waiting        # banned for ban_s seconds
