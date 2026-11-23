package ps

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// Config describes one shard.
type Config struct {
	ID, Shards, VNodes, Workers, Dim int
	LR                               float32
	Opt                              Optimizer
	SnapshotPath                     string
	WaitTimeout                      time.Duration
}

// Server is one parameter-server shard behind an HTTP API.
type Server struct {
	cfg     Config
	ring    *Ring
	store   *Store
	clocks  *Clocks
	version atomic.Uint64 // number of non-empty pushes applied
	snapMu  sync.Mutex

	pulls, pushes, keysIn, keysOut, bytesIn, bytesOut, misrouted atomic.Int64
	stale                                                        [9]atomic.Int64 // staleness buckets: 0,1,2,<=4,<=8,<=16,<=32,<=64,>64
	staleSum                                                     atomic.Int64
	snapshots                                                    atomic.Int64
	restoredKeys                                                 int
}

var staleBounds = [8]uint64{0, 1, 2, 4, 8, 16, 32, 64}

// NewServer creates a shard; if a snapshot exists at cfg.SnapshotPath it is loaded.
func NewServer(cfg Config) (*Server, error) {
	if cfg.VNodes == 0 {
		cfg.VNodes = 64
	}
	if cfg.WaitTimeout == 0 {
		cfg.WaitTimeout = 30 * time.Second
	}
	s := &Server{cfg: cfg, ring: NewRing(cfg.Shards, cfg.VNodes), store: NewStore(cfg.Dim, cfg.LR, cfg.Opt), clocks: NewClocks(cfg.Workers)}
	if cfg.SnapshotPath != "" {
		v, clocks, err := LoadSnapshot(cfg.SnapshotPath, s.store)
		switch {
		case err == nil:
			s.version.Store(v)
			s.clocks.Restore(clocks)
			s.restoredKeys = s.store.Len()
		case !isNotExist(err):
			return nil, err
		}
	}
	return s, nil
}

// RestoredKeys is how many keys were loaded from the snapshot at start.
func (s *Server) RestoredKeys() int { return s.restoredKeys }

// Snapshot writes the shard to its snapshot path now.
func (s *Server) Snapshot() (int, error) {
	if s.cfg.SnapshotPath == "" {
		return 0, fmt.Errorf("no snapshot path configured")
	}
	s.snapMu.Lock()
	defer s.snapMu.Unlock()
	clocks, _ := s.clocks.Snapshot()
	n, err := SaveSnapshot(s.cfg.SnapshotPath, s.store, s.version.Load(), clocks)
	if err == nil {
		s.snapshots.Add(1)
	}
	return n, err
}

// EvictLoop removes dead workers every interval until stop is closed.
func (s *Server) EvictLoop(timeout, every time.Duration, stop <-chan struct{}) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case <-t.C:
			if ids := s.clocks.Evict(timeout); len(ids) > 0 {
				log.Printf("shard %d: evicted workers %v (no request for %v)", s.cfg.ID, ids, timeout)
			}
		}
	}
}

// Handler returns the HTTP API.
func (s *Server) Handler() http.Handler {
	m := http.NewServeMux()
	m.HandleFunc("POST /pull", s.handlePull)
	m.HandleFunc("POST /push", s.handlePush)
	m.HandleFunc("POST /leave", func(w http.ResponseWriter, r *http.Request) {
		s.clocks.Leave(qint(r, "w", -1))
	})
	m.HandleFunc("POST /reset", func(w http.ResponseWriter, r *http.Request) {
		s.store.Reset()
		s.clocks.Reset(qint(r, "workers", s.cfg.Workers))
		s.version.Store(0)
		for _, c := range []*atomic.Int64{&s.pulls, &s.pushes, &s.keysIn, &s.keysOut, &s.bytesIn, &s.bytesOut, &s.misrouted, &s.staleSum} {
			c.Store(0)
		}
		for i := range s.stale {
			s.stale[i].Store(0)
		}
	})
	m.HandleFunc("POST /snapshot", func(w http.ResponseWriter, r *http.Request) {
		n, err := s.Snapshot()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		fmt.Fprintf(w, "{\"keys\": %d}\n", n)
	})
	m.HandleFunc("GET /stats", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(s.Stats())
	})
	m.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "ok\n") })
	return m
}

// owned rejects a request containing a key that belongs to another shard:
// a client with a different ring would otherwise silently split a key's state.
func (s *Server) owned(w http.ResponseWriter, keys []uint64) bool {
	for _, k := range keys {
		if o := s.ring.Owner(k); o != s.cfg.ID {
			s.misrouted.Add(1)
			http.Error(w, fmt.Sprintf("key %d belongs to shard %d, not %d", k, o, s.cfg.ID), http.StatusConflict)
			return false
		}
	}
	return true
}

func (s *Server) handlePull(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<28))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	keys, err := DecodeKeys(body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if !s.owned(w, keys) {
		return
	}
	wk, clk, stl := qint(r, "w", -1), int64(qint(r, "c", 0)), int64(qint(r, "s", -1))
	if err := s.clocks.WaitFor(r.Context(), wk, clk, stl, s.cfg.WaitTimeout); err != nil {
		http.Error(w, err.Error(), http.StatusGatewayTimeout)
		return
	}
	v := s.version.Load() // read the version before the rows: staleness is never under-counted
	vals := make([]float32, len(keys)*s.store.Dim())
	s.store.Pull(keys, vals)
	out := EncodeRows(s.store.Dim(), vals)
	s.pulls.Add(1)
	s.keysOut.Add(int64(len(keys)))
	s.bytesIn.Add(int64(len(body)))
	s.bytesOut.Add(int64(len(out)))
	w.Header().Set("X-Version", strconv.FormatUint(v, 10))
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Write(out)
}

func (s *Server) handlePush(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<28))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	keys, dim, grads, err := DecodePush(body)
	if err != nil || dim != s.store.Dim() {
		http.Error(w, fmt.Sprintf("bad push (dim %d, want %d): %v", dim, s.store.Dim(), err), http.StatusBadRequest)
		return
	}
	if !s.owned(w, keys) {
		return
	}
	if len(keys) > 0 {
		s.store.Push(keys, grads)
		now := s.version.Add(1) - 1 // updates applied before this one
		if base, err := strconv.ParseUint(r.URL.Query().Get("v"), 10, 64); err == nil && now >= base {
			st := now - base // updates this gradient did not see
			s.staleSum.Add(int64(st))
			b := 8
			for i, ub := range staleBounds {
				if st <= ub {
					b = i
					break
				}
			}
			s.stale[b].Add(1)
		}
	}
	s.clocks.Advance(qint(r, "w", -1), int64(qint(r, "c", 0)))
	s.pushes.Add(1)
	s.keysIn.Add(int64(len(keys)))
	s.bytesIn.Add(int64(len(body)))
}

// Stats is the JSON document served at /stats.
type Stats struct {
	Shard, Shards, Keys                   int
	Version                               uint64
	Pulls, Pushes, KeysPulled, KeysPushed int64
	BytesIn, BytesOut, Misrouted          int64
	MinClock                              int64
	Clocks                                map[string]int64
	Evicted                               []int
	Waits, WaitMs, MaxClockGap, Timeouts  int64
	MeanClockGap, MeanStaleness           float64
	StalenessHist                         map[string]int64
	Snapshots                             int64
	RestoredKeys                          int
}

// Stats collects counters.
func (s *Server) Stats() Stats {
	clocks, ev := s.clocks.Snapshot()
	cs := map[string]int64{}
	for k, v := range clocks {
		cs[strconv.Itoa(k)] = v
	}
	s.clocks.mu.Lock()
	waits, waitNs, maxGap, gs, gn, to := s.clocks.Waits, s.clocks.WaitNs, s.clocks.MaxGap, s.clocks.GapSum, s.clocks.GapN, s.clocks.Timeout
	s.clocks.mu.Unlock()
	mc := s.clocks.Min()
	if len(clocks) == 0 {
		mc = -1
	}
	hist, total := map[string]int64{}, int64(0)
	names := []string{"0", "1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", ">64"}
	for i := range s.stale {
		if v := s.stale[i].Load(); v > 0 {
			hist[names[i]] = v
			total += v
		}
	}
	st := Stats{Shard: s.cfg.ID, Shards: s.cfg.Shards, Keys: s.store.Len(), Version: s.version.Load(),
		Pulls: s.pulls.Load(), Pushes: s.pushes.Load(), KeysPulled: s.keysOut.Load(), KeysPushed: s.keysIn.Load(),
		BytesIn: s.bytesIn.Load(), BytesOut: s.bytesOut.Load(), Misrouted: s.misrouted.Load(), MinClock: mc,
		Clocks: cs, Evicted: ev, Waits: waits, WaitMs: waitNs / 1e6, MaxClockGap: maxGap, Timeouts: to,
		StalenessHist: hist, Snapshots: s.snapshots.Load(), RestoredKeys: s.restoredKeys}
	if gn > 0 {
		st.MeanClockGap = float64(gs) / float64(gn)
	}
	if total > 0 {
		st.MeanStaleness = float64(s.staleSum.Load()) / float64(total)
	}
	return st
}

func qint(r *http.Request, name string, def int) int {
	v, err := strconv.Atoi(r.URL.Query().Get(name))
	if err != nil {
		return def
	}
	return v
}
