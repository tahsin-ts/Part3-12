package ps

import (
	"context"
	"errors"
	"math"
	"sync"
	"time"
)

// ErrWaitTimeout is returned when a bounded-staleness pull waited too long.
var ErrWaitTimeout = errors.New("ps: staleness wait timed out")

// Clocks tracks, per worker, how many iterations it has finished (its clock)
// and implements the SSP rule: a worker at clock c may read only when every
// live worker has reached at least c - s.
//
//	s = 0   bulk synchronous (BSP): wait for everybody
//	s > 0   stale synchronous parallel (SSP)
//	s < 0   asynchronous: never wait
type Clocks struct {
	mu      sync.Mutex
	clock   map[int]int64
	last    map[int]time.Time
	waiting map[int]int
	seen    map[int]bool // has sent at least one request
	evicted []int
	changed chan struct{} // closed and replaced whenever a clock moves

	Waits   int64 // pulls that had to block
	WaitNs  int64 // total blocked time
	MaxGap  int64 // largest c - min(clock) seen when a pull was answered
	GapSum  int64
	GapN    int64
	Timeout int64
}

// NewClocks expects workers 0..n-1, all at clock 0.
func NewClocks(n int) *Clocks {
	c := &Clocks{}
	c.Reset(n)
	return c
}

// Reset forgets all workers and statistics.
func (c *Clocks) Reset(n int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.clock, c.last, c.waiting, c.seen = map[int]int64{}, map[int]time.Time{}, map[int]int{}, map[int]bool{}
	now := time.Now()
	for w := 0; w < n; w++ {
		c.clock[w], c.last[w] = 0, now
	}
	c.evicted = nil
	c.changed = make(chan struct{})
	c.Waits, c.WaitNs, c.MaxGap, c.GapSum, c.GapN, c.Timeout = 0, 0, 0, 0, 0, 0
}

func (c *Clocks) broadcast() {
	close(c.changed)
	c.changed = make(chan struct{})
}

// minLocked is the smallest clock among live workers (MaxInt64 if none).
func (c *Clocks) minLocked() int64 {
	m := int64(math.MaxInt64)
	for _, v := range c.clock {
		if v < m {
			m = v
		}
	}
	return m
}

// Advance records that worker w has finished clk iterations (clocks never go
// back). A worker that was evicted rejoins.
func (c *Clocks) Advance(w int, clk int64) {
	if w < 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.advanceLocked(w, clk)
}

func (c *Clocks) advanceLocked(w int, clk int64) {
	c.last[w] = time.Now()
	c.seen[w] = true
	old, ok := c.clock[w]
	if !ok || clk > old {
		c.clock[w] = clk
		c.broadcast()
	}
}

// WaitFor blocks until worker w at clock clk may read under staleness s.
func (c *Clocks) WaitFor(ctx context.Context, w int, clk, s int64, timeout time.Duration) error {
	if w < 0 {
		return nil // reads that are not part of training (evaluation)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.advanceLocked(w, clk)
	start := time.Now()
	blocked := false
	var timer *time.Timer
	for {
		m := c.minLocked()
		if s < 0 || m >= clk-s {
			gap := clk - m
			if m == math.MaxInt64 || gap < 0 {
				gap = 0
			}
			if gap > c.MaxGap {
				c.MaxGap = gap
			}
			c.GapSum += gap
			c.GapN++
			if blocked {
				c.Waits++
				c.WaitNs += int64(time.Since(start))
			}
			return nil
		}
		if timer == nil {
			timer = time.NewTimer(timeout)
			defer timer.Stop()
		}
		blocked = true
		ch := c.changed
		c.waiting[w]++
		c.mu.Unlock()
		var err error
		select {
		case <-ch:
		case <-timer.C:
			err = ErrWaitTimeout
		case <-ctx.Done():
			err = ctx.Err()
		}
		c.mu.Lock()
		c.waiting[w]--
		c.last[w] = time.Now() // a waiting worker is alive
		if err != nil {
			if err == ErrWaitTimeout {
				c.Timeout++
			}
			return err
		}
	}
}

// Leave removes a worker that finished (or is shutting down) cleanly.
func (c *Clocks) Leave(w int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.clock[w]; ok {
		delete(c.clock, w)
		c.broadcast()
	}
}

// Evict removes workers not seen for longer than timeout and not currently
// waiting; it returns their ids. Without this, one dead worker would block
// every synchronous pull forever. A worker that never sent a request gets
// 10 x timeout to start (loading data can take a while).
func (c *Clocks) Evict(timeout time.Duration) []int {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []int
	for w := range c.clock {
		limit := timeout
		if !c.seen[w] {
			limit = 10 * timeout
		}
		if c.waiting[w] == 0 && time.Since(c.last[w]) > limit {
			delete(c.clock, w)
			out = append(out, w)
		}
	}
	if len(out) > 0 {
		c.evicted = append(c.evicted, out...)
		c.broadcast()
	}
	return out
}

// Snapshot returns a copy of the live clocks and the evicted list.
func (c *Clocks) Snapshot() (map[int]int64, []int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	m := make(map[int]int64, len(c.clock))
	for k, v := range c.clock {
		m[k] = v
	}
	return m, append([]int(nil), c.evicted...)
}

// Restore sets clocks (after loading a snapshot); live workers keep the larger value.
func (c *Clocks) Restore(m map[int]int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for w, v := range m {
		if v > c.clock[w] {
			c.clock[w] = v
		}
		c.last[w] = time.Now()
	}
	c.broadcast()
}

// Min is the smallest live clock.
func (c *Clocks) Min() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.minLocked()
}
