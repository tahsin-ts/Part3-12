package ps

import (
	"context"
	"testing"
	"time"
)

func TestBSPWaitsForSlowestWorker(t *testing.T) {
	c := NewClocks(2)
	c.Advance(0, 1) // worker 0 finished iteration 0
	done := make(chan error, 1)
	go func() { done <- c.WaitFor(context.Background(), 0, 1, 0, time.Second) }()
	select {
	case <-done:
		t.Fatal("BSP read must wait for worker 1")
	case <-time.After(50 * time.Millisecond):
	}
	c.Advance(1, 1)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if c.Waits != 1 {
		t.Fatalf("waits = %d", c.Waits)
	}
}

func TestSSPAllowsBoundedLead(t *testing.T) {
	c := NewClocks(2)
	ctx := context.Background()
	// staleness 3: worker 0 may run 3 clocks ahead of worker 1 without waiting
	for clk := int64(0); clk <= 3; clk++ {
		if err := c.WaitFor(ctx, 0, clk, 3, 10*time.Millisecond); err != nil {
			t.Fatalf("clock %d: %v", clk, err)
		}
	}
	if err := c.WaitFor(ctx, 0, 4, 3, 30*time.Millisecond); err != ErrWaitTimeout {
		t.Fatalf("clock 4 must wait, got %v", err)
	}
	if c.MaxGap != 3 {
		t.Fatalf("max gap %d, want 3", c.MaxGap)
	}
	if err := c.WaitFor(ctx, 0, 100, -1, time.Millisecond); err != nil {
		t.Fatal("async must never wait")
	}
}

func TestEvictDeadWorkerUnblocksOthers(t *testing.T) {
	c := NewClocks(3)
	if ids := c.Evict(time.Second); len(ids) != 0 {
		t.Fatalf("workers that never started must get a grace period, evicted %v", ids)
	}
	c.Advance(2, 0) // worker 2 starts, then dies at clock 0
	c.Advance(0, 5)
	c.Advance(1, 5)
	done := make(chan error, 1)
	go func() { done <- c.WaitFor(context.Background(), 0, 5, 1, 5*time.Second) }()
	time.Sleep(80 * time.Millisecond)
	c.Advance(1, 5) // worker 1 is alive
	ids := c.Evict(50 * time.Millisecond)
	if len(ids) != 1 || ids[0] != 2 {
		t.Fatalf("evicted %v, want [2] (worker 0 is waiting, worker 1 was just seen)", ids)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	c.Advance(2, 7) // a restarted worker rejoins
	if m, _ := c.Snapshot(); m[2] != 7 {
		t.Fatal("rejoin failed")
	}
}
