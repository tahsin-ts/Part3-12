package ps

import "testing"

// The Python client must route keys exactly like the server; tests/test_hashring.py
// checks the same vectors.
func TestRingMatchesPythonVectors(t *testing.T) {
	want := []int{2, 0, 2, 1, 1, 0, 0, 2, 1, 2, 3, 3, 2, 1, 2, 2}
	r := NewRing(4, 64)
	for k, w := range want {
		if got := r.Owner(uint64(k)); got != w {
			t.Fatalf("key %d: owner %d, want %d", k, got, w)
		}
	}
	if Mix64(12345) != 2454886589211414944 {
		t.Fatalf("Mix64 changed: %d", Mix64(12345))
	}
}

func TestRingBalance(t *testing.T) {
	r := NewRing(4, 64)
	counts := make([]int, 4)
	const n = 200000
	for k := uint64(0); k < n; k++ {
		counts[r.Owner(k)]++
	}
	for i, c := range counts {
		if c < n/4*7/10 || c > n/4*13/10 { // every shard within +-30% of a fair share
			t.Fatalf("shard %d owns %d of %d keys: %v", i, c, n, counts)
		}
	}
}

// Adding a fifth shard must move about 1/5 of the keys, and only to the new shard.
func TestRingMinimalMovement(t *testing.T) {
	old := NewRing(4, 64)
	grown := NewRingNames([]string{"shard-0", "shard-1", "shard-2", "shard-3", "shard-4"}, 64)
	moved := 0
	const n = 100000
	for k := uint64(0); k < n; k++ {
		a, b := old.Owner(k), grown.Owner(k)
		if a != b {
			moved++
			if b != 4 {
				t.Fatalf("key %d moved from %d to %d, not to the new shard", k, a, b)
			}
		}
	}
	if f := float64(moved) / n; f < 0.10 || f > 0.30 {
		t.Fatalf("moved fraction %.3f, want about 0.2", f)
	}
}
