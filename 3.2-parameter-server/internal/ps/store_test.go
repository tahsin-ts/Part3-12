package ps

import (
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestStoreSGDAndAdagrad(t *testing.T) {
	s := NewStore(2, 0.5, SGD)
	out := make([]float32, 4)
	s.Pull([]uint64{7, 8}, out)
	if out[0] != 0 || s.Len() != 0 {
		t.Fatal("unknown keys must read as zero and not be created")
	}
	s.Push([]uint64{7}, []float32{1, -2})
	s.Pull([]uint64{7}, out[:2])
	if out[0] != -0.5 || out[1] != 1 {
		t.Fatalf("sgd: %v", out[:2])
	}
	a := NewStore(1, 0.1, Adagrad)
	a.Push([]uint64{1}, []float32{2}) // acc=4, w = -0.1*2/2 = -0.1
	a.Push([]uint64{1}, []float32{2}) // acc=8, w = -0.1 - 0.2/sqrt(8)
	a.Pull([]uint64{1}, out[:1])
	want := -0.1 - 0.2/math.Sqrt(8)
	if math.Abs(float64(out[0])-want) > 1e-6 {
		t.Fatalf("adagrad: %v want %v", out[0], want)
	}
}

// 8 goroutines push to the same keys: no update may be lost (run with -race).
func TestStoreConcurrentPushes(t *testing.T) {
	s := NewStore(1, 1, SGD)
	keys := []uint64{1, 2, 3, 4, 5}
	g := []float32{1, 1, 1, 1, 1}
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out := make([]float32, 5)
			for i := 0; i < 1000; i++ {
				s.Push(keys, g)
				s.Pull(keys, out)
			}
		}()
	}
	wg.Wait()
	out := make([]float32, 5)
	s.Pull(keys, out)
	for _, v := range out {
		if v != -8000 {
			t.Fatalf("lost updates: %v", out)
		}
	}
}

func TestSnapshotRoundTripAndCorruption(t *testing.T) {
	s := NewStore(2, 0.1, Adagrad)
	for k := uint64(0); k < 1000; k++ {
		s.Push([]uint64{k}, []float32{float32(k), 1})
	}
	path := filepath.Join(t.TempDir(), "shard.snap")
	n, err := SaveSnapshot(path, s, 1000, map[int]int64{0: 12, 3: 9})
	if err != nil || n != 1000 {
		t.Fatal(n, err)
	}
	r := NewStore(2, 0.1, Adagrad)
	v, clocks, err := LoadSnapshot(path, r)
	if err != nil || v != 1000 || clocks[0] != 12 || clocks[3] != 9 || r.Len() != 1000 {
		t.Fatal(v, clocks, err, r.Len())
	}
	a, b := make([]float32, 2), make([]float32, 2)
	s.Pull([]uint64{777}, a)
	r.Pull([]uint64{777}, b)
	if a[0] != b[0] || a[1] != b[1] {
		t.Fatal("restored row differs")
	}
	data, _ := os.ReadFile(path)
	data[100] ^= 0xff // flip one byte: the CRC must catch it
	os.WriteFile(path, data, 0o644)
	if _, _, err := LoadSnapshot(path, NewStore(2, 0.1, Adagrad)); err != ErrCorrupt {
		t.Fatalf("corruption not detected: %v", err)
	}
	if _, _, err := LoadSnapshot(path, NewStore(3, 0.1, Adagrad)); err == nil {
		t.Fatal("dim mismatch accepted")
	}
}
