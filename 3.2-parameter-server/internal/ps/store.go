package ps

import (
	"math"
	"sync"
)

// Optimizer is the update rule the shard applies to pushed gradients.
type Optimizer int

const (
	SGD     Optimizer = iota // w -= lr * g
	Adagrad                  // acc += g^2; w -= lr * g / sqrt(acc + eps)
)

const nStripes = 64 // lock striping: pushes to different keys rarely contend

type stripe struct {
	mu   sync.RWMutex
	rows map[uint64][]float32 // SGD: [w]; Adagrad: [w | acc]
}

// Store holds the rows of one shard. Rows are created on first push; a pull of
// an unknown key returns zeros without creating it.
type Store struct {
	dim     int
	lr      float32
	opt     Optimizer
	stripes [nStripes]stripe
}

// NewStore creates an empty store.
func NewStore(dim int, lr float32, opt Optimizer) *Store {
	s := &Store{dim: dim, lr: lr, opt: opt}
	s.Reset()
	return s
}

// Dim is the number of float32 values per key.
func (s *Store) Dim() int { return s.dim }

func (s *Store) rowLen() int {
	if s.opt == Adagrad {
		return 2 * s.dim
	}
	return s.dim
}

func (s *Store) stripeOf(k uint64) *stripe { return &s.stripes[Mix64(k)%nStripes] }

// Pull copies the current rows of keys into out (len(keys)*dim).
func (s *Store) Pull(keys []uint64, out []float32) {
	for i, k := range keys {
		st := s.stripeOf(k)
		st.mu.RLock()
		if row, ok := st.rows[k]; ok {
			copy(out[i*s.dim:(i+1)*s.dim], row[:s.dim])
		} else {
			clear(out[i*s.dim : (i+1)*s.dim])
		}
		st.mu.RUnlock()
	}
}

// Push applies one gradient row per key with the store's optimizer.
func (s *Store) Push(keys []uint64, grads []float32) {
	d := s.dim
	for i, k := range keys {
		g := grads[i*d : (i+1)*d]
		st := s.stripeOf(k)
		st.mu.Lock()
		row, ok := st.rows[k]
		if !ok {
			row = make([]float32, s.rowLen())
			st.rows[k] = row
		}
		if s.opt == Adagrad {
			for j, gj := range g {
				row[d+j] += gj * gj
				row[j] -= s.lr * gj / float32(math.Sqrt(float64(row[d+j])+1e-8))
			}
		} else {
			for j, gj := range g {
				row[j] -= s.lr * gj
			}
		}
		st.mu.Unlock()
	}
}

// Len is the number of keys stored.
func (s *Store) Len() int {
	n := 0
	for i := range s.stripes {
		s.stripes[i].mu.RLock()
		n += len(s.stripes[i].rows)
		s.stripes[i].mu.RUnlock()
	}
	return n
}

// Range calls fn for every row (full row, including optimizer state), one
// stripe at a time. fn must not keep the slice.
func (s *Store) Range(fn func(k uint64, row []float32)) {
	for i := range s.stripes {
		st := &s.stripes[i]
		st.mu.RLock()
		for k, r := range st.rows {
			fn(k, r)
		}
		st.mu.RUnlock()
	}
}

// Set installs a full row (used when restoring a snapshot).
func (s *Store) Set(k uint64, row []float32) {
	st := s.stripeOf(k)
	st.mu.Lock()
	st.rows[k] = append([]float32(nil), row...)
	st.mu.Unlock()
}

// Reset drops every row.
func (s *Store) Reset() {
	for i := range s.stripes {
		s.stripes[i].mu.Lock()
		s.stripes[i].rows = make(map[uint64][]float32)
		s.stripes[i].mu.Unlock()
	}
}
