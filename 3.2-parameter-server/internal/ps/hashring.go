// Package ps implements one shard of a sharded parameter server.
package ps

import (
	"fmt"
	"sort"
)

// Mix64 is the splitmix64 finalizer: it spreads nearby integer keys over the
// whole 64-bit range. The Python client uses the same function.
func Mix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// fnv1a hashes a string (the name of a virtual node).
func fnv1a(s string) uint64 {
	h := uint64(14695981039346656037)
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	return h
}

// Ring is a consistent-hash ring: every shard owns many points (virtual
// nodes); a key belongs to the first point clockwise from its hash.
type Ring struct {
	points []uint64
	owner  []int
	names  []string
}

// NewRing builds a ring for shards named shard-0 .. shard-(n-1).
func NewRing(n, vnodes int) *Ring {
	names := make([]string, n)
	for i := range names {
		names[i] = fmt.Sprintf("shard-%d", i)
	}
	return NewRingNames(names, vnodes)
}

// NewRingNames builds a ring from explicit shard names; shard i is names[i].
func NewRingNames(names []string, vnodes int) *Ring {
	type pt struct {
		h uint64
		o int
	}
	pts := make([]pt, 0, len(names)*vnodes)
	for i, name := range names {
		for v := 0; v < vnodes; v++ {
			pts = append(pts, pt{Mix64(fnv1a(fmt.Sprintf("%s#%d", name, v))), i})
		}
	}
	sort.Slice(pts, func(a, b int) bool {
		if pts[a].h != pts[b].h {
			return pts[a].h < pts[b].h
		}
		return pts[a].o < pts[b].o
	})
	r := &Ring{names: names, points: make([]uint64, len(pts)), owner: make([]int, len(pts))}
	for i, p := range pts {
		r.points[i], r.owner[i] = p.h, p.o
	}
	return r
}

// Owner returns the index of the shard that stores key.
func (r *Ring) Owner(key uint64) int {
	h := Mix64(key)
	i := sort.Search(len(r.points), func(i int) bool { return r.points[i] >= h })
	if i == len(r.points) {
		i = 0 // wrap around the ring
	}
	return r.owner[i]
}

// Shards returns the number of shards on the ring.
func (r *Ring) Shards() int { return len(r.names) }
