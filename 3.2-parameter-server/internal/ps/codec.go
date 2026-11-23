package ps

import (
	"encoding/binary"
	"errors"
	"math"
)

// Wire format, all little-endian:
//
//	pull request:  u32 n | n x u64 key
//	pull reply:    u32 n | u32 dim | n*dim x f32 value
//	push request:  u32 n | u32 dim | n x u64 key | n*dim x f32 gradient
var ErrShort = errors.New("ps: message too short or wrong length")

const maxKeys = 1 << 24 // refuse absurd sizes before allocating

// EncodeKeys builds a pull request.
func EncodeKeys(keys []uint64) []byte {
	b := make([]byte, 4+8*len(keys))
	binary.LittleEndian.PutUint32(b, uint32(len(keys)))
	for i, k := range keys {
		binary.LittleEndian.PutUint64(b[4+8*i:], k)
	}
	return b
}

// DecodeKeys parses a pull request.
func DecodeKeys(b []byte) ([]uint64, error) {
	if len(b) < 4 {
		return nil, ErrShort
	}
	n := int(binary.LittleEndian.Uint32(b))
	if n > maxKeys || len(b) != 4+8*n {
		return nil, ErrShort
	}
	keys := make([]uint64, n)
	for i := range keys {
		keys[i] = binary.LittleEndian.Uint64(b[4+8*i:])
	}
	return keys, nil
}

// EncodeRows builds a pull reply: len(vals) must be n*dim.
func EncodeRows(dim int, vals []float32) []byte {
	n := len(vals) / dim
	b := make([]byte, 8+4*len(vals))
	binary.LittleEndian.PutUint32(b, uint32(n))
	binary.LittleEndian.PutUint32(b[4:], uint32(dim))
	putFloats(b[8:], vals)
	return b
}

// DecodeRows parses a pull reply.
func DecodeRows(b []byte) (dim int, vals []float32, err error) {
	if len(b) < 8 {
		return 0, nil, ErrShort
	}
	n, d := int(binary.LittleEndian.Uint32(b)), int(binary.LittleEndian.Uint32(b[4:]))
	if n > maxKeys || d == 0 || d > 4096 || len(b) != 8+4*n*d {
		return 0, nil, ErrShort
	}
	vals = make([]float32, n*d)
	getFloats(b[8:], vals)
	return d, vals, nil
}

// EncodePush builds a push request.
func EncodePush(keys []uint64, dim int, grads []float32) []byte {
	n := len(keys)
	b := make([]byte, 8+8*n+4*len(grads))
	binary.LittleEndian.PutUint32(b, uint32(n))
	binary.LittleEndian.PutUint32(b[4:], uint32(dim))
	for i, k := range keys {
		binary.LittleEndian.PutUint64(b[8+8*i:], k)
	}
	putFloats(b[8+8*n:], grads)
	return b
}

// DecodePush parses a push request.
func DecodePush(b []byte) (keys []uint64, dim int, grads []float32, err error) {
	if len(b) < 8 {
		return nil, 0, nil, ErrShort
	}
	n, d := int(binary.LittleEndian.Uint32(b)), int(binary.LittleEndian.Uint32(b[4:]))
	if n > maxKeys || d == 0 || d > 4096 || len(b) != 8+8*n+4*n*d {
		return nil, 0, nil, ErrShort
	}
	keys = make([]uint64, n)
	for i := range keys {
		keys[i] = binary.LittleEndian.Uint64(b[8+8*i:])
	}
	grads = make([]float32, n*d)
	getFloats(b[8+8*n:], grads)
	return keys, d, grads, nil
}

func putFloats(b []byte, v []float32) {
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
}

func getFloats(b []byte, v []float32) {
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
}
