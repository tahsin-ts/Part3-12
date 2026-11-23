package ps

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"
	"path/filepath"
)

// Snapshot file layout (little-endian), followed by a CRC-32 of all bytes before it:
//
//	"PSS1" | u32 dim | u32 rowLen | u64 version | u64 nKeys | nKeys x (u64 key, rowLen x f32)
//	| u32 nClocks | nClocks x (i32 worker, i64 clock) | u32 crc
var ErrCorrupt = errors.New("ps: snapshot corrupt")

type crcWriter struct {
	w   io.Writer
	crc uint32
}

func (c *crcWriter) Write(p []byte) (int, error) {
	c.crc = crc32.Update(c.crc, crc32.IEEETable, p)
	return c.w.Write(p)
}

// SaveSnapshot writes the store, version and clocks to path atomically:
// write a temporary file, fsync it, then rename over the old snapshot.
func SaveSnapshot(path string, st *Store, version uint64, clocks map[int]int64) (int, error) {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".snap-*")
	if err != nil {
		return 0, err
	}
	defer os.Remove(tmp.Name()) // no-op after a successful rename
	bw := bufio.NewWriterSize(tmp, 1<<20)
	cw := &crcWriter{w: bw}
	le := binary.LittleEndian
	rowLen := st.rowLen()
	hdr := make([]byte, 28)
	copy(hdr, "PSS1")
	le.PutUint32(hdr[4:], uint32(st.dim))
	le.PutUint32(hdr[8:], uint32(rowLen))
	le.PutUint64(hdr[12:], version)
	cw.Write(hdr[:20])
	// The key count is written after the rows, because pushes may add keys while we iterate.
	buf := make([]byte, 8+4*rowLen)
	var body []byte
	count := 0
	st.Range(func(k uint64, row []float32) {
		le.PutUint64(buf, k)
		putFloats(buf[8:], row)
		body = append(body, buf...)
		count++
	})
	le.PutUint64(hdr[20:], uint64(count))
	cw.Write(hdr[20:28])
	cw.Write(body)
	cb := make([]byte, 4)
	le.PutUint32(cb, uint32(len(clocks)))
	cw.Write(cb)
	for w, c := range clocks {
		e := make([]byte, 12)
		le.PutUint32(e, uint32(int32(w)))
		le.PutUint64(e[4:], uint64(c))
		cw.Write(e)
	}
	le.PutUint32(cb, cw.crc)
	if _, err := bw.Write(cb); err != nil {
		return 0, err
	}
	if err := bw.Flush(); err != nil {
		return 0, err
	}
	if err := tmp.Sync(); err != nil {
		return 0, err
	}
	if err := tmp.Close(); err != nil {
		return 0, err
	}
	return count, os.Rename(tmp.Name(), path)
}

// LoadSnapshot reads a snapshot into st (which must have the same dim and
// optimizer) and returns the saved version and clocks.
func LoadSnapshot(path string, st *Store) (uint64, map[int]int64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, nil, err
	}
	if len(data) < 36 || string(data[:4]) != "PSS1" {
		return 0, nil, ErrCorrupt
	}
	le := binary.LittleEndian
	body, want := data[:len(data)-4], le.Uint32(data[len(data)-4:])
	if crc32.ChecksumIEEE(body) != want {
		return 0, nil, ErrCorrupt
	}
	dim, rowLen := int(le.Uint32(data[4:])), int(le.Uint32(data[8:]))
	if dim != st.dim || rowLen != st.rowLen() {
		return 0, nil, fmt.Errorf("ps: snapshot has dim %d rowLen %d, server wants %d %d", dim, rowLen, st.dim, st.rowLen())
	}
	version, n := le.Uint64(data[12:]), le.Uint64(data[20:])
	off, rec := 28, 8+4*rowLen
	if uint64(len(body)-off) < n*uint64(rec)+4 {
		return 0, nil, ErrCorrupt
	}
	row := make([]float32, rowLen)
	for i := uint64(0); i < n; i++ {
		k := le.Uint64(data[off:])
		for j := range row {
			row[j] = math.Float32frombits(le.Uint32(data[off+8+4*j:]))
		}
		st.Set(k, row)
		off += rec
	}
	nc := int(le.Uint32(data[off:]))
	off += 4
	if len(body)-off != 12*nc {
		return 0, nil, ErrCorrupt
	}
	clocks := make(map[int]int64, nc)
	for i := 0; i < nc; i++ {
		clocks[int(int32(le.Uint32(data[off:])))] = int64(le.Uint64(data[off+4:]))
		off += 12
	}
	return version, clocks, nil
}

func isNotExist(err error) bool { return errors.Is(err, os.ErrNotExist) }
