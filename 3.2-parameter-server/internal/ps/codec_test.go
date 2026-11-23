package ps

import (
	"math/rand"
	"testing"
)

func TestCodecRoundTrip(t *testing.T) {
	keys := []uint64{0, 1, 1 << 63, 42}
	got, err := DecodeKeys(EncodeKeys(keys))
	if err != nil || len(got) != 4 || got[2] != 1<<63 {
		t.Fatalf("keys: %v %v", got, err)
	}
	vals := []float32{1.5, -2, 0, 3.25, 7, 8, 9, 10}
	d, v, err := DecodeRows(EncodeRows(2, vals))
	if err != nil || d != 2 || len(v) != 8 || v[3] != 3.25 {
		t.Fatalf("rows: %d %v %v", d, v, err)
	}
	k2, d2, g2, err := DecodePush(EncodePush(keys, 2, vals))
	if err != nil || d2 != 2 || k2[3] != 42 || g2[7] != 10 {
		t.Fatalf("push: %v %d %v %v", k2, d2, g2, err)
	}
	if len(EncodePush(keys, 2, vals)) != 8+8*4+4*8 {
		t.Fatal("push size")
	}
}

func TestCodecRejectsMalformed(t *testing.T) {
	b := EncodePush([]uint64{1, 2}, 1, []float32{1, 2})
	for _, bad := range [][]byte{nil, b[:3], b[:len(b)-1], append(append([]byte{}, b...), 0)} {
		if _, _, _, err := DecodePush(bad); err == nil {
			t.Fatalf("accepted %d bytes", len(bad))
		}
	}
	huge := EncodeKeys(nil)
	huge[3] = 0xff // n = 4 billion: must fail before allocating
	if _, err := DecodeKeys(huge); err == nil {
		t.Fatal("accepted huge n")
	}
	r := rand.New(rand.NewSource(1))
	for i := 0; i < 2000; i++ { // random garbage never panics
		g := make([]byte, r.Intn(64))
		r.Read(g)
		DecodeKeys(g)
		DecodeRows(g)
		DecodePush(g)
	}
}
