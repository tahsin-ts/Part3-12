package ps

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func post(t *testing.T, url string, body []byte) (*http.Response, []byte) {
	t.Helper()
	r, err := http.Post(url, "application/octet-stream", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	b, _ := io.ReadAll(r.Body)
	return r, b
}

// keysFor returns n keys owned by shard id on a ring of `shards`.
func keysFor(id, shards, n int) []uint64 {
	r := NewRing(shards, 64)
	var out []uint64
	for k := uint64(0); len(out) < n; k++ {
		if r.Owner(k) == id {
			out = append(out, k)
		}
	}
	return out
}

func TestServerPushPullStatsAndRouting(t *testing.T) {
	srv, _ := NewServer(Config{ID: 1, Shards: 2, Workers: 2, Dim: 1, LR: 1, Opt: SGD})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	keys := keysFor(1, 2, 3)
	r, _ := post(t, ts.URL+"/push?w=0&c=1&v=0", EncodePush(keys, 1, []float32{1, 2, 3}))
	if r.StatusCode != 200 {
		t.Fatal(r.Status)
	}
	r, b := post(t, ts.URL+"/pull?w=-1", EncodeKeys(keys))
	_, vals, err := DecodeRows(b)
	if err != nil || vals[0] != -1 || vals[2] != -3 || r.Header.Get("X-Version") != "1" {
		t.Fatalf("pull: %v %v %s", vals, err, r.Header.Get("X-Version"))
	}
	// a push based on version 0 when version is already 1 has staleness 1
	post(t, ts.URL+"/push?w=1&c=1&v=0", EncodePush(keys[:1], 1, []float32{1}))
	wrong := keysFor(0, 2, 1)
	if r, _ := post(t, ts.URL+"/pull?w=-1", EncodeKeys(wrong)); r.StatusCode != http.StatusConflict {
		t.Fatalf("misrouted key accepted: %s", r.Status)
	}
	if r, _ := post(t, ts.URL+"/push", []byte{1, 2, 3}); r.StatusCode != http.StatusBadRequest {
		t.Fatalf("garbage accepted: %s", r.Status)
	}
	resp, _ := http.Get(ts.URL + "/stats")
	var st Stats
	json.NewDecoder(resp.Body).Decode(&st)
	if st.Keys != 3 || st.Pushes != 2 || st.Misrouted != 1 || st.StalenessHist["1"] != 1 || st.MinClock != 1 {
		t.Fatalf("stats: %+v", st)
	}
}

// Four workers run BSP over HTTP: with staleness 0, a pull at clock c must see
// all 4*c pushes of earlier clocks (it may also see up to 3 from clock c of
// faster workers), so the value read is in [-(4c+3), -4c].
func TestServerBSPOverHTTP(t *testing.T) {
	srv, _ := NewServer(Config{ID: 0, Shards: 1, Workers: 4, Dim: 1, LR: 1, Opt: SGD, WaitTimeout: 2 * time.Second})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	var wg sync.WaitGroup
	errs := make(chan string, 4)
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for c := 0; c < 20; c++ {
				_, b := post(t, ts.URL+"/pull?s=0&w="+itoa(w)+"&c="+itoa(c), EncodeKeys([]uint64{9}))
				_, v, err := DecodeRows(b)
				if err != nil {
					errs <- "worker " + itoa(w) + " clock " + itoa(c) + ": " + string(b)
					return
				}
				if v[0] > float32(-4*c) || v[0] < float32(-4*c-3) {
					errs <- "worker " + itoa(w) + " read a stale value at clock " + itoa(c)
					return
				}
				post(t, ts.URL+"/push?w="+itoa(w)+"&c="+itoa(c+1), EncodePush([]uint64{9}, 1, []float32{1}))
			}
		}(w)
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Fatal(e)
	}
}

func TestServerRestartFromSnapshot(t *testing.T) {
	path := filepath.Join(t.TempDir(), "s.snap")
	cfg := Config{ID: 0, Shards: 1, Workers: 1, Dim: 1, LR: 1, Opt: Adagrad, SnapshotPath: path}
	a, _ := NewServer(cfg)
	ts := httptest.NewServer(a.Handler())
	post(t, ts.URL+"/push?w=0&c=3", EncodePush([]uint64{5, 6}, 1, []float32{1, 1}))
	if r, _ := post(t, ts.URL+"/snapshot", nil); r.StatusCode != 200 {
		t.Fatal(r.Status)
	}
	ts.Close()
	b, err := NewServer(cfg) // the "restarted" process
	if err != nil || b.RestoredKeys() != 2 || b.version.Load() != 1 || b.clocks.Min() != 3 {
		t.Fatal(err, b.RestoredKeys(), b.version.Load(), b.clocks.Min())
	}
}

func itoa(i int) string { return string(rune('0'+i/10%10)) + string(rune('0'+i%10)) }
