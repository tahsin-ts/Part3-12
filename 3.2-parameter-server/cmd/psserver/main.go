// Command psserver runs one shard of the parameter server.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/tahsin-ts/part3-12/3.2-parameter-server/internal/ps"
)

func main() {
	addr := flag.String("addr", ":7000", "listen address")
	id := flag.Int("id", 0, "shard index (0-based)")
	idHost := flag.Bool("id-from-hostname", false, "take the shard index from the hostname suffix (StatefulSet pod ps-2 -> 2)")
	shards := flag.Int("shards", 1, "total number of shards on the ring")
	vnodes := flag.Int("vnodes", 64, "virtual nodes per shard")
	workers := flag.Int("workers", 1, "number of workers (ids 0..n-1) the clocks start with")
	dim := flag.Int("dim", 1, "float32 values per key")
	lr := flag.Float64("lr", 0.1, "learning rate")
	opt := flag.String("opt", "adagrad", "optimizer: sgd or adagrad")
	snap := flag.String("snapshot", "", "snapshot file (loaded at start if present)")
	every := flag.Duration("snapshot-every", 0, "write a snapshot this often (0 = only on SIGTERM and POST /snapshot)")
	wto := flag.Duration("worker-timeout", 5*time.Second, "evict a worker after this long without a request")
	pullTO := flag.Duration("wait-timeout", 30*time.Second, "longest a pull may wait for slower workers")
	flag.Parse()

	if *idHost {
		h, _ := os.Hostname()
		n, err := strconv.Atoi(h[strings.LastIndex(h, "-")+1:])
		if err != nil {
			log.Fatalf("hostname %q has no numeric suffix", h)
		}
		*id = n
	}
	o := ps.Adagrad
	if *opt == "sgd" {
		o = ps.SGD
	}
	srv, err := ps.NewServer(ps.Config{ID: *id, Shards: *shards, VNodes: *vnodes, Workers: *workers, Dim: *dim,
		LR: float32(*lr), Opt: o, SnapshotPath: *snap, WaitTimeout: *pullTO})
	if err != nil {
		log.Fatal(err)
	}
	stop := make(chan struct{})
	go srv.EvictLoop(*wto, *wto/5, stop)
	if *every > 0 && *snap != "" {
		go func() {
			t := time.NewTicker(*every)
			for {
				select {
				case <-stop:
					return
				case <-t.C:
					if _, err := srv.Snapshot(); err != nil {
						log.Printf("snapshot: %v", err)
					}
				}
			}
		}()
	}
	hs := &http.Server{Addr: *addr, Handler: srv.Handler()}
	done := make(chan struct{})
	go func() {
		defer close(done)
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGTERM, syscall.SIGINT)
		<-sig
		close(stop)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		hs.Shutdown(ctx)
		if *snap != "" {
			n, err := srv.Snapshot()
			log.Printf("final snapshot: %d keys, err=%v", n, err)
		}
	}()
	fmt.Printf("shard %d/%d listening on %s (dim=%d opt=%s lr=%g, restored %d keys)\n", *id, *shards, *addr, *dim, *opt, *lr, srv.RestoredKeys())
	if err := hs.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatal(err)
	}
	<-done // the final snapshot is written before we exit
}
