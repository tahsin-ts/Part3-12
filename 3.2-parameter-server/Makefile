.PHONY: build test test-go test-py demo bench sim drills docker clean

build:
	go build -o bin/psserver ./cmd/psserver

test: test-go test-py

test-go:
	go vet ./...
	go test -race -count=1 ./...

test-py: build
	python -m pytest -q

demo: build
	python -m pskit train --mode ssp --staleness 2 --workers 4 --shards 2

bench: build
	python -m pskit bench --out bench.json
	python -m pskit sim
	python -m pskit ring
	python -m pskit compare

drills: build
	python -m pskit failover

docker:
	docker build -t paramserver:latest .

clean:
	rm -rf bin bench.json
