PY ?= python3
RUN = PYTHONPATH=src $(PY) -m streamload
DATA ?= /tmp/streamload-data

.PHONY: test demo bench starve docs docker clean

test:
	PYTHONPATH=src $(PY) -m pytest -v

demo:
	$(RUN) make-data $(DATA) --records 2000 --side 64 --per-shard 100
	$(RUN) verify $(DATA)
	$(RUN) read $(DATA) --workers 2 --batch 64
	$(RUN) read $(DATA) --workers 2 --batch 64 --world 2 --rank 1
	$(RUN) resume-check $(DATA) --workers 2 --batch 64 --stop-at 7

bench:
	mkdir -p results
	$(RUN) bench --out results/bench.json | tee results/bench.txt

starve:
	$(RUN) starve --per-core 300 --cores-per-gpu 12

docs:
	PYTHONPATH=src $(PY) scripts/make_docs.py

docker:
	docker build -t streamload:latest .
	docker run --rm --entrypoint sh streamload:latest -c "streamload make-data /data/ds --records 400 && streamload resume-check /data/ds --stop-at 3"

clean:
	rm -rf results .pytest_cache src/streamload/__pycache__ tests/__pycache__ $(DATA)
