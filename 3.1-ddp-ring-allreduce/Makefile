PY ?= python3
RUN = PYTHONPATH=src $(PY) -m ringddp

.PHONY: test demo bench docker clean

test:
	PYTHONPATH=src $(PY) -m pytest -v

demo:
	$(RUN) train --workers 4 --epochs 3
	$(RUN) check --workers-list 1,2,4
	$(RUN) train --workers 4 --crash-rank 1 --crash-step 5 || true
	$(RUN) project

bench:
	mkdir -p results
	$(RUN) bench allreduce --workers-list 2,4 --repeat 3 --json results/allreduce.json
	$(RUN) bench scaling --workers-list 1,2,4 --epochs 1 --samples 16384 --batch 512 --hidden 512 --repeat 3 --json results/scaling.json
	$(RUN) bench overlap --workers 2 --epochs 1 --samples 8192 --batch 512 --hidden 512 --layers 6 --repeat 3 --json results/overlap.json

docker:
	docker build -t ringddp:latest .
	docker run --rm ringddp:latest check --workers-list 1,2

clean:
	rm -rf results .pytest_cache src/ringddp/__pycache__ tests/__pycache__
