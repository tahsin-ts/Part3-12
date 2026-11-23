PY ?= python3
RUN = PYTHONPATH=src $(PY) -m threed

.PHONY: test demo bench plan docs docker clean

test:
	PYTHONPATH=src $(PY) -m pytest -v

demo:
	$(RUN) demo
	$(RUN) failover --grid 2x2x2 --steps 6 --kill-rank 5 --kill-step 3

bench:
	mkdir -p results
	$(RUN) pipeline --stages 4 --micro 8 --measure | tee results/pipeline.txt
	$(RUN) train --grid 2x2x2 --steps 12 | tee results/train_2x2x2.txt
	$(RUN) plan --hw h100 --gpus 64 128 256 512 1024 --explain 1024 | tee results/plan_h100.txt

plan:
	$(RUN) plan --hw h100 --explain 1024
	$(RUN) plan --hw a100 --gpus 512 1024 2048

docs:
	PYTHONPATH=src $(PY) scripts/make_docs.py

docker:
	docker build -t threed:latest .
	docker run --rm threed:latest train --grid 2x2x2 --steps 4

clean:
	rm -rf results .pytest_cache src/threed/__pycache__ tests/__pycache__
