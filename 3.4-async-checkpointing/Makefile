PY ?= python3

.PHONY: test demo bench crash plan docs
test:
	$(PY) -m pytest -q
demo:
	$(PY) -m ackpt demo
bench:
	OMP_NUM_THREADS=1 $(PY) -m ackpt bench
crash:
	$(PY) -m ackpt crash-test
plan:
	$(PY) -m ackpt plan --params 7e9 --nodes 1
	$(PY) -m ackpt plan --params 70e9 --nodes 8
docs:
	$(PY) docs/make_figures.py
