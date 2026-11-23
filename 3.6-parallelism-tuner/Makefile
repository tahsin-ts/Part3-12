PY ?= python3
export PYTHONPATH := src

.PHONY: test demo bench validate figures docker

test:
	$(PY) -m pytest -q

demo:
	$(PY) -m ptuner search --model gpt3-13b --gpus 8 --batch 512 --top 5
	$(PY) -m ptuner bubble --p 4 8 --m 8 32

validate:
	$(PY) -m ptuner validate --loo

bench:
	$(PY) scripts/bench.py

figures:
	$(PY) scripts/make_figures.py

docker:
	docker build -t ptuner .
	docker run --rm ptuner validate
