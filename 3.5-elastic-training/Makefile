PY ?= python3

.PHONY: test demo bench docs clean

test:
	$(PY) -m pytest -q

demo:
	$(PY) -m elastic demo --scenario kill-node
	$(PY) -m elastic demo --scenario pause

bench:
	$(PY) -m elastic bench --out bench_results.json

docs:
	$(PY) scripts/make_figures.py

clean:
	rm -rf /tmp/elastic-demo /tmp/elastic-bench .pytest_cache
