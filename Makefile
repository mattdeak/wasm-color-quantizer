.PHONY: build serve bench

PYTHON_ENV = .venv
ROOT_DIR := $(shell pwd)

build:
	cd rust && wasm-pack build --target web --profiling
	cd ..
	rm -rf pkg
	mv rust/pkg .
	rm -f pkg/.gitignore pkg/*.ts

build-release:
	cd rust && wasm-pack build --target web --release
	cd ..
	rm -rf pkg
	mv rust/pkg .
	rm -f pkg/.gitignore pkg/*.ts

analysis-env:
	if ! [ -d $(PYTHON_ENV) ]; then \
		python3 -m venv $(PYTHON_ENV) ; \
	fi
	$(PYTHON_ENV)/bin/python -m pip install scipy colorama

bench: analysis-env
	$(eval PREVIOUS_BENCHMARK := $(shell ls -t benchmark_history/ | head -n 1))
	$(eval NEW_BENCHMARK := bench_$(shell date +%s).txt)
	@echo "Benchmarking into $(NEW_BENCHMARK)"
	cd rust && cargo wasi bench --profile release --features gpu >> $(ROOT_DIR)/benchmark_history/$(NEW_BENCHMARK)
	@echo "Comparing $(PREVIOUS_BENCHMARK) to $(NEW_BENCHMARK)"
	cd $(ROOT_DIR) && $(PYTHON_ENV)/bin/python3 scripts/compare_benchmarks.py benchmark_history/$(PREVIOUS_BENCHMARK) benchmark_history/$(NEW_BENCHMARK)

bench-compare: analysis-env
	$(eval MOST_RECENT_BENCHMARK := $(shell ls -t benchmark_history | head -n 1))
	$(eval PREVIOUS_BENCHMARK := $(shell ls -t benchmark_history | head -n 2 | tail -n 1))
	$(PYTHON_ENV)/bin/python3 scripts/compare_benchmarks.py benchmark_history/$(PREVIOUS_BENCHMARK) benchmark_history/$(MOST_RECENT_BENCHMARK)


serve-profiling: build
	python3 -m http.server 8000

serve-release: build-release
	python3 -m http.server 8000

all: serve-release