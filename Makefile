# Convenience targets for building and testing canns-lib

.PHONY: build test compare clean

build:
	uv run maturin develop --release

test:
	uv run python -m pytest tests

compare:
	uv run python example/trajectory_comparison.py

clean:
	rm -rf dist build target __pycache__ */__pycache__
