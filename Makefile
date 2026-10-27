.PHONY: install run debug clean lint lint-strict

UV_CACHE_DIR ?= /tmp/rag-against-the-machine-uv-cache
export UV_CACHE_DIR

install:
	mkdir -p "$(UV_CACHE_DIR)"
	uv sync


# uv run python -m src <command> [options]
run:
	uv run python -m src --help

# make debug ARGS="index --max_chunk_size 2000"
debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -exec rm -f {} +
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf "$(UV_CACHE_DIR)"

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs