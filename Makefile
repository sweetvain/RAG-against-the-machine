.PHONY: install run debug clean lint lint-strict

install:
	uv sync

# uv run python -m src <command> [options]
run:
	uv run python -m src --help

# make debug ARGS="index --max_chunk_size 2000"
debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete
	rm -rf .mypy_cache .pytest_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
