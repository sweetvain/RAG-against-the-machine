.PHONY: install run debug clean lint lint-strict

install:
	uv sync


# affiche donc l'aide listant les commandes disponibles. Pour lancer une
# commande précise, voir le README (section "Example usage") ou taper
# directement : uv run python -m src <command> [options
run:
	uv run python -m src --help

# make debug ARGS="index --max_chunk_size 2000"
debug:
	uv run python -m pdb -m src $(ARGS)

# Supprime les fichiers et caches temporaires générés par Python/mypy
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -exec rm -f {} +
	rm -rf .mypy_cache
	rm -rf .pytest_cache

# Vérifie le respect des standards flake8 et mypy
lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

# Vérification renforcée avec mypy --strict
# lint-strict:
# 	uv run flake8 .
# 	uv run mypy . --strict