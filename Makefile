.PHONY: format dev test coverage


format:
	uv run ruff check --fix
	uv run ruff format
	uv run bandit -c pyproject.toml -r src/magicborder

test:
	uv run pytest

coverage:
	uv run pytest --cov --cov-report=term-missing --cov-report=html

dev:
	uv run magicborder
