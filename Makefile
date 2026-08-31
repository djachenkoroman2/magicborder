.PHONY: format dev


format:
	uv run ruff check --fix
	uv run ruff format
	uv run bandit -c pyproject.toml -r src/magicborder

dev:
	uv run magicborder
