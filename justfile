# justfile for development tasks
# Run with: `uv run just <task>`

set shell := ["bash", "-uc"]

# --- Dev commands ---

fmt:
    ruff format .
    black .

lint:
    ruff check .

type:
    mypy src tests

test:
    pytest -q

all:
    just fmt
    just lint
    just type
    just test

docs:
    mkdocs serve

