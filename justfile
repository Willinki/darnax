# justfile for development tasks
# Run with: `uv run just <task>`

set shell := ["bash", "-uc"]

# --- Dev commands ---

fmt:
    ruff format .
    black .

lint:
    ruff check .

fix:
    ruff format .
    black .
    ruff check . --fix

fix-unsafe:
    ruff format .
    black .
    ruff check . --fix --unsafe-fixes

type:
    mypy src

test:
    pytest -q

all:
    just fmt
    just lint
    just type
    just test

docs:
    mkdocs serve
