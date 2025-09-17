# 📄 `GETTING_STARTED.md`

# Getting Started with BioNet

This guide walks you through setting up your local development environment after cloning the repository.

---

## 1. Install uv

We use [uv](https://github.com/astral-sh/uv) to manage Python versions, virtual environments, and dependencies.

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Homebrew
brew install uv
```

## Install just

We also use [just](https://github.com/casey/just) to run simple scripts.

```bash
# macOS/Linux
brew install just
```

---

## 2. Clone the repository

```bash
git clone git@github.com:<org>/<repo>.git
cd <repo>
```

If you are working from a fork, clone your fork instead and add the upstream remote:

```bash
git remote add upstream git@github.com:<org>/<repo>.git
```

---

## 3. Set up Python and dependencies

```bash
uv python install 3.11   # install Python if missing
uv python pin 3.11       # pin project to Python 3.11
uv sync                  # install all dependencies into .venv
```

This creates a `.venv/` tied to the project.

---

## 3.5 Enforce rebase and ff-only commits for safety

```bash
# one-time (recommended) safety settings
git config --global pull.rebase true          # pulls rebase by default
git config --global merge.ff only             # disallow non-FF merges
git config --global rebase.autosquash true    # honors fixup!/squash! commits
git config --global rebase.autoStash true     # stash/unstash during rebase
```

---

## 4. Install pre-commit hooks

We use [pre-commit](https://pre-commit.com/) to enforce style and type checks.

```bash
uv run pre-commit install
uv run pre-commit run --all-files   # run checks once on the full repo
```

---

## 5. Run checks and tests

```bash
uv run all     # lint + format + type + test (script defined in pyproject)
uv run test    # run pytest
uv run lint    # run ruff
uv run fmt     # run ruff format + black
uv run type    # run mypy
```

---

## 6. Build and preview docs

Docs use [MkDocs](https://www.mkdocs.org/):

```bash
uv run mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 7. Typical development loop

```bash
# update from main branch
git pull --rebase origin main
uv sync                 # update deps if pyproject changed

# make changes...
uv run all              # run full check suite

# commit with Conventional Commit style
git add -p
git commit -m "feat(orchestrator): add inference flag"
git push origin feature/my-change
```

---

## 8. Submitting your work

* Push your branch (to the main repo if you have write access, or to your fork otherwise).
* Open a Pull Request against `<org>/<repo>:main`.
* Ensure CI passes and address review feedback.

---
