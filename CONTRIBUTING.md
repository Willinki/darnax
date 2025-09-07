# Contributing to BioNet

We welcome contributions from both core team members and external contributors.
This document describes our collaboration model, workflows, and expectations.

---

## 👩‍💻 Contribution Workflow

### Core developers (repo write access)
* Branch off `main` using short-lived branches:
  ```bash
    git switch -c feature/my-feature
  ```

* Open **draft PRs early** to trigger CI and gather feedback.
* Rebase regularly onto `main`:

  ```bash
  git fetch origin
  git rebase origin/main
  ```
* Merge policy:
  * **Fast-forward only** (linear history).
  * No merge commits into `main`.

* PRs require green CI + at least 1 CODEOWNER approval.

### External contributors (no write access)

1. **Fork** the repository on GitHub.
2. Clone your fork and set upstream:

   ```bash
   git clone git@github.com:<you>/<repo>.git
   cd <repo>
   git remote add upstream git@github.com:<org>/<repo>.git
   ```

3. Create a feature branch:

   ```bash
   git switch -c feature/my-change
   ```
4. Sync with upstream before opening a PR:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

5. Push to your fork and open a PR against `<org>/<repo>:main`.

---

## ✅ Pull Requests

* Keep PRs **small and focused** (< 500 LOC when possible).
* PRs must pass all checks (lint, types, tests, docs) before review.
* Update **docs/tutorials** if public API changes.
* Add **tests** for new functionality or bugfixes.
* Include **perf numbers** if touching hot paths.
* Add a **changelog entry** if user-visible.

---

## 📝 Commit Style

We use [Conventional Commits](https://www.conventionalcommits.org/).
Examples:

* `feat(orchestrator): add inference flag`
* `fix(state): correct gradient update`
* `docs: improve orchestrator tutorial`
* `perf(core): speed up jit compile`
* `chore: bump deps`

Squash or rebase messy commit histories before merge.

---

## 🔐 Security

Please report security issues privately via [SECURITY.md](./SECURITY.md).
Do not open public issues for vulnerabilities.

---

## 📜 License

By contributing, you agree that your contributions are licensed under the [GPLv3](./LICENSE).

