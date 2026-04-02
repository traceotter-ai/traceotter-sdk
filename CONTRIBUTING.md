# Contributing

Thanks for helping improve the TraceOtter Python SDK.

## Prerequisites

- **Python 3.11+** (see `requires-python` in `pyproject.toml`)
- **[uv](https://docs.astral.sh/uv/)** for installs and running tools

## Local setup

```bash
git clone <repo-url>
cd traceotter-python-sdk
uv sync --dev
```

## Before you open a PR

Run the same checks CI runs:

```bash
uv run ruff check
uv run ruff format --check
uv run pytest
```

Fix formatting and import issues with:

```bash
uv run ruff check --fix
uv run ruff format
```

## Commits

Use a short [Conventional Commits](https://www.conventionalcommits.org/) prefix when possible, for example `feat:`, `fix:`, `docs:`, `ci:`, `chore:`, `test:`.

## Pull requests

Use the PR template, keep changes focused, and ensure workflows pass on your branch.
