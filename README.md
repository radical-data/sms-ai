# sms-ai

## Development

### Prerequisites

- uv

### Setup

Install dependencies and dev tools:
```bash
uv sync --all-extras --dev
```

### Commands

```bash
# Install (or refresh) deps + dev tooling
uv sync --all-extras --dev

# Lint only
uv run ruff check .

# Autoformat
uv run ruff format .

# Type-check
uv run pyright

# Run tests
uv run pytest

# Run app (for local dev)
uv run uvicorn sms_ai.main:app --reload
```

### Pre-commit

Install pre-commit hooks:
```bash
uv run pre-commit install
```

This will automatically run ruff and pyright on staged files before commits.


## Dependency updates

Dependabot checks Python dependencies (`uv.lock`), GitHub Actions and Docker
images weekly. Patch and minor Python/Actions updates are grouped to reduce PR
noise. Patch and minor updates enable GitHub auto-merge; major updates need review.
GitHub must have auto-merge enabled and require both `test` and `container` checks
on `main`. These settings are managed in the repository settings, not this file.

CI and Docker install the committed lockfile with `uv sync --locked`. To update
manually, run `uv lock --upgrade`, then run the CI checks before committing the
lockfile. Dependabot also updates the pinned uv image used by Docker.

Offline tests use a temporary database and simulated model responses; they do not
require API keys. The live smoke test is opt-in: run `RUN_LIVE_TESTS=1 uv run pytest` with
`GOOGLE_API_KEY` and `TAVILY_API_KEY` configured. It also uses a temporary database. Offline tests cannot detect model
retirements or provider outages. Pre-commit runs Ruff and Pyright from the same lockfile as CI, so there are no
separate hook versions to maintain.
