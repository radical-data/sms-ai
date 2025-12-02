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

