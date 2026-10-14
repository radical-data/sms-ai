FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    DATABASE_URL=sqlite:////app/data/sms_ai.db \
    PROJECT_ROOT=/app \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
RUN mkdir -p /app/data

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY static ./static

RUN uv sync --locked --no-dev --no-editable

EXPOSE 8000
CMD ["uvicorn", "sms_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
