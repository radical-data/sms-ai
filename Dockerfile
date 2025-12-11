# Python 3.12 to match .python-version
FROM python:3.12-slim

# Basic Python env hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Default DB location inside the container (overrides config.py default)
    DATABASE_URL=sqlite:////app/data/sms_ai.db

# Where the app lives in the container
WORKDIR /app

# System build tools for any dependencies that need compiling
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ensure data directory exists (Coolify will mount a volume here)
RUN mkdir -p /app/data

# Copy metadata + code
COPY pyproject.toml README.md ./
COPY src ./src
COPY static ./static
# NOTE: if/when a data/ directory is added to the repo (for glossary, etc.),
# add:  COPY data ./data

# Install the project and its dependencies from pyproject.toml
RUN pip install --upgrade pip \
    && pip install .

# The app listens on port 8000
EXPOSE 8000

# Run FastAPI app with uvicorn
CMD ["uvicorn", "sms_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
