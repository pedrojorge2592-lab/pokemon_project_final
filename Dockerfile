# Dockerfile
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (build tools for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Install deps
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install jupyterlab mlflow streamlit

# Copy source last (so rebuilds are faster when only code changes)
COPY . .

EXPOSE 8501 5000 8888

# Default command is a shell; docker-compose will set the processes
CMD ["/bin/bash"]
