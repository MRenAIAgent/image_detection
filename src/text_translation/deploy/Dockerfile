FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && \ 
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        curl \
        g++ \
        build-essential \
        git \
        && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && \
    apt-get install -y google-cloud-sdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM base AS dependencies

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

FROM dependencies AS runtime

# Set environment variables
ENV PYTHONPATH=/app \
    DEBUG=0 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.huggingface \
    TRANSFORMERS_CACHE=/app/.huggingface \
    MODEL_CACHE_DIR=/app/models

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p models logs /app/.cache /app/.huggingface && \
    chmod 755 models logs /app/.cache /app/.huggingface

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8080

# Health check
HEALTHCHECK --interval=60s --timeout=60s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Command to run the application
WORKDIR /app
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1"]