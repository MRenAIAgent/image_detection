FROM python:3.11-slim AS base

# Install dependencies
RUN apt-get update && \ 
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends apt-transport-https ca-certificates gnupg curl g++ build-essential && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && \
    apt-get install -y google-cloud-sdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM base AS dependencies1

WORKDIR /

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM dependencies1 AS runtime

# Set model environment variables with build args
ARG MODEL_BUCKET="model_bin"
ARG MODEL_PATH="fasttext"
ARG MODEL_FILENAME="lid.176.bin"

ENV MODEL_BUCKET=${MODEL_BUCKET}
ENV MODEL_PATH=${MODEL_PATH}
ENV MODEL_FILENAME=${MODEL_FILENAME}

# Copy application code
COPY . .

# Create the models directory
RUN mkdir -p models

# Download the model directly to the correct relative path
RUN echo "Downloading model from gs://${MODEL_BUCKET}/${MODEL_PATH}" && \
    gsutil cp gs://${MODEL_BUCKET}/${MODEL_PATH}/${MODEL_FILENAME} models/${MODEL_FILENAME} || \
    echo "Warning: Failed to download model. You may need to authenticate or check the GCS path."

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# Calculate the number of workers based on CPU count at runtime
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --workers $(($(nproc) * 2 + 1))"]
