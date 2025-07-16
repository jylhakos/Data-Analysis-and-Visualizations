# SageMaker Inference Dockerfile for Arcee Agent
FROM huggingface/transformers-pytorch-gpu:4.30.0-torch2.0.0-cu118-ubuntu20.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for inference
RUN pip install --no-cache-dir \
    torch==2.0.0 \
    transformers==4.30.0 \
    datasets==2.14.0 \
    peft==0.4.0 \
    bitsandbytes==0.39.1 \
    accelerate==0.21.0 \
    flask==2.3.2 \
    gunicorn==21.2.0 \
    multi-model-server==1.1.11

# Copy inference code
COPY docker/inference/inference.py /opt/ml/code/inference.py
COPY main.py /opt/ml/code/main.py
COPY sagemaker_inference.py /opt/ml/code/sagemaker_inference.py

# Set working directory
WORKDIR /opt/ml/code

# Define the inference entrypoint
ENV SAGEMAKER_PROGRAM inference.py
