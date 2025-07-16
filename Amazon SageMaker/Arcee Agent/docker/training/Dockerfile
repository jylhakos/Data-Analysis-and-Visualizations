# SageMaker Training Dockerfile for Arcee Agent Fine-tuning
FROM huggingface/transformers-pytorch-gpu:4.30.0-torch2.0.0-cu118-ubuntu20.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for training
RUN pip install --no-cache-dir \
    datasets==2.14.0 \
    peft==0.4.0 \
    bitsandbytes==0.39.1 \
    accelerate==0.21.0 \
    trl==0.5.0 \
    wandb \
    scipy

# Copy training scripts
COPY fine_tune_arcee.py /opt/ml/code/train.py
COPY main.py /opt/ml/code/main.py
COPY requirements.txt /opt/ml/code/requirements.txt

# Set working directory
WORKDIR /opt/ml/code

# Define the training entrypoint
ENV SAGEMAKER_PROGRAM train.py
