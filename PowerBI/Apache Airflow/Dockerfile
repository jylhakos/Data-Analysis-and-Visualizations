# Use official Python runtime as base image
FROM python:3.9-slim

# Set environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV PYTHONPATH="${PYTHONPATH}:${AIRFLOW_HOME}"

# Set work directory
WORKDIR /opt/airflow

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p ${AIRFLOW_HOME}/dags \
    ${AIRFLOW_HOME}/logs \
    ${AIRFLOW_HOME}/plugins \
    ${AIRFLOW_HOME}/ml_data \
    ${AIRFLOW_HOME}/ml_models

# Initialize Airflow database
RUN airflow db init

# Create admin user
RUN airflow users create \
    --username admin \
    --firstname ML \
    --lastname Admin \
    --role Admin \
    --email admin@mlproject.com \
    --password admin

# Expose port
EXPOSE 8080

# Command to run Airflow webserver
CMD ["airflow", "webserver", "--port", "8080"]
