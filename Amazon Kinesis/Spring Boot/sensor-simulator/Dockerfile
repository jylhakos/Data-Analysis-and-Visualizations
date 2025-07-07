FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the simulator code
COPY simulator.py .

# Run the simulator
CMD ["python", "simulator.py"]
