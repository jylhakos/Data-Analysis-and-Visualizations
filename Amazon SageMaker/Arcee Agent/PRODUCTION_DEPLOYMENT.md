# Arcee Agent production deployment

The document shows you how to deploy and integrate your fine-tuned Arcee Agent model in production.

## Steps

### Option 1: Docker production deployment (Recommended)

```bash
# 1. Build and deploy the production stack
./deploy_production.sh build

# 2. Test the deployment
curl -X POST http://localhost/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "arcee-ai/Arcee-Agent",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the weather like today?"}
    ]
  }'

# 3. Use in your applications
python production_client_example.py
```

### Option 2: Direct API server

```bash
# Start the API server with vLLM (recommended)
python api_server.py --backend vllm --model arcee-ai/Arcee-Agent --port 8000

# Or with Ollama
python api_server.py --backend ollama --model arcee-agent --port 8000

# Test the API
python main.py --model arcee-ai/Arcee-Agent --base_url http://127.0.0.1:8000/v1
```

### Option 3: AWS SageMaker deployment

```bash
# 1. Configure AWS credentials
aws configure

# 2. Setup IAM roles and permissions
./scripts/setup_aws.sh

# 3. Deploy to SageMaker
./deploy_to_sagemaker.sh

# 4. Use SageMaker endpoint
python main.py --model arcee-ai/Arcee-Agent --base_url https://your-endpoint.amazonaws.com/v1
```

## Integration methods

### 1. OpenAI compatible client (Recommended)

```python
from openai import OpenAI

# Production deployment
client = OpenAI(
    base_url="http://localhost:8000/v1",  # or your production URL
    api_key="dummy"  # or your actual API key
)

# Function calling (same as your main.py)
response = client.chat.completions.create(
    model="arcee-ai/Arcee-Agent",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What tools should I use for weather?"}
    ],
    temperature=0.1,
    max_tokens=512
)
```

### 2. Production client class

```python
from production_client_example import ProductionArceeAgent

# Initialize for different deployments
client = ProductionArceeAgent(
    deployment_type="docker",  # "docker", "api", or "sagemaker" 
    base_url="http://localhost:8000/v1"
)

# Function calling
tools = [{"name": "get_weather", "description": "Get weather info"}]
result = client.function_call("What's the weather?", tools)
```

### 3. Direct HTTP integration

```python
import requests

def call_arcee_agent(query, tools):
    response = requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "arcee-ai/Arcee-Agent",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Query: {query}, Tools: {tools}"}
        ]
    })
    return response.json()
```

## Deployment architecture

### Docker production stack

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │────│  Arcee Agent API │────│   vLLM/Ollama   │
│  (Load Balancer)│    │     (FastAPI)    │    │    (Backend)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌──────────────────┐            │
         │              │  Redis (Cache)   │            │
         │              └──────────────────┘            │
         │                                              │
┌─────────────────┐              │              ┌─────────────────┐
│   Prometheus    │──────────────┼──────────────│    Grafana     │
│  (Metrics)      │              │              │ (Monitoring)   │
└─────────────────┘              │              └─────────────────┘
```

### Production URLs

- **API Endpoint**: `http://localhost:8000/v1` (or your domain)
- **Monitoring**: `http://localhost:3000` (Grafana)
- **Metrics**: `http://localhost:9090` (Prometheus)

## 🔧 Configuration

### Environment variables

```bash
# API Configuration
export ARCEE_MODEL="arcee-ai/Arcee-Agent"
export ARCEE_BACKEND="vllm"  # or "ollama"
export ARCEE_PORT="8000"
export ARCEE_HOST="0.0.0.0"

# Production Settings
export WORKERS=4
export MAX_CONCURRENT_REQUESTS=100
export TIMEOUT=30

# AWS Configuration (for SageMaker)
export AWS_REGION="us-east-1"
export SAGEMAKER_ENDPOINT_NAME="arcee-agent-endpoint"
```

### Docker Compose

Create `docker-compose.override.yml` for custom configurations:

```yaml
version: '3.8'
services:
  arcee-agent:
    environment:
      - MODEL_NAME=your-custom-model
      - GPU_MEMORY_UTILIZATION=0.9
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2  # Use 2 GPUs
```

## Monitoring & scaling

### Health checks

```bash
# Check API health
curl http://localhost:8000/health

# Check model status
curl http://localhost:8000/v1/models

# Monitor logs
docker-compose logs -f arcee-agent
```

### Scaling

```bash
# Scale API servers
docker-compose up --scale arcee-agent=3

# Use load balancer
./deploy_production.sh scale --replicas=5
```

### Metrics

Access Grafana at `http://localhost:3000` to monitor:
- Request throughput
- Response latency
- GPU utilization
- Memory usage
- Error rates

## 🔒 Security

### API authentication

```python
# Add authentication to your API calls
headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers=headers,
    json=payload
)
```

### Network security

```bash
# Use HTTPS in production
# Configure SSL certificates in docker-compose.prod.yml

# Restrict access
# Configure firewall rules
# Use VPN for internal APIs
```

## Testing

### Unit tests

```bash
# Run all tests
python -m pytest test_arcee_agent.py -v

# Test specific deployment
python test_api.py --base_url http://localhost:8000/v1
```

### Load testing

```bash
# Install testing tools
pip install locust

# Run load tests
locust -f load_test.py --host http://localhost:8000
```

### Integration tests

```bash
# Test end-to-end functionality
python integration_examples.py

# Test different backends
python main.py --max_samples 5 --base_url http://localhost:8000/v1
```

## Troubleshooting

### Issues

1. **Out of memory**
   ```bash
   # Reduce batch size or use smaller model
   export GPU_MEMORY_UTILIZATION=0.7
   ```

2. **Connection refused**
   ```bash
   # Check if service is running
   docker-compose ps
   curl http://localhost:8000/health
   ```

3. **Slow responses**
   ```bash
   # Check GPU utilization
   nvidia-smi
   
   # Monitor metrics
   docker stats
   ```

### Log analysis

```bash
# Check API logs
docker-compose logs arcee-agent

# Check system logs
journalctl -u docker

# Monitor resource usage
htop
```

