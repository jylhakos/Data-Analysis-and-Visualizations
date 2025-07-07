from fastapi import FastAPI, Request
from fastapi.middleware.base import BaseHTTPMiddleware
import json
import time
from kafka import KafkaProducer
from datetime import datetime
import uvicorn

app = FastAPI()

# Kafka producer for sending HTTP request events
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

class HTTPEventMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Capture request details
        request_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", ""),
            "event_type": "http_request"
        }
        
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Capture response details
        response_event = {
            **request_event,
            "status_code": response.status_code,
            "processing_time_ms": round(process_time * 1000, 2),
            "event_type": "http_response"
        }
        
        # Send event to Kafka
        try:
            producer.send('http-events', value=response_event)
            producer.flush()
        except Exception as e:
            print(f"Failed to send event to Kafka: {e}")
        
        return response

app.add_middleware(HTTPEventMiddleware)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id, "name": f"User {user_id}"}

@app.post("/api/data")
async def create_data(data: dict):
    return {"status": "created", "data": data}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)