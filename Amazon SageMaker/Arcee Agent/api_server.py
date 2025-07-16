#!/usr/bin/env python3
"""
FastAPI service for Arcee Agent function calling

Provides RESTful API endpoints for client requests to interact with
fine-tuned Arcee Agent model deployed on SageMaker.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import logging
import os
import asyncio
from datetime import datetime
import uuid

# Import our modules
from main import create_function_calling_prompt, parse_tool_calls
from sagemaker_inference import SageMakerInference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Arcee Agent Function Calling API",
    description="RESTful API for function calling with fine-tuned Arcee Agent model on AWS SageMaker",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SageMaker inference
try:
    sagemaker_inference = SageMakerInference()
    logger.info("SageMaker inference client initialized")
except Exception as e:
    logger.error(f"Failed to initialize SageMaker client: {e}")
    sagemaker_inference = None

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class FunctionCallRequest(BaseModel):
    query: str
    tools: List[ToolDefinition]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 512
    request_id: Optional[str] = None

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class FunctionCallResponse(BaseModel):
    request_id: str
    query: str
    tool_calls: List[ToolCall]
    processing_time: float
    model_response: str
    timestamp: str
    status: str

class BatchFunctionCallRequest(BaseModel):
    requests: List[FunctionCallRequest]
    batch_id: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Arcee Agent Function Calling API",
        "status": "healthy",
        "version": "1.0.0",
        "description": "RESTful API for function calling with Arcee Agent",
        "endpoints": {
            "health": "/health",
            "function_call": "/function-call",
            "batch_function_call": "/batch-function-call",
            "models": "/models",
            "docs": "/docs"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check SageMaker client availability
        sagemaker_healthy = sagemaker_inference is not None
        
        # Check if endpoint is accessible (if configured)
        endpoint_status = "unknown"
        if sagemaker_inference and sagemaker_inference.endpoint_name:
            try:
                endpoint_valid = sagemaker_inference.validate_endpoint()
                endpoint_status = "healthy" if endpoint_valid else "unhealthy"
            except Exception as e:
                endpoint_status = f"error: {str(e)}"
        
        status = "healthy" if sagemaker_healthy else "degraded"
        
        return {
            "status": status,
            "sagemaker_client": "healthy" if sagemaker_healthy else "unhealthy",
            "endpoint_status": endpoint_status,
            "endpoint_name": sagemaker_inference.endpoint_name if sagemaker_inference else None,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
async def health_check():
    """Detailed health check including SageMaker endpoint status."""
    try:
        health_info = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "sagemaker_endpoint": "unknown"
        }
        
        if sagemaker_inference:
            endpoint_status = await sagemaker_inference.check_endpoint_status()
            health_info["sagemaker_endpoint"] = endpoint_status
            
            if endpoint_status != "InService":
                health_info["status"] = "degraded"
                health_info["warning"] = "SageMaker endpoint not in service"
        else:
            health_info["status"] = "degraded"
            health_info["error"] = "SageMaker client not initialized"
        
        return health_info
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/function-call", response_model=FunctionCallResponse)
async def function_call(request: FunctionCallRequest):
    """
    Process a function calling request with Arcee Agent model.
    
    Args:
        request: Function call request with query and available tools
        
    Returns:
        Function call response with extracted tool calls
    """
    start_time = datetime.utcnow()
    request_id = request.request_id or str(uuid.uuid4())
    
    logger.info(f"Processing function call request {request_id}")
    
    if not sagemaker_inference:
        raise HTTPException(status_code=503, detail="SageMaker inference not available")
    
    try:
        # Convert tools to the expected format
        tools_list = [tool.dict() for tool in request.tools]
        
        # Create function calling prompt
        prompt = create_function_calling_prompt(request.query, tools_list)
        logger.debug(f"Generated prompt for {request_id}: {prompt[:200]}...")
        
        # Call SageMaker endpoint
        model_response = await sagemaker_inference.predict(
            prompt=prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        logger.debug(f"Model response for {request_id}: {model_response}")
        
        # Parse tool calls from response
        tool_calls_raw = parse_tool_calls(model_response)
        
        # Convert to Pydantic models
        tool_calls = [ToolCall(**call) for call in tool_calls_raw]
        
        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Completed request {request_id} in {processing_time:.2f}s, found {len(tool_calls)} tool calls")
        
        return FunctionCallResponse(
            request_id=request_id,
            query=request.query,
            tool_calls=tool_calls,
            processing_time=processing_time,
            model_response=model_response,
            timestamp=end_time.isoformat(),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error processing function call {request_id}: {e}")
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        return FunctionCallResponse(
            request_id=request_id,
            query=request.query,
            tool_calls=[],
            processing_time=processing_time,
            model_response="",
            timestamp=end_time.isoformat(),
            status=f"error: {str(e)}"
        )

@app.post("/batch-function-call")
async def batch_function_call(batch_request: BatchFunctionCallRequest):
    """
    Process multiple function calling requests in batch.
    
    Args:
        batch_request: Batch of function call requests
        
    Returns:
        Batch processing results
    """
    batch_id = batch_request.batch_id or str(uuid.uuid4())
    requests = batch_request.requests
    
    if len(requests) > 20:  # Limit batch size for performance
        raise HTTPException(
            status_code=400, 
            detail="Batch size limited to 20 requests. Use multiple batches for larger datasets."
        )
    
    logger.info(f"Processing batch {batch_id} with {len(requests)} requests")
    
    try:
        # Process all requests concurrently
        tasks = []
        for i, req in enumerate(requests):
            req.request_id = f"{batch_id}-{i}"
            tasks.append(function_call(req))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        processed_results = []
        error_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                error_count += 1
                processed_results.append({
                    "status": "error",
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        logger.info(f"Batch {batch_id} completed: {len(results) - error_count}/{len(results)} successful")
        
        return {
            "batch_id": batch_id,
            "batch_size": len(requests),
            "successful_requests": len(results) - error_count,
            "failed_requests": error_count,
            "results": processed_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch processing error for {batch_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

@app.get("/models")
async def list_models():
    """List available models and their status."""
    endpoint_name = os.getenv("SAGEMAKER_ENDPOINT_NAME", "arcee-agent-endpoint")
    
    if sagemaker_inference:
        endpoint_status = await sagemaker_inference.check_endpoint_status()
    else:
        endpoint_status = "unavailable"
    
    return {
        "models": [
            {
                "id": "arcee-agent-fine-tuned",
                "name": "Fine-tuned Arcee Agent",
                "description": "Arcee Agent model fine-tuned on function calling dataset",
                "endpoint": endpoint_name,
                "status": endpoint_status,
                "capabilities": ["function_calling", "tool_use"],
                "version": "1.0.0"
            }
        ],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Get basic API metrics."""
    # In production, you'd integrate with CloudWatch or similar
    return {
        "api_version": "1.0.0",
        "uptime": "Available via CloudWatch",
        "requests_processed": "Available via CloudWatch",
        "average_response_time": "Available via CloudWatch",
        "error_rate": "Available via CloudWatch",
        "note": "Detailed metrics available in AWS CloudWatch",
        "timestamp": datetime.utcnow().isoformat()
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested endpoint was not found",
        "available_endpoints": [
            "/", "/health", "/function-call", "/batch-function-call", "/models", "/docs"
        ]
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Arcee Agent API server on {host}:{port}")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port, 
        log_level="info",
        access_log=True
    )
