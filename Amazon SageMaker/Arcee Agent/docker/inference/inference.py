#!/usr/bin/env python3
"""
SageMaker Inference Script for Arcee Agent

This script handles model loading and inference for SageMaker endpoints.
It follows the SageMaker inference protocol with model_fn, input_fn, 
predict_fn, and output_fn functions.
"""

import json
import logging
import os
import sys
import torch
from typing import Dict, Any, List

# Add current directory to path for imports
sys.path.append('/opt/ml/code')

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel, PeftConfig
import main

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

def model_fn(model_dir: str):
    """
    Load the model for inference.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Loaded model and tokenizer
    """
    global model, tokenizer, device
    
    try:
        logger.info(f"Loading model from {model_dir}")
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load base model configuration
        base_model_name = "arcee-ai/Arcee-Agent"
        
        # Configure quantization for GPU inference
        if device.type == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = None
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True
        )
        
        # Check if LoRA adapters exist
        adapter_path = os.path.join(model_dir, "adapter_model.bin")
        if os.path.exists(adapter_path):
            logger.info("Loading LoRA adapters...")
            model = PeftModel.from_pretrained(base_model, model_dir)
            model = model.merge_and_unload()  # Merge adapters for inference
        else:
            logger.info("No LoRA adapters found, using base model")
            model = base_model
        
        # Set model to evaluation mode
        model.eval()
        
        logger.info("Model loaded successfully")
        return {"model": model, "tokenizer": tokenizer, "device": device}
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def input_fn(request_body: str, request_content_type: str = "application/json") -> Dict[str, Any]:
    """
    Parse input data for inference.
    
    Args:
        request_body: Raw request body
        request_content_type: Content type of the request
        
    Returns:
        Parsed input data
    """
    try:
        if request_content_type == "application/json":
            input_data = json.loads(request_body)
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
        
        logger.info(f"Parsed input: {input_data}")
        return input_data
        
    except Exception as e:
        logger.error(f"Error parsing input: {e}")
        raise

def predict_fn(input_data: Dict[str, Any], model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform inference with the loaded model.
    
    Args:
        input_data: Parsed input data
        model_dict: Model, tokenizer, and device
        
    Returns:
        Inference results
    """
    try:
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        device = model_dict["device"]
        
        # Extract input parameters
        query = input_data.get("inputs", input_data.get("query", ""))
        tools = input_data.get("tools", [])
        parameters = input_data.get("parameters", {})
        
        # Default parameters
        max_new_tokens = parameters.get("max_new_tokens", 512)
        temperature = parameters.get("temperature", 0.1)
        top_p = parameters.get("top_p", 0.9)
        do_sample = parameters.get("do_sample", temperature > 0)
        
        logger.info(f"Performing inference for query: {query[:100]}...")
        
        # Create function calling prompt
        if tools:
            prompt = main.create_function_calling_prompt(query, tools)
        else:
            prompt = query
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        )
        
        # Move to device
        if device.type == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode response
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Parse function calls if tools were provided
        function_calls = []
        if tools:
            try:
                function_calls = main.parse_tool_calls(generated_text)
            except Exception as e:
                logger.warning(f"Failed to parse function calls: {e}")
        
        # Calculate confidence score (average of token probabilities)
        scores = torch.softmax(torch.stack(outputs.scores), dim=-1)
        confidence = torch.mean(torch.max(scores, dim=-1)[0]).item()
        
        result = {
            "generated_text": generated_text,
            "function_calls": function_calls,
            "confidence": confidence,
            "metadata": {
                "model": "arcee-ai/Arcee-Agent",
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
        }
        
        logger.info("Inference completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

def output_fn(prediction: Dict[str, Any], accept: str = "application/json") -> str:
    """
    Format prediction output.
    
    Args:
        prediction: Model prediction results
        accept: Accept header for response format
        
    Returns:
        Formatted output
    """
    try:
        if accept == "application/json":
            return json.dumps(prediction, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported accept type: {accept}")
            
    except Exception as e:
        logger.error(f"Error formatting output: {e}")
        raise

# Alternative: Single function interface for simpler deployments
def handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """
    Alternative handler function for Lambda-style invocation.
    
    Args:
        event: Event data containing the request
        context: Lambda context (not used)
        
    Returns:
        Response data
    """
    try:
        # Load model if not already loaded
        if model is None:
            model_dict = model_fn("/opt/ml/model")
        else:
            model_dict = {"model": model, "tokenizer": tokenizer, "device": device}
        
        # Parse input
        input_data = event
        
        # Perform inference
        prediction = predict_fn(input_data, model_dict)
        
        return prediction
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"error": str(e)}

# Health check endpoint
def health_check() -> Dict[str, str]:
    """
    Health check for the inference service.
    
    Returns:
        Health status
    """
    try:
        if model is not None and tokenizer is not None:
            return {"status": "healthy", "model_loaded": True}
        else:
            return {"status": "unhealthy", "model_loaded": False}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# For testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SageMaker inference script")
    parser.add_argument("--model-dir", default="/opt/ml/model", help="Model directory")
    parser.add_argument("--test-query", default="What is the weather like?", help="Test query")
    
    args = parser.parse_args()
    
    # Test the inference script
    print("Loading model...")
    model_dict = model_fn(args.model_dir)
    
    # Test input
    test_input = {
        "inputs": args.test_query,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "Location name"}
                    },
                    "required": ["location"]
                }
            }
        ],
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.1
        }
    }
    
    print("Running inference...")
    result = predict_fn(test_input, model_dict)
    
    print("Result:")
    print(json.dumps(result, indent=2))
