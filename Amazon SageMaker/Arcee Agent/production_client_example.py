#!/usr/bin/env python3
"""
Production Arcee Agent Client Integration Example

This demonstrates the recommended way to integrate your deployed Arcee Agent
model into production applications using OpenAI-compatible methods.
"""

from openai import OpenAI
import json
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionArceeAgent:
    """
    Production-ready Arcee Agent client for function calling
    """
    
    def __init__(
        self, 
        deployment_type: str = "docker",  # "docker", "api", or "sagemaker"
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy",
        model: str = "arcee-ai/Arcee-Agent"
    ):
        """
        Initialize the production client
        
        Args:
            deployment_type: Type of deployment
            base_url: API endpoint URL
            api_key: API key (dummy for local deployments)
            model: Model name
        """
        self.deployment_type = deployment_type
        self.model = model
        
        # Initialize OpenAI client (works with vLLM, Ollama, and SageMaker)
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=30
        )
        
        logger.info(f"Initialized Arcee Agent client: {deployment_type} @ {base_url}")
    
    def function_call(
        self, 
        query: str, 
        tools: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 512
    ) -> List[Dict[str, Any]]:
        """
        Perform function calling with the Arcee Agent model
        
        Args:
            query: User query
            tools: Available tools/functions
            temperature: Model temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of tool calls
        """
        try:
            # Create function calling prompt (same as your main.py)
            prompt = self._create_function_calling_prompt(query, tools)
            
            # Call the model
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that excels at function calling. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            tool_calls = self._parse_tool_calls(response_text)
            
            logger.info(f"Generated {len(tool_calls)} tool calls for query: {query[:50]}...")
            return tool_calls
            
        except Exception as e:
            logger.error(f"Function call failed: {e}")
            return []
    
    def _create_function_calling_prompt(self, query: str, tools: List[Dict[str, Any]]) -> str:
        """Create function calling prompt (same as main.py)"""
        tools_str = json.dumps(tools, indent=2)
        
        prompt = f"""You are an AI assistant with access to the following tools:

{tools_str}

Based on the user's query, determine which tool(s) to call and with what arguments.
Your response should be a JSON array of tool calls in the format:
[{{"name": "tool_name", "arguments": {{"param": "value"}}}}]

User Query: {query}

Tool Calls:"""
        
        return prompt
    
    def _parse_tool_calls(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from response (same as main.py)"""
        try:
            # Find JSON in the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                # Look for single tool call format
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    tool_call = json.loads(json_str)
                    return [tool_call]
                return []
                
            json_str = response_text[start_idx:end_idx]
            tool_calls = json.loads(json_str)
            
            # Validate the format
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict) or 'name' not in call:
                        logger.warning(f"Invalid tool call format: {call}")
                        return []
                return tool_calls
            elif isinstance(tool_calls, dict) and 'name' in tool_calls:
                return [tool_calls]
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            logger.warning(f"Response text: {response_text}")
        except Exception as e:
            logger.warning(f"Unexpected error parsing tool calls: {e}")
        
        return []

# =============================================================================
# Usage Examples
# =============================================================================

def example_usage():
    """Demonstrate different deployment integrations"""
    
    # Example tools (same format as your dataset)
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state or country"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        },
        {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object", 
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    ]
    
    # 1. Docker Deployment (Production)
    print("=== Docker Deployment Example ===")
    docker_client = ProductionArceeAgent(
        deployment_type="docker",
        base_url="http://localhost:8000/v1"
    )
    
    query = "What's the weather in New York and calculate 25 * 4?"
    tool_calls = docker_client.function_call(query, tools)
    print(f"Tool calls: {json.dumps(tool_calls, indent=2)}")
    
    # 2. Direct API Server
    print("\n=== Direct API Server Example ===")
    api_client = ProductionArceeAgent(
        deployment_type="api",
        base_url="http://127.0.0.1:8000/v1"
    )
    
    query = "Calculate the square root of 144"
    tool_calls = api_client.function_call(query, tools)
    print(f"Tool calls: {json.dumps(tool_calls, indent=2)}")
    
    # 3. AWS SageMaker Deployment
    print("\n=== SageMaker Deployment Example ===")
    sagemaker_client = ProductionArceeAgent(
        deployment_type="sagemaker",
        base_url="https://your-sagemaker-endpoint.amazonaws.com/v1"
    )
    
    query = "Get weather for London in celsius"
    tool_calls = sagemaker_client.function_call(query, tools)
    print(f"Tool calls: {json.dumps(tool_calls, indent=2)}")

def integrate_with_existing_main():
    """Show how to integrate with your existing main.py"""
    
    print("=== Integration with Existing main.py ===")
    
    # Your main.py already has the perfect structure!
    # Just change the client initialization:
    
    # OLD (from your main.py):
    # client = OpenAI(base_url=base_url, api_key=api_key)
    
    # NEW (production-ready):
    production_client = ProductionArceeAgent(
        deployment_type="docker",  # or "api" or "sagemaker"
        base_url="http://localhost:8000/v1"
    )
    
    # Your existing process_dataset function will work perfectly
    # Just replace the OpenAI client calls with production_client.function_call()
    
    print("✅ Your main.py is already production-ready!")
    print("✅ Just swap the client initialization for different deployments")

if __name__ == "__main__":
    # Run examples (make sure your API server is running first)
    try:
        example_usage()
        integrate_with_existing_main()
    except Exception as e:
        logger.error(f"Examples failed: {e}")
        print("💡 Make sure your API server is running:")
        print("   Docker: ./deploy_production.sh build")
        print("   Direct: python api_server.py --backend vllm --model arcee-ai/Arcee-Agent")
