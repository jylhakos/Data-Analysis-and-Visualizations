#!/usr/bin/env python3
"""
Production Arcee Agent Client

This module demonstrates the proper way to integrate and call the fine-tuned
Arcee Agent model in production using OpenAI-compatible methods.

Usage Examples:
    # Local Docker deployment
    client = ArceeAgentClient(base_url="http://localhost:8000/v1")
    
    # Production deployment
    client = ArceeAgentClient(base_url="https://your-api-gateway-url/v1")
    
    # SageMaker direct
    client = ArceeAgentClient(deployment_method="sagemaker", endpoint_name="arcee-agent-endpoint")
"""

import json
import os
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from openai import OpenAI
import logging

# Import SageMaker client if available
try:
    from sagemaker_inference import SageMakerInference
    SAGEMAKER_AVAILABLE = True
except ImportError:
    SAGEMAKER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ToolCall:
    """Represents a tool call from the model"""
    name: str
    arguments: Dict[str, Any]
    confidence: Optional[float] = None

@dataclass
class FunctionCallResponse:
    """Response from function calling"""
    tool_calls: List[ToolCall]
    raw_response: str
    processing_time: float
    model_used: str

class ArceeAgentClient:
    """
    Production-ready client for Arcee Agent model with multiple deployment options
    """
    
    def __init__(
        self,
        deployment_method: str = "openai_api",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy",
        model: str = "arcee-agent-finetuned",
        endpoint_name: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Arcee Agent client
        
        Args:
            deployment_method: "openai_api" or "sagemaker"
            base_url: Base URL for OpenAI-compatible API
            api_key: API key (dummy for local deployment)
            model: Model name
            endpoint_name: SageMaker endpoint name (if using SageMaker)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
        """
        self.deployment_method = deployment_method
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize the appropriate client
        if deployment_method == "openai_api":
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout
            )
            self.sagemaker_client = None
            logger.info(f"Initialized OpenAI client with base URL: {base_url}")
        
        elif deployment_method == "sagemaker":
            if not SAGEMAKER_AVAILABLE:
                raise ImportError("SageMaker client not available. Install boto3 and configure AWS credentials.")
            
            if not endpoint_name:
                endpoint_name = os.getenv("SAGEMAKER_ENDPOINT_NAME")
                if not endpoint_name:
                    raise ValueError("SageMaker endpoint name required")
            
            self.sagemaker_client = SageMakerInference(endpoint_name=endpoint_name)
            self.client = None
            logger.info(f"Initialized SageMaker client with endpoint: {endpoint_name}")
        
        else:
            raise ValueError(f"Unsupported deployment method: {deployment_method}")
    
    def function_call(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 512,
        top_p: float = 0.9,
        include_reasoning: bool = False
    ) -> FunctionCallResponse:
        """
        Perform function calling with the Arcee Agent model
        
        Args:
            query: User query/question
            tools: List of available tools with descriptions and parameters
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            include_reasoning: Include reasoning in the response
            
        Returns:
            FunctionCallResponse with tool calls and metadata
        """
        start_time = time.time()
        
        try:
            if self.deployment_method == "openai_api":
                response = self._call_openai_api(
                    query, tools, temperature, max_tokens, top_p, include_reasoning
                )
            elif self.deployment_method == "sagemaker":
                response = self._call_sagemaker(
                    query, tools, temperature, max_tokens, top_p, include_reasoning
                )
            else:
                raise ValueError(f"Unsupported deployment method: {self.deployment_method}")
            
            processing_time = time.time() - start_time
            
            # Parse tool calls
            tool_calls = self._parse_tool_calls(response)
            
            return FunctionCallResponse(
                tool_calls=tool_calls,
                raw_response=response,
                processing_time=processing_time,
                model_used=self.model
            )
            
        except Exception as e:
            logger.error(f"Function call failed: {e}")
            raise
    
    def _call_openai_api(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        include_reasoning: bool
    ) -> str:
        """Call using OpenAI-compatible API"""
        
        # Create function calling prompt
        prompt = self._create_function_calling_prompt(query, tools, include_reasoning)
        
        for attempt in range(self.max_retries):
            try:
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
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise
    
    def _call_sagemaker(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        include_reasoning: bool
    ) -> str:
        """Call using SageMaker endpoint"""
        
        payload = {
            "inputs": {
                "query": query,
                "tools": tools,
                "include_reasoning": include_reasoning
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "do_sample": True
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self.sagemaker_client.invoke_endpoint(payload)
                
                if "generated_text" in response:
                    return response["generated_text"]
                elif "tool_calls" in response:
                    return json.dumps(response["tool_calls"])
                else:
                    return str(response)
                    
            except Exception as e:
                logger.warning(f"SageMaker attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise
    
    def _create_function_calling_prompt(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        include_reasoning: bool = False
    ) -> str:
        """Create optimized function calling prompt"""
        
        tools_str = json.dumps(tools, indent=2)
        
        reasoning_instruction = ""
        if include_reasoning:
            reasoning_instruction = """
Before providing the tool calls, briefly explain your reasoning.
Format: Reasoning: [your reasoning]
Tool Calls: [JSON array]"""
        
        prompt = f"""You are an AI assistant with access to the following tools:

{tools_str}

Based on the user's query, determine which tool(s) to call and with what arguments.
Your response should be a JSON array of tool calls in the format:
[{{"name": "tool_name", "arguments": {{"param": "value"}}}}]
{reasoning_instruction}

User Query: {query}

Tool Calls:"""
        
        return prompt
    
    def _parse_tool_calls(self, response_text: str) -> List[ToolCall]:
        """Parse tool calls from model response"""
        
        try:
            # Handle reasoning if present
            if "Reasoning:" in response_text and "Tool Calls:" in response_text:
                # Extract only the tool calls part
                tool_calls_start = response_text.find("Tool Calls:") + len("Tool Calls:")
                response_text = response_text[tool_calls_start:].strip()
            
            # Find JSON array or object
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                # Look for single tool call object
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    tool_call_dict = json.loads(json_str)
                    
                    if 'name' in tool_call_dict:
                        return [ToolCall(
                            name=tool_call_dict['name'],
                            arguments=tool_call_dict.get('arguments', {})
                        )]
                
                return []
            
            # Parse JSON array
            json_str = response_text[start_idx:end_idx]
            tool_calls_data = json.loads(json_str)
            
            if not isinstance(tool_calls_data, list):
                tool_calls_data = [tool_calls_data]
            
            tool_calls = []
            for call_data in tool_calls_data:
                if isinstance(call_data, dict) and 'name' in call_data:
                    tool_calls.append(ToolCall(
                        name=call_data['name'],
                        arguments=call_data.get('arguments', {}),
                        confidence=call_data.get('confidence')
                    ))
            
            return tool_calls
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            logger.warning(f"Response text: {response_text}")
        except Exception as e:
            logger.warning(f"Unexpected error parsing tool calls: {e}")
        
        return []
    
    def batch_function_call(
        self,
        queries_and_tools: List[tuple],
        temperature: float = 0.1,
        max_tokens: int = 512
    ) -> List[FunctionCallResponse]:
        """
        Process multiple function calls in batch
        
        Args:
            queries_and_tools: List of (query, tools) tuples
            temperature: Sampling temperature
            max_tokens: Maximum tokens per call
            
        Returns:
            List of FunctionCallResponse objects
        """
        responses = []
        
        for query, tools in queries_and_tools:
            try:
                response = self.function_call(
                    query=query,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                responses.append(response)
                
            except Exception as e:
                logger.error(f"Batch call failed for query '{query}': {e}")
                # Add empty response for failed calls
                responses.append(FunctionCallResponse(
                    tool_calls=[],
                    raw_response=f"Error: {str(e)}",
                    processing_time=0.0,
                    model_used=self.model
                ))
        
        return responses
    
    def health_check(self) -> bool:
        """Check if the deployment is healthy"""
        try:
            if self.deployment_method == "openai_api":
                # Try a simple completion
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                return response.choices[0].message.content is not None
                
            elif self.deployment_method == "sagemaker":
                # Check endpoint status
                return self.sagemaker_client.validate_endpoint()
                
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
        
        return False


def demo_usage():
    """Demonstrate proper usage of the Arcee Agent client"""
    
    print("🚀 Arcee Agent Client Demo")
    print("=" * 50)
    
    # Initialize client (change base_url for your deployment)
    client = ArceeAgentClient(
        deployment_method="openai_api",
        base_url="http://localhost:8000/v1",
        model="arcee-agent-finetuned"
    )
    
    # Health check
    if not client.health_check():
        print("❌ Client health check failed. Make sure the API server is running.")
        return
    
    print("✅ Client initialized and healthy")
    
    # Define sample tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or coordinates"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                        "description": "Temperature units"
                    }
                },
                "required": ["location"]
            }
        },
        {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of results"
                    }
                },
                "required": ["query"]
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
    
    # Test queries
    test_queries = [
        "What's the weather like in Paris?",
        "Search for the latest news about artificial intelligence",
        "Calculate 25 * 47 + 18",
        "Get weather in New York and Tokyo",
        "Search for python tutorials and calculate 100 / 3"
    ]
    
    print("\n📞 Testing Function Calls")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)
        
        try:
            # Make function call
            response = client.function_call(
                query=query,
                tools=tools,
                temperature=0.1,
                include_reasoning=True
            )
            
            # Display results
            print(f"🕐 Processing time: {response.processing_time:.2f}s")
            print(f"🤖 Model: {response.model_used}")
            
            if response.tool_calls:
                print(f"📞 Tool calls ({len(response.tool_calls)}):")
                for j, tool_call in enumerate(response.tool_calls, 1):
                    print(f"  {j}. {tool_call.name}")
                    for param, value in tool_call.arguments.items():
                        print(f"     • {param}: {value}")
            else:
                print("❌ No tool calls generated")
                print(f"📝 Raw response: {response.raw_response[:100]}...")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n🔄 Testing Batch Processing")
    print("=" * 50)
    
    # Prepare batch data
    batch_data = [
        ("What's 50 + 75?", [tools[2]]),  # Only calculator
        ("Weather in London", [tools[0]]),  # Only weather
        ("Search for Python", [tools[1]])   # Only search
    ]
    
    batch_responses = client.batch_function_call(batch_data)
    
    for i, (query, _) in enumerate(batch_data):
        response = batch_responses[i]
        print(f"\n{i+1}. {query}")
        print(f"   Time: {response.processing_time:.2f}s")
        if response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"   Tool: {tool_call.name} - {tool_call.arguments}")
        else:
            print("   No tool calls")
    
    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    demo_usage()
