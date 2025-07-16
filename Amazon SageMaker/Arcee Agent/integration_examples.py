#!/usr/bin/env python3
"""
Arcee Agent Integration Examples

This file demonstrates all the different ways to integrate and call the
fine-tuned Arcee Agent model in production environments.

Examples included:
1. OpenAI-compatible client (Recommended)
2. Direct HTTP requests
3. SageMaker endpoint integration
4. Batch processing
5. Streaming responses
6. Error handling and retries
"""

import json
import time
import asyncio
import requests
from typing import List, Dict, Any, Optional
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Example 1: OpenAI-Compatible Client (RECOMMENDED)
# =============================================================================

class ArceeAgentOpenAI:
    """
    Production-ready integration using OpenAI client
    This is the RECOMMENDED approach for most use cases
    """
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "dummy"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = "arcee-agent-finetuned"
    
    def function_call(self, query: str, tools: List[Dict]) -> List[Dict]:
        """Standard function calling using OpenAI client"""
        
        # Create function calling prompt
        tools_json = json.dumps(tools, indent=2)
        prompt = f"""You are an AI assistant with access to the following tools:

{tools_json}

Based on the user's query, determine which tool(s) to call and with what arguments.
Your response should be a JSON array of tool calls in the format:
[{{"name": "tool_name", "arguments": {{"param": "value"}}}}]

User Query: {query}

Tool Calls:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that excels at function calling. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=512
            )
            
            # Parse response
            content = response.choices[0].message.content
            return self._parse_tool_calls(content)
            
        except Exception as e:
            logger.error(f"OpenAI client error: {e}")
            return []
    
    def _parse_tool_calls(self, response_text: str) -> List[Dict]:
        """Parse tool calls from response"""
        try:
            # Find JSON in response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                # Single tool call
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    return [json.loads(json_str)]
                return []
            
            json_str = response_text[start_idx:end_idx]
            tool_calls = json.loads(json_str)
            return tool_calls if isinstance(tool_calls, list) else [tool_calls]
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {response_text}")
            return []


# =============================================================================
# Example 2: Direct HTTP Requests
# =============================================================================

class ArceeAgentHTTP:
    """
    Direct HTTP integration without OpenAI client
    Useful for non-Python integrations or custom requirements
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'ArceeAgent-Client/1.0'
        })
    
    def function_call(self, query: str, tools: List[Dict]) -> List[Dict]:
        """Function calling via direct HTTP requests"""
        
        endpoint = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": "arcee-agent-finetuned",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that excels at function calling."
                },
                {
                    "role": "user",
                    "content": self._create_prompt(query, tools)
                }
            ],
            "temperature": 0.1,
            "max_tokens": 512
        }
        
        try:
            response = self.session.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return self._parse_tool_calls(content)
            
        except requests.RequestException as e:
            logger.error(f"HTTP request error: {e}")
            return []
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Response parsing error: {e}")
            return []
    
    def _create_prompt(self, query: str, tools: List[Dict]) -> str:
        """Create function calling prompt"""
        tools_json = json.dumps(tools, indent=2)
        return f"""You have access to these tools:

{tools_json}

Based on the user's query, determine which tool(s) to call.
Respond with a JSON array of tool calls.

User Query: {query}

Tool Calls:"""
    
    def _parse_tool_calls(self, response_text: str) -> List[Dict]:
        """Parse tool calls (same as OpenAI example)"""
        try:
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    return [json.loads(response_text[start_idx:end_idx])]
                return []
            
            json_str = response_text[start_idx:end_idx]
            tool_calls = json.loads(json_str)
            return tool_calls if isinstance(tool_calls, list) else [tool_calls]
            
        except json.JSONDecodeError:
            return []


# =============================================================================
# Example 3: SageMaker Direct Integration
# =============================================================================

try:
    import boto3
    
    class ArceeAgentSageMaker:
        """
        Direct SageMaker endpoint integration
        For when you want to bypass the API server
        """
        
        def __init__(self, endpoint_name: str, region: str = "us-east-1"):
            self.endpoint_name = endpoint_name
            self.runtime = boto3.client('sagemaker-runtime', region_name=region)
        
        def function_call(self, query: str, tools: List[Dict]) -> List[Dict]:
            """Function calling via SageMaker endpoint"""
            
            payload = {
                "inputs": {
                    "query": query,
                    "tools": tools
                },
                "parameters": {
                    "temperature": 0.1,
                    "max_tokens": 512,
                    "do_sample": True
                }
            }
            
            try:
                response = self.runtime.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType='application/json',
                    Body=json.dumps(payload)
                )
                
                result = json.loads(response['Body'].read().decode())
                
                if "tool_calls" in result:
                    return result["tool_calls"]
                elif "generated_text" in result:
                    return self._parse_tool_calls(result["generated_text"])
                
                return []
                
            except Exception as e:
                logger.error(f"SageMaker invocation error: {e}")
                return []
        
        def _parse_tool_calls(self, response_text: str) -> List[Dict]:
            """Parse tool calls from generated text"""
            # Same parsing logic as other examples
            try:
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx != -1 and end_idx != 0:
                    return json.loads(response_text[start_idx:end_idx])
                return []
            except:
                return []
    
except ImportError:
    logger.warning("boto3 not available, SageMaker integration disabled")
    ArceeAgentSageMaker = None


# =============================================================================
# Example 4: Async/Streaming Integration
# =============================================================================

class ArceeAgentAsync:
    """
    Asynchronous integration for high-throughput applications
    """
    
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.base_url = base_url
    
    async def function_call_async(self, query: str, tools: List[Dict]) -> List[Dict]:
        """Async function calling"""
        import aiohttp
        
        payload = {
            "model": "arcee-agent-finetuned",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": self._create_prompt(query, tools)}
            ],
            "temperature": 0.1,
            "max_tokens": 512
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=30
                ) as response:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    return self._parse_tool_calls(content)
                    
            except Exception as e:
                logger.error(f"Async request error: {e}")
                return []
    
    async def batch_function_calls(self, queries_and_tools: List[tuple]) -> List[List[Dict]]:
        """Process multiple function calls concurrently"""
        tasks = [
            self.function_call_async(query, tools)
            for query, tools in queries_and_tools
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def _create_prompt(self, query: str, tools: List[Dict]) -> str:
        """Create function calling prompt"""
        tools_json = json.dumps(tools, indent=2)
        return f"""Tools available: {tools_json}

Query: {query}

Tool calls:"""
    
    def _parse_tool_calls(self, response_text: str) -> List[Dict]:
        """Parse tool calls"""
        try:
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                return json.loads(response_text[start_idx:end_idx])
            return []
        except:
            return []


# =============================================================================
# Example 5: Production Integration with Error Handling
# =============================================================================

class ArceeAgentProduction:
    """
    Production-ready integration with comprehensive error handling,
    retries, caching, and monitoring
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_enabled: bool = True
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = "arcee-agent-finetuned"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_enabled = cache_enabled
        self.cache = {} if cache_enabled else None
    
    def function_call(
        self,
        query: str,
        tools: List[Dict],
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Production function calling with error handling and metrics
        
        Returns:
            {
                "success": bool,
                "tool_calls": List[Dict],
                "error": str (if any),
                "processing_time": float,
                "from_cache": bool
            }
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(query, tools, temperature)
        if self.cache_enabled and cache_key in self.cache:
            return {
                "success": True,
                "tool_calls": self.cache[cache_key],
                "error": None,
                "processing_time": time.time() - start_time,
                "from_cache": True
            }
        
        # Attempt function call with retries
        for attempt in range(self.max_retries):
            try:
                prompt = self._create_prompt(query, tools)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant that excels at function calling."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
                
                content = response.choices[0].message.content
                tool_calls = self._parse_tool_calls(content)
                
                # Cache successful result
                if self.cache_enabled:
                    self.cache[cache_key] = tool_calls
                
                return {
                    "success": True,
                    "tool_calls": tool_calls,
                    "error": None,
                    "processing_time": time.time() - start_time,
                    "from_cache": False
                }
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    return {
                        "success": False,
                        "tool_calls": [],
                        "error": str(e),
                        "processing_time": time.time() - start_time,
                        "from_cache": False
                    }
    
    def _get_cache_key(self, query: str, tools: List[Dict], temperature: float) -> str:
        """Generate cache key"""
        import hashlib
        content = f"{query}_{json.dumps(tools, sort_keys=True)}_{temperature}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _create_prompt(self, query: str, tools: List[Dict]) -> str:
        """Create optimized function calling prompt"""
        tools_json = json.dumps(tools, indent=2)
        return f"""You have access to these tools:

{tools_json}

Based on the query, determine which tool(s) to call.
Respond with a JSON array: [{{"name": "tool_name", "arguments": {{"param": "value"}}}}]

Query: {query}

Tool calls:"""
    
    def _parse_tool_calls(self, response_text: str) -> List[Dict]:
        """Parse tool calls with error handling"""
        try:
            # Find JSON array
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                tool_calls = json.loads(json_str)
                
                # Validate format
                if isinstance(tool_calls, list):
                    valid_calls = []
                    for call in tool_calls:
                        if isinstance(call, dict) and 'name' in call:
                            valid_calls.append({
                                'name': call['name'],
                                'arguments': call.get('arguments', {})
                            })
                    return valid_calls
            
            # Try single object
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                call = json.loads(json_str)
                if 'name' in call:
                    return [{'name': call['name'], 'arguments': call.get('arguments', {})}]
            
            return []
            
        except Exception as e:
            logger.warning(f"Tool call parsing error: {e}")
            return []


# =============================================================================
# Demo and Usage Examples
# =============================================================================

def demo_all_integrations():
    """Demonstrate all integration methods"""
    
    print("🚀 Arcee Agent Integration Demo")
    print("=" * 50)
    
    # Sample tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        },
        {
            "name": "search_web",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    ]
    
    query = "What's the weather like in Paris and search for French restaurants?"
    
    # Example 1: OpenAI Client (Recommended)
    print("\n1. OpenAI Client Integration (RECOMMENDED)")
    print("-" * 45)
    
    try:
        openai_client = ArceeAgentOpenAI()
        result = openai_client.function_call(query, tools)
        print(f"✅ Result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 2: HTTP Client
    print("\n2. Direct HTTP Integration")
    print("-" * 30)
    
    try:
        http_client = ArceeAgentHTTP()
        result = http_client.function_call(query, tools)
        print(f"✅ Result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 3: Production Client
    print("\n3. Production Integration (with error handling)")
    print("-" * 48)
    
    try:
        prod_client = ArceeAgentProduction()
        result = prod_client.function_call(query, tools)
        print(f"✅ Success: {result['success']}")
        print(f"📞 Tool calls: {result['tool_calls']}")
        print(f"⏱️  Time: {result['processing_time']:.2f}s")
        print(f"💾 From cache: {result['from_cache']}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 4: Async Integration
    print("\n4. Async Integration")
    print("-" * 20)
    
    async def test_async():
        try:
            async_client = ArceeAgentAsync()
            result = await async_client.function_call_async(query, tools)
            print(f"✅ Async result: {result}")
            
            # Batch processing
            batch_data = [
                ("Weather in London", [tools[0]]),
                ("Search for Python tutorials", [tools[1]])
            ]
            batch_results = await async_client.batch_function_calls(batch_data)
            print(f"✅ Batch results: {batch_results}")
            
        except Exception as e:
            print(f"❌ Async error: {e}")
    
    try:
        asyncio.run(test_async())
    except Exception as e:
        print(f"❌ Async setup error: {e}")


def integration_best_practices():
    """Show best practices for production integration"""
    
    print("\n🎯 Integration Best Practices")
    print("=" * 35)
    
    print("""
1. **Use OpenAI Client** (Recommended)
   - Standard interface, works with existing tools
   - Easy to integrate with LangChain, etc.
   
   client = OpenAI(base_url="http://your-api/v1")
   response = client.chat.completions.create(...)

2. **Error Handling & Retries**
   - Always implement retry logic with exponential backoff
   - Handle network timeouts and API errors gracefully
   - Use circuit breakers for high-volume applications

3. **Caching**
   - Cache responses for identical queries
   - Use Redis or in-memory cache for production
   - Set appropriate TTL based on your use case

4. **Monitoring**
   - Track response times and error rates
   - Monitor token usage and costs
   - Set up alerts for service health

5. **Load Balancing**
   - Use multiple API replicas for high availability
   - Implement proper load balancing (round-robin, least connections)
   - Consider auto-scaling based on demand

6. **Security**
   - Use HTTPS in production
   - Implement proper authentication/authorization
   - Rate limiting to prevent abuse

7. **Testing**
   - Unit tests for function call parsing
   - Integration tests with the deployed model
   - Load testing for performance validation
    """)


if __name__ == "__main__":
    demo_all_integrations()
    integration_best_practices()
