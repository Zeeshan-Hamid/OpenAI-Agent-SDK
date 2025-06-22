import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import requests
from pydantic import BaseModel

# First, install the required packages:
# pip install openai-agents openai pydantic requests

from agents import Agent, Runner, function_tool, InputGuardrail, GuardrailFunctionOutput, RunContextWrapper
from openai.types.responses import ResponseTextDeltaEvent

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Define data models for structured outputs
class WeatherData(BaseModel):
    location: str
    temperature: float
    description: str
    humidity: int

class CalculationResult(BaseModel):
    expression: str
    result: float
    explanation: str

class SearchResult(BaseModel):
    query: str
    summary: str
    sources: List[str]

class TaskClassification(BaseModel):
    task_type: str
    confidence: float
    reasoning: str

# Define tools using the @function_tool decorator
@function_tool
async def get_weather(location: str) -> str:
    """Get current weather information for a location.
    
    Args:
        location: The location to fetch the weather for.
    """
    # Mock weather API call - replace with real API like OpenWeatherMap
    try:
        # In a real implementation, you would call an actual weather API
        mock_data = {
            "location": location,
            "temperature": 22.5,
            "description": "Partly cloudy",
            "humidity": 65,
            "wind_speed": 10
        }
        
        return f"Weather in {location}: {mock_data['temperature']}°C, {mock_data['description']}, Humidity: {mock_data['humidity']}%, Wind: {mock_data['wind_speed']} km/h"
    except Exception as e:
        return f"Error fetching weather for {location}: {str(e)}"

@function_tool
async def calculate(expression: str) -> str:
    """Perform mathematical calculations and arithmetic operations.
    
    Args:
        expression: The mathematical expression to evaluate (e.g., "2 + 2", "10 * 5", "sqrt(16)")
    """
    try:
        # Safe evaluation of mathematical expressions
        import math
        
        # Replace common math functions
        expression = expression.replace('^', '**')
        expression = expression.replace('sqrt', 'math.sqrt')
        expression = expression.replace('sin', 'math.sin')
        expression = expression.replace('cos', 'math.cos')
        expression = expression.replace('tan', 'math.tan')
        expression = expression.replace('log', 'math.log')
        expression = expression.replace('pi', 'math.pi')
        expression = expression.replace('e', 'math.e')
        
        # Create a safe environment for evaluation
        safe_dict = {
            "__builtins__": {},
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }
        
        result = eval(expression, safe_dict)
        return f"The result of {expression} is {result}"
        
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"

@function_tool
async def search_web(query: str) -> str:
    """Search the web for information on any topic.
    
    Args:
        query: The search query to look up information for.
    """
    try:
        # Mock web search - in real implementation, use actual search API
        # You could integrate with Google Search API, Bing Search API, etc.
        mock_results = {
            "query": query,
            "summary": f"Here are the search results for '{query}'. Based on multiple sources, this topic involves various aspects and recent developments. Key findings include relevant information from authoritative sources.",
            "sources": [
                f"https://example.com/article-about-{query.replace(' ', '-')}",
                f"https://research.example.org/{query.replace(' ', '_')}-study",
                f"https://news.example.com/latest-on-{query.replace(' ', '-')}"
            ]
        }
        
        return f"Search results for '{query}':\n\n{mock_results['summary']}\n\nSources:\n" + "\n".join(f"- {source}" for source in mock_results['sources'])
        
    except Exception as e:
        return f"Error searching for '{query}': {str(e)}"

@function_tool
async def file_operation(operation: str, filename: str = "", content: str = "") -> str:
    """Perform file operations like reading, writing, or listing files.
    
    Args:
        operation: The operation to perform ('read', 'write', 'list')
        filename: The name of the file to operate on (required for read/write)
        content: The content to write to the file (required for write operation)
    """
    try:
        if operation == "read":
            if not filename:
                return "Error: filename is required for read operation"
            with open(filename, 'r', encoding='utf-8') as f:
                file_content = f.read()
            return f"Contents of {filename}:\n{file_content}"
            
        elif operation == "write":
            if not filename or not content:
                return "Error: both filename and content are required for write operation"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote content to {filename}"
            
        elif operation == "list":
            import os
            files = [f for f in os.listdir('.') if os.path.isfile(f)]
            directories = [d for d in os.listdir('.') if os.path.isdir(d)]
            
            result = "Current directory contents:\n\nFiles:\n"
            result += "\n".join(f"  - {file}" for file in files)
            result += "\n\nDirectories:\n"
            result += "\n".join(f"  - {dir}/" for dir in directories)
            
            return result
        else:
            return f"Error: Unsupported operation '{operation}'. Use 'read', 'write', or 'list'"
            
    except FileNotFoundError:
        return f"Error: File '{filename}' not found"
    except PermissionError:
        return f"Error: Permission denied accessing '{filename}'"
    except Exception as e:
        return f"Error performing file operation: {str(e)}"

@function_tool
async def get_current_time() -> str:
    """Get the current date and time.
    """
    current_time = datetime.now()
    return f"Current date and time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"

# Define specialized agents with updated models
def create_weather_agent():
    """
    Creates an agent specialized in handling weather-related queries.
    
    This agent is configured with the following properties:
    - name: "Weather Agent" - A descriptive name for identification.
    - handoff_description: "Specialist for weather-related queries and forecasts" - 
      Used by the orchestrator to determine when to route a request to this agent.
    - instructions: A detailed prompt telling the agent how to behave. It's instructed 
      to use the `get_weather` tool and provide comprehensive weather details.
    - tools: A list containing the `get_weather` tool, which allows the agent to fetch 
      weather information.
    - model: "gpt-4o" - Specifies the OpenAI model to use for this agent, chosen for 
      its strong performance.
    """
    return Agent(
        name="Weather Agent",
        handoff_description="Specialist for weather-related queries and forecasts",
        instructions="""You are a weather specialist agent. Use the get_weather tool to provide 
        accurate weather information. Always include location, temperature, conditions, and any 
        relevant weather details. Be helpful and provide context about weather conditions.""",
        tools=[get_weather],
        model="gpt-4o"  # Use GPT-4o for better performance
    )

def create_math_agent():
    """
    Creates an agent specialized in performing mathematical calculations.

    This agent is configured with the following properties:
    - name: "Math Agent" - A descriptive name.
    - handoff_description: "Specialist for mathematical calculations and problem solving" -
      Guides the orchestrator on when to use this agent.
    - instructions: A prompt instructing the agent to act as a math specialist, use the 
      `calculate` tool, and explain its reasoning. It supports a variety of mathematical
      functions.
    - tools: Contains the `calculate` tool for evaluating mathematical expressions.
    - model: "gpt-4o" - Chosen for its advanced reasoning capabilities suitable for math problems.
    """
    return Agent(
        name="Math Agent", 
        handoff_description="Specialist for mathematical calculations and problem solving",
        instructions="""You are a mathematics specialist agent. Use the calculate tool to solve 
        mathematical problems. Explain your steps clearly and provide detailed reasoning. 
        Handle arithmetic, algebra, and basic mathematical operations. Support functions like 
        sqrt, sin, cos, tan, log, pi, e, etc.""",
        tools=[calculate],
        model="gpt-4o"  # Use GPT-4o for better mathematical reasoning
    )

def create_research_agent():
    """
    Creates an agent specialized in research and gathering information from the web.

    This agent is configured with the following properties:
    - name: "Research Agent" - A descriptive name.
    - handoff_description: "Specialist for research, information gathering, and web searches" -
      Helps the orchestrator decide when to delegate research tasks.
    - instructions: A prompt that directs the agent to use the `search_web` tool, provide
      summaries, and cite sources, ensuring thorough and accurate research.
    - tools: Includes the `search_web` tool to perform web searches.
    - model: "gpt-4o" - Selected for its ability to handle complex research queries and
      summarize information effectively.
    """
    return Agent(
        name="Research Agent",
        handoff_description="Specialist for research, information gathering, and web searches",
        instructions="""You are a research specialist agent. Use the search_web tool to find 
        information on various topics. Provide comprehensive summaries and cite your sources. 
        Be thorough and accurate in your research.""",
        tools=[search_web],
        model="gpt-4o"  # Use GPT-4o for better research capabilities
    )

def create_file_agent():
    """
    Creates an agent specialized in handling file system operations.

    This agent is configured with the following properties:
    - name: "File Agent" - A descriptive name.
    - handoff_description: "Specialist for file operations and data management" -
      Indicates to the orchestrator that this agent handles file-related tasks.
    - instructions: A prompt instructing the agent to use the `file_operation` tool for
      tasks like reading, writing, and listing files. It also emphasizes caution.
    - tools: Contains the `file_operation` tool for file system interaction and `get_current_time`
      for time-related queries.
    - model: "gpt-4o-mini" - A smaller, faster model is sufficient for the straightforward
      logic of file operations.
    """
    return Agent(
        name="File Agent",
        handoff_description="Specialist for file operations and data management",
        instructions="""You are a file operations specialist agent. Use the file_operation tool 
        to handle file-related tasks like reading, writing, and listing files. Be careful with 
        file operations and always confirm actions with users when appropriate. You can also 
        provide the current time when needed.""",
        tools=[file_operation, get_current_time],
        model="gpt-4o-mini"  # Use mini model for file operations
    )

def create_general_assistant_agent():
    """
    Creates a general-purpose agent for handling conversations and basic queries.

    This agent serves as a fallback for requests that don't fit into a specialized
    category. It can engage in general conversation and answer simple questions.
    
    This agent is configured with the following properties:
    - name: "General Assistant" - A descriptive name.
    - handoff_description: "General purpose assistant for conversations and basic queries" -
      Used by the orchestrator for non-specialized tasks.
    - instructions: A prompt that tells the agent to be a helpful assistant for general
      topics and conversation.
    - tools: Includes the `get_current_time` tool to answer questions about the time.
    - model: "gpt-4o" - A powerful model to ensure high-quality conversational responses.
    """
    return Agent(
        name="General Assistant",
        handoff_description="General purpose assistant for conversations and basic queries",
        instructions="""You are a helpful general assistant. Handle general questions, 
        conversations, and provide helpful information. You can also provide the current 
        time when asked.""",
        tools=[get_current_time],
        model="gpt-4o"  # Use GPT-4o for general assistance
    )

# Create guardrail for input validation
def create_input_guardrail():
    """
    Creates an input guardrail to classify user requests before they are processed.

    A guardrail is a mechanism to inspect and potentially modify or block input before
    it reaches the main agent. This guardrail uses a dedicated "Input Classifier" agent
    to categorize the user's request.
    
    The classification helps the orchestrator make better routing decisions. The guardrail
    is configured not to block any requests (`tripwire_triggered=False`), but rather to
    attach classification metadata to the request.
    
    The inner `input_validation_guardrail` function is the core logic that runs the
    classification agent.
    """
    
    guardrail_agent = Agent(
        name="Input Classifier",
        instructions="""Classify the user's request into one of these categories:
        - weather: for weather-related queries
        - math: for mathematical calculations
        - research: for information seeking and web searches  
        - file: for file operations
        - general: for general conversation and other queries
        
        Provide confidence level (0.0 to 1.0) and reasoning for your classification.""",
        output_type=TaskClassification,
        model="gpt-4o-mini"  # Use mini model for classification
    )
    
    async def input_validation_guardrail(ctx, agent, input_data):
        try:
            result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
            classification = result.final_output_as(TaskClassification)
            
            # Allow all classifications but provide metadata
            return GuardrailFunctionOutput(
                output_info=classification,
                tripwire_triggered=False  # Don't block any requests
            )
        except Exception as e:
            # If classification fails, default to general
            return GuardrailFunctionOutput(
                output_info=TaskClassification(
                    task_type="general",
                    confidence=0.5,
                    reasoning=f"Classification failed: {str(e)}"
                ),
                tripwire_triggered=False
            )
    
    return InputGuardrail(guardrail_function=input_validation_guardrail)

# Create the central orchestrator agent
def create_orchestrator_agent():
    """
    Creates the central orchestrator agent that manages the entire agent system.

    The orchestrator is the entry point for all user requests. Its primary role is to
    route requests to the appropriate specialized agent based on the user's intent.
    
    This agent is configured with the following key properties:
    - name: "Central Orchestrator" - A descriptive name.
    - instructions: A detailed prompt that defines its routing logic. It is instructed
      to analyze the user's request and hand it off to the correct specialist agent
      (Weather, Math, Research, File, or General).
    - handoffs: A list of all the specialized agents it can delegate tasks to. This is
      the core of the multi-agent system, enabling the orchestrator to pass control
      to another agent.
    - input_guardrails: Includes the input classifier guardrail, which provides the
      orchestrator with metadata to improve its routing decisions.
    - tools: It has direct access to the `get_current_time` tool, allowing it to handle
      simple time-based queries without needing to hand off.
    - model: "gpt-4o" - A powerful model is used to handle the complex reasoning required
      for accurate routing and orchestration.
    """
    
    # Create specialized agents
    weather_agent = create_weather_agent()
    math_agent = create_math_agent()
    research_agent = create_research_agent()
    file_agent = create_file_agent()
    general_agent = create_general_assistant_agent()
    
    # Create input guardrail
    input_guardrail = create_input_guardrail()
    
    # Create orchestrator with all handoffs
    orchestrator = Agent(
        name="Central Orchestrator",
        instructions="""You are a central orchestrator agent that routes user requests to 
        specialized agents based on the type of task. 
        
        Route requests as follows:
        - Weather queries → Weather Agent
        - Mathematical calculations → Math Agent  
        - Research and information gathering → Research Agent
        - File operations → File Agent
        - General questions and conversation → General Assistant
        
        If you're unsure about routing, you can handle general queries directly or route 
        to the General Assistant. Always be helpful and provide clear explanations of 
        what you're doing.""",
        
        handoffs=[weather_agent, math_agent, research_agent, file_agent, general_agent],
        input_guardrails=[input_guardrail],
        tools=[get_current_time],  # Orchestrator can also handle time queries directly
        model="gpt-4o"  # Use GPT-4o for orchestration
    )
    
    return orchestrator

# Main execution class with streaming support
class AgentSystem:
    def __init__(self):
        self.orchestrator = create_orchestrator_agent()
        
    async def process_request(self, user_input: str) -> str:
        """Process a user request through the agent system (non-streaming)"""
        try:
            result = await Runner.run(self.orchestrator, user_input)
            return result.final_output
        except Exception as e:
            return f"Error processing request: {str(e)}"
    
    async def process_request_streaming(self, user_input: str, callback=None):
        """Process a user request with streaming support"""
        try:
            result = Runner.run_streamed(self.orchestrator, user_input)
            
            # Handle streaming events
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    # Stream text token by token
                    if callback:
                        await callback("token", event.data.delta)
                    else:
                        print(event.data.delta, end="", flush=True)
                        
                elif event.type == "agent_updated_stream_event":
                    # Agent handoff occurred
                    if callback:
                        await callback("agent_update", f"🔄 Switched to {event.new_agent.name}")
                    else:
                        print(f"\n🔄 Switched to {event.new_agent.name}")
                        
                elif event.type == "run_item_stream_event":
                    # Handle different types of run items
                    if event.item.type == "tool_call_item":
                        if callback:
                            await callback("tool_call", "🔧 Calling tool...")
                        else:
                            print("\n🔧 Calling tool...")
                            
                    elif event.item.type == "tool_call_output_item":
                        if callback:
                            await callback("tool_output", f"✅ Tool completed")
                        else:
                            print(f"\n✅ Tool completed")
            
            # Return the final result
            return result.final_output
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            if callback:
                await callback("error", error_msg)
            return error_msg
    
    async def run_interactive_session(self, streaming: bool = True):
        """Run an interactive session with the agent system"""
        print("🤖 OpenAI Agents System Started!")
        print(f"Streaming: {'Enabled' if streaming else 'Disabled'}")
        print("Type 'quit' to exit, 'help' for examples, 'toggle' to switch streaming mode\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self.show_examples()
                    continue
                elif user_input.lower() == 'toggle':
                    streaming = not streaming
                    print(f"Streaming {'enabled' if streaming else 'disabled'}\n")
                    continue
                elif not user_input:
                    continue
                
                print("🤖 ", end="", flush=True)
                
                if streaming:
                    response = await self.process_request_streaming(user_input)
                    print(f"\n\n")  # Add spacing after streaming
                else:
                    print("Processing...")
                    response = await self.process_request(user_input)
                    print(f"Agent: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}\n")
    
    async def run_streaming_demo(self):
        """Demonstrate streaming capabilities with various queries"""
        print("🎬 Streaming Demo - Watch responses appear in real-time!\n")
        
        demo_queries = [
            "What time is it and tell me a short joke?",
            "What's the weather like in New York?",
            "Calculate 25 * 4 + 100 and explain the steps",
            "Search for information about artificial intelligence",
            "Tell me about the benefits of renewable energy"
        ]
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{'='*60}")
            print(f"Demo {i}/5: {query}")
            print('='*60)
            print("🤖 ", end="", flush=True)
            
            try:
                await self.process_request_streaming(query)
                print("\n")
                
                # Small delay between demos
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"\nError in demo: {str(e)}")
        
        print("\n🎬 Demo completed!")
    
    def show_examples(self):
        """Show example queries for different agent types"""
        examples = {
            "Weather": [
                "What's the weather in New York?",
                "Get weather forecast for London",
                "How's the weather in Tokyo today?"
            ],
            "Math": [
                "Calculate 15 * 23 + 45",
                "What is 2^10?",
                "Solve (100 + 50) / 3",
                "Find sqrt(144)",
                "Calculate sin(30) + cos(60)"
            ],
            "Research": [
                "Search for information about artificial intelligence",
                "Find recent news about climate change",
                "Look up information about quantum computing"
            ],
            "File Operations": [
                "List all files in current directory",
                "Read the contents of config.txt",
                "Write 'Hello World' to test.txt"
            ],
            "General": [
                "What time is it?",
                "Tell me a joke",
                "How are you today?",
                "What can you help me with?"
            ]
        }
        
        print("\n📝 Example queries:")
        for category, queries in examples.items():
            print(f"\n{category}:")
            for query in queries:
                print(f"  • {query}")
        print()

# Example usage and testing
async def main():
    """Main function to demonstrate the agent system with streaming"""
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set your OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   or set it in your code: os.environ['OPENAI_API_KEY'] = 'your-key'")
        return
    
    # Initialize the agent system
    agent_system = AgentSystem()
    
    # Ask user what they want to do
    print("🚀 OpenAI Multi-Agent System with Streaming")
    print("\nChoose an option:")
    print("1. Run streaming demo")
    print("2. Interactive session")
    print("3. Test non-streaming mode")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            await agent_system.run_streaming_demo()
        elif choice == "2":
            await agent_system.run_interactive_session(streaming=True)
        elif choice == "3":
            # Test non-streaming mode
            test_queries = [
                "What time is it?",
                "What's the weather like in San Francisco?",
                "Calculate 25 * 4 + 100"
            ]
            
            print("\n🧪 Testing non-streaming mode:\n")
            
            for query in test_queries:
                print(f"Query: {query}")
                try:
                    response = await agent_system.process_request(query)
                    print(f"Response: {response}\n")
                except Exception as e:
                    print(f"Error: {str(e)}\n")
                print("-" * 60)
        else:
            print("Invalid choice. Starting interactive session...")
            await agent_system.run_interactive_session(streaming=True)
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())