#!/usr/bin/env python3
"""
WebAgents V2.0 Server Demo

Demonstrates the complete V2.0 server with OpenAI-compatible endpoints,
multiple agents, streaming support, and comprehensive capabilities.
"""

import os
import asyncio
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.core.llm.openai import OpenAISkill
from webagents.agents.tools.decorators import tool
from webagents.server.core.app import WebAgentsServer
from webagents.utils.logging import setup_logging, capture_all_loggers


# Example tool for demonstration
@tool(scope="all")
def get_weather(location: str = "New York") -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny and 75°F"


@tool(scope="all") 
def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> dict:
    """Calculate tip amount and total bill."""
    tip = (bill_amount * tip_percentage) / 100
    total = bill_amount + tip
    return {
        "bill_amount": bill_amount,
        "tip_percentage": tip_percentage,
        "tip_amount": round(tip, 2),
        "total_amount": round(total, 2)
    }


def create_demo_agents():
    """Create demonstration agents with different capabilities"""
    
    # Agent 1: General Assistant with OpenAI
    general_agent = BaseAgent(
        name="assistant",
        instructions="You are a helpful AI assistant. You can check weather and calculate tips. Be friendly and concise.",
        model="openai/gpt-4o-mini",  # This will automatically create OpenAISkill
        scope="all"
    )
    
    # Register tools manually for demo
    general_agent.register_tool(get_weather, source="demo")
    general_agent.register_tool(calculate_tip, source="demo")
    
    # Agent 2: Weather Specialist
    weather_agent = BaseAgent(
        name="weather",
        instructions="You are a weather specialist. You only provide weather information using the weather tool.",
        model="openai/gpt-4o-mini",
        scope="all"
    )
    weather_agent.register_tool(get_weather, source="weather_service")
    
    # Agent 3: Math Helper  
    math_agent = BaseAgent(
        name="calculator",
        instructions="You are a mathematics assistant. You help with calculations including tip calculations.",
        model="openai/gpt-3.5-turbo", 
        scope="all"
    )
    math_agent.register_tool(calculate_tip, source="math_tools")
    
    return [general_agent, weather_agent, math_agent]


def main():
    """Run the V2.0 server demo"""
    
    print("🚀 WebAgents V2.0 Server Demo")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not set. Agents will fail to respond.")
        print("   Set your API key: export OPENAI_API_KEY='your-key-here'")
        print()
    
    # Create demo agents
    print("📋 Creating demo agents...")
    agents = create_demo_agents()
    
    for agent in agents:
        skills = list(agent.skills.keys())
        tools = len(agent._registered_tools)
        print(f"   ✅ {agent.name}: {len(skills)} skills, {tools} tools")
    
    # Create server
    print("\n🌐 Creating WebAgents V2.0 Server...")
    server = WebAgentsServer(
        agents=agents,
        title="WebAgents V2.0 Demo Server",
        description="OpenAI-compatible AI agent server with multiple specialized agents"
    )
    
    # Print endpoint information
    print("\n✨ Available Endpoints:")
    print("   GET  /                              - Server info & agent discovery")
    print("   GET  /health                        - Health check")  
    print("   GET  /health/detailed               - Detailed health with agent status")
    print()
    
    for agent in agents:
        print(f"   Agent '{agent.name}':")
        print(f"     GET  /{agent.name}                     - Agent info")
        print(f"     POST /{agent.name}/chat/completions    - Chat completions (OpenAI compatible)")
        print()
    
    print("💡 Example Requests:")
    print()
    
    # Non-streaming example
    print("   # Non-streaming chat completion:")
    print("   curl -X POST http://localhost:8000/assistant/chat/completions \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{")
    print("       \"model\": \"assistant\",")
    print("       \"messages\": [{\"role\": \"user\", \"content\": \"What's the weather like?\"}]")
    print("     }'")
    print()
    
    # Streaming example  
    print("   # Streaming chat completion:")
    print("   curl -X POST http://localhost:8000/weather/chat/completions \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{")
    print("       \"model\": \"weather\",") 
    print("       \"messages\": [{\"role\": \"user\", \"content\": \"Weather in Paris?\"}],")
    print("       \"stream\": true")
    print("     }'")
    print()
    
    # External tools example
    print("   # With external tools:")
    print("   curl -X POST http://localhost:8000/calculator/chat/completions \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{")
    print("       \"messages\": [{\"role\": \"user\", \"content\": \"Calculate tip for $50\"}],")
    print("       \"tools\": [")
    print("         {")
    print("           \"type\": \"function\",")
    print("           \"function\": {")
    print("             \"name\": \"custom_calculation\",")
    print("             \"description\": \"Perform custom calculation\",")
    print("             \"parameters\": {\"type\": \"object\", \"properties\": {}}")
    print("           }")
    print("         }")
    print("       ]")
    print("     }'")
    print()
    
    print("🔧 Features Demonstrated:")
    print("   ✅ Multiple specialized agents")
    print("   ✅ OpenAI-compatible API (streaming & non-streaming)")
    print("   ✅ Automatic tool registration and execution")
    print("   ✅ External tools support")
    print("   ✅ Health monitoring and agent discovery")
    print("   ✅ Context management and middleware")
    print("   ✅ CORS support for web clients")
    print()
    
    print("🌐 Starting server on http://localhost:8000")
    print("🔥 Press Ctrl+C to stop")
    print("=" * 50)
    
    # Ensure all loggers use structured format
    capture_all_loggers()
    
    # Start the server
    try:
        import uvicorn
        uvicorn.run(server.app, host="0.0.0.0", port=8000, log_level="info")
    except ImportError:
        print("❌ uvicorn not installed. Install with: pip install uvicorn")
        print("   Or run manually: python -m uvicorn v2_server_demo:server.app --host 0.0.0.0 --port 8000")
    except KeyboardInterrupt:
        print("\n👋 Server stopped!")


if __name__ == "__main__":
    main() 