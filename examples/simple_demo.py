#!/usr/bin/env python3
"""
Simple Agent Demo - Minimalistic Example

This demonstrates the absolute simplest way to create a discoverable agent 
with payments enabled using WebAgentsServer.
"""

from webagents.server import WebAgentsServer, pricing
from webagents.agent.agent import WebAgentsAgent
from agents import function_tool


@function_tool
@pricing(credits_per_call=1000)
async def get_greeting(name: str = "there") -> str:
    """Get a personalized greeting."""
    return f"Hello {name}! Welcome to WebAgents!"


# Create a simple agent
simple_agent = WebAgentsAgent(
    name="greeter",
    instructions="You are a friendly greeting assistant. Use the get_greeting tool to greet users.",
    credits_per_token=5,
    tools=[get_greeting],
    model="gpt-4o-mini"
)

# Create server - payments enabled by default!
app = WebAgentsServer(agents=[simple_agent], min_balance=1000, root_path="/agents")


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Simple Agent Demo")
    print("\n✨ Auto-created endpoint:")
    print("  POST /agents/greeter/chat/completions - Greeting agent (5 credits/token)")
    print("  GET  /agents/greeter                  - Agent info")
    
    print("\n💡 Example request:")
    print("  curl -X POST http://localhost:2225/agents/greeter/chat/completions \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"model\": \"gpt-4o-mini\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'")
    
    print("\n🌐 Starting server on http://localhost:2225")
    print("💰 Payments are ENABLED by default - minimum balance: 1000 credits")
    
    uvicorn.run(app, host="0.0.0.0", port=2225) 