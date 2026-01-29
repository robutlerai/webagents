#!/usr/bin/env python3
"""
UCP Commerce Demo

Runs a server with merchant and client agents for UCP commerce testing.
"""

import os
import asyncio
from pathlib import Path

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.ecosystem.ucp import UCPSkill
from webagents.agents.tools.decorators import tool
from webagents.server.core.app import WebAgentsServer


# Merchant tools (executed after purchase)
@tool(scope="all")
def analyze_data_quick(data: str = "sample data") -> str:
    """Quick analysis of provided data."""
    return f"Quick Analysis Results:\n- Data points: {len(data.split())}\n- Summary: Data processed successfully\n- Confidence: 92%"


@tool(scope="all")
def analyze_data_deep(data: str = "sample data") -> str:
    """Deep comprehensive analysis of provided data."""
    return f"""Deep Analysis Results:
- Data points analyzed: {len(data.split())}
- Patterns detected: 3 significant trends
- Anomalies: None detected
- Recommendations:
  1. Continue current trajectory
  2. Monitor key metrics
  3. Consider optimization in Q2
- Confidence: 97%"""


@tool(scope="all")
def generate_summary(data: str = "sample data") -> str:
    """Generate executive summary."""
    return f"Executive Summary:\nAnalyzed {len(data.split())} data points. Key finding: stable growth pattern detected."


def create_merchant_agent() -> BaseAgent:
    """Create the merchant agent with UCP server mode."""
    # Create UCP skill in server mode
    ucp_skill = UCPSkill(config={
        "mode": "server",
        "agent_description": "Expert data analysis agent offering analytics services",
        "accepted_handlers": ["ai.robutler.token", "google.pay"],
        "services": [
            {
                "id": "quick_analysis",
                "title": "Quick Data Analysis",
                "description": "Fast analysis with key insights",
                "price": 500,  # $5.00
                "tool_name": "analyze_data_quick"
            },
            {
                "id": "deep_analysis", 
                "title": "Deep Data Analysis",
                "description": "Comprehensive analysis with recommendations",
                "price": 2500,  # $25.00
                "tool_name": "analyze_data_deep"
            },
            {
                "id": "summary_report",
                "title": "Summary Report",
                "description": "Executive summary of your data",
                "price": 1000,  # $10.00
                "tool_name": "generate_summary"
            }
        ]
    })
    
    agent = BaseAgent(
        name="ucp-merchant",
        instructions="""You are a merchant agent that sells data analysis services.
        
You can:
1. List your available services using list_services
2. Check orders received using list_orders
3. Provide data analysis when customers purchase your services

When customers ask about your services, describe them clearly with pricing.
""",
        model="openai/gpt-4o-mini",
        scopes=["all"],
        skills={"UCPSkill": ucp_skill}
    )
    
    # Add analysis tools
    agent.register_tool(analyze_data_quick, source="merchant_tools")
    agent.register_tool(analyze_data_deep, source="merchant_tools")
    agent.register_tool(generate_summary, source="merchant_tools")
    
    return agent


def create_client_agent() -> BaseAgent:
    """Create the client agent with UCP client mode."""
    # Create UCP skill in client mode
    ucp_skill = UCPSkill(config={
        "mode": "client",
        "enabled_handlers": ["ai.robutler.token"],
        "default_currency": "USD"
    })
    
    agent = BaseAgent(
        name="ucp-client",
        instructions="""You are a client agent that can discover and purchase services from other agents.

Your capabilities:
1. Use discover_merchant to find what services a merchant offers
2. Use create_checkout to start purchasing a service  
3. Use complete_purchase to pay and complete the transaction
4. Use get_checkout_status to check on pending purchases

When a user wants to buy a service:
1. First discover the merchant's capabilities
2. Create a checkout with the desired items
3. Complete the purchase with payment
""",
        model="openai/gpt-4o-mini",
        scopes=["all"],
        skills={"UCPSkill": ucp_skill}
    )
    
    return agent


def main():
    """Run the UCP commerce demo server."""
    print("🛒 UCP Commerce Demo")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print()
    
    # Create agents
    print("📋 Creating agents...")
    merchant = create_merchant_agent()
    client = create_client_agent()
    
    print(f"   ✅ {merchant.name}: UCP Server (sells services)")
    print(f"   ✅ {client.name}: UCP Client (buys services)")
    
    # Create server
    print("\n🌐 Creating server...")
    server = WebAgentsServer(
        agents=[merchant, client],
        title="UCP Commerce Demo",
        description="Agent-to-agent commerce demonstration"
    )
    
    print("\n✨ Endpoints:")
    print(f"   Merchant: http://localhost:8000/{merchant.name}")
    print(f"     - GET  /{merchant.name}/.well-known/ucp    (UCP profile)")
    print(f"     - GET  /{merchant.name}/ucp/services       (list services)")
    print(f"     - POST /{merchant.name}/checkout-sessions  (create checkout)")
    print()
    print(f"   Client:   http://localhost:8000/{client.name}")
    print(f"     - POST /{client.name}/chat/completions     (chat with client)")
    
    print("\n💡 Test Commands:")
    print(f'   # Get merchant profile')
    print(f'   curl http://localhost:8000/{merchant.name}/.well-known/ucp')
    print()
    print(f'   # List services')
    print(f'   curl http://localhost:8000/{merchant.name}/ucp/services')
    print()
    
    print("\n🔥 Starting server on http://localhost:8000")
    print("   Press Ctrl+C to stop")
    print("=" * 50)
    
    # Start server
    try:
        import uvicorn
        uvicorn.run(server.app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        print("\n👋 Server stopped!")


if __name__ == "__main__":
    main()
