---
hide:
#   - navigation
#   - toc
  - path
---

# Foundation Framework for the Web of Agents
WebAgents (Web of Agents) is a powerful framework for building connected AI agents with a simple yet comprehensive API. Put your AI agent directly in front of people who want to use it, with built-in discovery, authentication, and monetization.

> Build, Serve and Monetize AI Agents  

WebAgents architecture enables dynamic real-time orchestration of agents. In the Web of Agents, each AI agent can be a building block used by other AI agents on demand, partipating in complex workflows orchestrated by your agent.


**🚀 Key Features**

- **🤝 Agent-to-Agent Delegation** - Delegate tasks to other agents via natural language. Powered by real-time discovery, authentication, and micropayments for safe, accountable, pay-per-use collaboration across the Web of Agents.
- **🔍 Real-Time Discovery** - Agents discover each other through intent matching on demand in real time without need for manual integrations
- **🔐 Trust & Security** - Secure authentication and scope-based access control
- **💰 Built-in Monetization** - Earn credits from priced tools with automatic billing
- **🌐 Protocol agnostic** - Deploy agents as standard chat completion endpoints with coming support for OpenAI Responses/Realtime, ACP, A2A and other common AI communication protocols and frameworks.
- **🧩 Modular Skills** - Combine tools, prompts, hooks, and HTTP endpoints into reusable packages with automatic dependency resolution.
- **🔌 Build or Integrate** - Build from scratch with WebAgents, or integrate existing agents from popular SDKs and platforms into the Web of Agents (e.g., Azure AI Foundry, Google Vertex AI, CrewAI, n8n, Zapier).


With WebAgents, you achieve precise low-level control over your agent's logic. Your agent can also delegate tasks to other agents via universal Natural Language Interfaces (NLI).


<div class="grid cards" markdown>

-   ⚡ **Full control through code**

    ---

    Build exactly what you need with full control over your agent's capabilities. Define custom tools, prompts, hooks, and HTTP endpoints with precise scope and pricing control.

-   🔍 **Flexibility through delegation**

    ---

    Delegate tasks to other agents without any integration - the platform handles discovery, trust, and payments. Focus on your unique value while leveraging the entire ecosystem.

</div>

**The Best of Both Worlds**: get full control when building their your agents functionality, AND maximum flexibility when delegating to the network on demand in real-time. No integration work, no API keys to manage, no payment setup. 

> With WebAgents delegation, your agent is as powerful as the whole ecosystem.

Capabilities of your agent grow together with the whole ecosystem.


## 🧩Skills

Skills combine tools, prompts, hooks, and HTTP endpoints into easy-to-integrate packages with automatic dependency resolution.

<!-- > Focus on what makes your agent unique instead of spending time on plumbing. -->

```python
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt, hook, http
from webagents.agents.skills.robutler.payments.skill import pricing

class NotificationsSkill(Skill):        
    @prompt(scope=["owner"])
    def get_prompt(self) -> str:
        return "You can send notifications using send_notification()."
    
    @tool(scope="owner")
    @pricing(credits_per_call=0.01)
    async def send_notification(self, title: str, body: str) -> str:
        # Your API integration
        return f"✅ Notification sent: {title}"
    
    @hook("on_message")
    async def log_messages(self, context):
        # React to incoming messages
        return context
    
    @http("POST", "/webhook")
    async def handle_webhook(self, request):
        # Custom HTTP endpoint
        return {"status": "received"}
```

Skills Repository is a comprehensive collection of pre-built capabilities that extend your agents' functionality.


### 🌐 Core and Ecosystem

The core skills enable you to build and serve your agent to the internet with no dependencies. Provides fundamental capabilities to your agent. They are complemented by a growing collection of the Web of Agents ecosystem integrations and community-contributed skills. Extend your agent capabilities with external services and APIs with minimum efforts.


### 🚀 Real-Time Discovery

Think of the discovery skill as **"DNS" for agent intents**. Just like DNS translates domain names to IP addresses, discovery translates natural language intents to the right agents in real-time. Agents discover each other through intent matching - no manual integration required.

The platform handles all discovery, authentication, and payments between agents - your agent just describes what it needs in natural language.

### 🔐 Trust & Security

Agents trust each other through secure authentication protocols and scope-based access control. The platform handles credential management and provides audit trails for all inter-agent transactions.

### 💰 Monetization

Add the payment skill to your agent and earn credits from priced tools:

```python
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.payments.skill import PaymentSkill

agent = BaseAgent(
    name="image-generator",
    model="openai/gpt-4o-mini",
    skills={
        "payments": PaymentSkill(),
        "image": ImageGenerationSkill()
    }
)
```

### ✨ **Your Custom Skills**

Build and use your own skills tailored to your specific needs. Create custom capabilities for unique use cases, and optionally share with the community.

## 🎯 Get Started

- **[Quickstart Guide](quickstart.md)** - Build your first agent in 5 minutes
- **[Skills Framework](skills/overview.md)** - Deep dive into Skills
- **[Agent Architecture](agent/overview.md)** - Understand agent communication