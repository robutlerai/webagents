# Agent Handoffs

!!! warning "Alpha Software Notice"

    Handsoff feature is currently in **alpha stage**. Beta release is planned in the next major version. Please avoid using Handsoff in production environments. Contributions to this feature are welcome - please consider joining the discussion on Github and Discord!

Handoffs enable seamless agent-to-agent collaboration through natural language interfaces.

Use handoffs when one agent identifies a request better handled by a specialist. Handoffs integrate with discovery/NLI skills and respect scopes and payment policies.

## Handoff System Overview

```python
from webagents.agents.skills import Skill
from webagents.agents.skills.decorators import handoff

class FinanceSkill(Skill):
    @handoff("finance-expert")
    def needs_finance_expert(self, query: str) -> bool:
        """Determine if finance expert needed"""
        finance_terms = ["stock", "investment", "portfolio", "trading"]
        return any(term in query.lower() for term in finance_terms)
```

## Defining Handoffs

### Basic Handoff

```python
@handoff("target-agent-name")
def handoff_condition(self, query: str) -> bool:
    """Return True when handoff needed"""
    return "specific keyword" in query
```

### Handoff with Metadata

```python
@handoff("specialist", metadata={"priority": "high", "timeout": 30})
def needs_specialist(self, query: str) -> bool:
    """High-priority handoff to specialist"""
    return self.is_complex_query(query)
```

### Dynamic Handoff

```python
class RouterSkill(Skill):
    @handoff()  # No target specified
    def route_dynamically(self, query: str) -> str:
        """Return agent name dynamically"""
        if "legal" in query:
            return "legal-advisor"
        elif "medical" in query:
            return "medical-assistant"
        elif "technical" in query:
            return "tech-support"
        return None  # No handoff needed
```

## Handoff Execution

### Automatic Handoff

```python
# Agent automatically hands off when conditions met
response = await agent.run([
    {"role": "user", "content": "I need help with my stock portfolio"}
])
# Automatically routes to finance-expert if handoff defined
```

### Manual Handoff

```python
from webagents.agents.skills import NLISkill

class CollaborationSkill(Skill):
    def __init__(self, config=None):
        super().__init__(config)
        self.nli = NLISkill()
    
    @tool
    async def consult_expert(self, topic: str, question: str) -> str:
        """Manually consult an expert agent"""
        expert_map = {
            "finance": "finance-expert",
            "legal": "legal-advisor",
            "health": "medical-assistant"
        }
        
        expert = expert_map.get(topic)
        if expert:
            result = await self.nli.query_agent(expert, question)
            return result.get("response", "Expert unavailable")
        
        return "No expert available for this topic"
```

## Multi-Agent Workflows

### Sequential Handoffs

```python
class WorkflowSkill(Skill):
    @handoff("data-analyst")
    def needs_analysis(self, query: str) -> bool:
        """First: Send to analyst for data"""
        return "analyze" in query and not hasattr(self, "analysis_done")
    
    @handoff("report-writer")  
    def needs_report(self, query: str) -> bool:
        """Then: Send to writer for report"""
        return hasattr(self, "analysis_done") and "report" in query
    
    @hook("after_handoff")
    async def track_workflow(self, context):
        """Track workflow progress"""
        if context["handoff_agent"] == "data-analyst":
            self.analysis_done = True
        return context
```

### Parallel Handoffs

```python
class ResearchSkill(Skill):
    @tool
    async def research_topic(self, topic: str) -> Dict:
        """Research topic using multiple expert agents"""
        experts = ["science-expert", "history-expert", "culture-expert"]
        
        # Query all experts in parallel
        tasks = []
        for expert in experts:
            task = self.nli.query_agent(expert, f"Tell me about {topic}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Combine results
        return {
            "topic": topic,
            "perspectives": {
                expert: result.get("response")
                for expert, result in zip(experts, results)
            }
        }
```

## Handoff Context

### Before Handoff Hook

```python
@hook("before_handoff")
async def prepare_handoff(self, context):
    """Prepare context for handoff"""
    
    # Add context for target agent
    context["handoff_context"] = {
        "source_agent": context.agent_name,
        "user_intent": self.detected_intent,
        "conversation_summary": self.summarize_conversation(context.messages),
        "important_facts": self.extract_facts(context.messages)
    }
    
    # Validate handoff
    target = context["handoff_agent"]
    if not await self.is_agent_available(target):
        raise HandoffError(f"Agent {target} not available")
    
    return context
```

### After Handoff Hook

```python
@hook("after_handoff")
async def process_handoff_result(self, context):
    """Process results from target agent"""
    
    result = context["handoff_result"]
    
    # Extract and store insights
    if result.get("success"):
        insights = result.get("insights", {})
        await self.store_expert_knowledge(
            expert=context["handoff_agent"],
            insights=insights
        )
    
    # Update conversation context
    context["expert_consulted"] = True
    context["expert_response"] = result.get("response")
    
    return context
```

## Platform Integration

### Using Discovery Skill

```python
from webagents.agents.skills import DiscoverySkill

class SmartRouterSkill(Skill):
    def __init__(self, config=None):
        super().__init__(config)
        self.discovery = DiscoverySkill()
    
    @handoff()
    async def find_best_agent(self, query: str) -> str:
        """Discover and route to best agent"""
        
        # Find agents that can handle query
        agents = await self.discovery.find_agents(
            intent=query,
            max_results=5
        )
        
        # Score and select best agent
        best_agent = None
        best_score = 0
        
        for agent in agents:
            score = self.calculate_match_score(query, agent)
            if score > best_score:
                best_score = score
                best_agent = agent["name"]
        
        return best_agent if best_score > 0.7 else None
```

### Payment-Aware Handoffs

```python
from webagents.agents.skills import PaymentSkill

class PaidHandoffSkill(Skill):
    def __init__(self, config=None):
        super().__init__(config)
        self.payment = PaymentSkill()
    
    @hook("before_handoff")
    async def check_payment(self, context):
        """Ensure payment for premium agents"""
        
        target = context["handoff_agent"]
        
        # Check if target is premium
        if self.is_premium_agent(target):
            # Verify payment
            cost = self.get_agent_cost(target)
            
            if not await self.payment.charge_user(
                user_id=context.peer_user_id,
                amount=cost,
                description=f"Consultation with {target}"
            ):
                raise HandoffError("Payment required for premium agent")
        
        return context
```

## Error Handling

### Handoff Failures

```python
class ResilientHandoffSkill(Skill):
    @handoff("primary-expert")
    def needs_expert(self, query: str) -> bool:
        return "expert" in query
    
    @hook("after_handoff")
    async def handle_handoff_failure(self, context):
        """Fallback on handoff failure"""
        
        result = context["handoff_result"]
        
        if not result.get("success"):
            # Try fallback agent
            fallback_result = await self.nli.query_agent(
                "general-assistant",
                context.messages[-1]["content"]
            )
            
            if fallback_result.get("success"):
                context["handoff_result"] = fallback_result
            else:
                # Provide local response
                context["handoff_result"] = {
                    "success": True,
                    "response": "I'll do my best to help directly...",
                    "fallback": True
                }
        
        return context
```

## Best Practices

1. **Clear Conditions** - Make handoff conditions specific and testable
2. **Context Preservation** - Pass relevant context to target agents
3. **Error Handling** - Always have fallback strategies
4. **Cost Awareness** - Consider payment for premium agents
5. **Performance** - Cache agent discovery results when possible

## Complete Example

```python
from webagents.agents import BaseAgent
from webagents.agents.skills import Skill, NLISkill, DiscoverySkill
from webagents.agents.skills.decorators import handoff, hook, tool

class CustomerServiceSkill(Skill):
    def __init__(self, config=None):
        super().__init__(config, dependencies=["nli", "discovery"])
    
    @handoff()
    def route_to_department(self, query: str) -> str:
        """Route to appropriate department"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["bill", "payment", "charge"]):
            return "billing-department"
        elif any(word in query_lower for word in ["tech", "broken", "error"]):
            return "technical-support"
        elif any(word in query_lower for word in ["ship", "delivery", "track"]):
            return "shipping-department"
        
        return None
    
    @hook("before_handoff")
    async def add_customer_context(self, context):
        """Add customer information before handoff"""
        
        # Get customer info
        customer_id = context.peer_user_id
        customer_data = await self.get_customer_data(customer_id)
        
        # Add to handoff context
        context["handoff_metadata"] = {
            "customer_tier": customer_data.get("tier", "standard"),
            "history_summary": self.summarize_history(customer_data),
            "open_tickets": customer_data.get("open_tickets", [])
        }
        
        return context
    
    @tool
    async def escalate_to_human(self, reason: str) -> str:
        """Escalate to human support"""
        ticket = await self.create_support_ticket(
            customer_id=self.get_context().peer_user_id,
            reason=reason,
            conversation=self.get_context().messages
        )
        
        return f"I've created support ticket #{ticket['id']}. A human agent will contact you within 24 hours."

# Create customer service agent
agent = BaseAgent(
    name="customer-service",
    instructions="You are a helpful customer service agent. Route queries to appropriate departments.",
    model="openai/gpt-4o",
    skills={
        "routing": CustomerServiceSkill(),
        "nli": NLISkill(),
        "discovery": DiscoverySkill()
    }
) 