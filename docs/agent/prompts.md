# Agent Prompts

Enhance your agent's system prompt dynamically using the `@prompt` decorator. Prompt functions execute before each LLM call and contribute contextual information to the system message.

Prompts run in priority order and support scope-based access control. Use them for dynamic context, user-specific information, or system status updates.

## Overview

Prompt functions generate dynamic content that gets added to the agent's system message before LLM execution. They're perfect for injecting real-time context, user information, or environmental data.

**Key Features:**
- Dynamic system prompt enhancement
- Priority-based execution order
- Scope-based access control
- Context injection support
- Automatic string concatenation
- Sync and async support

## Basic Usage

### Simple Prompt

```python
from webagents import BaseAgent, prompt

@prompt()
def system_status_prompt(context) -> str:
    """Add current system status to the prompt"""
    return f"System Status: Online - All services operational"

agent = BaseAgent(
    name="assistant",
    model="openai/gpt-4o",
    capabilities=[system_status_prompt]
)
```

**Enhanced System Message:**
```
You are a helpful AI assistant.

System Status: Online - All services operational

Your name is assistant, you are an AI agent in the Internet of Agents. Current time: 2024-01-15T10:30:00
```

### Priority-Based Execution

```python
@prompt(priority=5)
def time_prompt(context) -> str:
    """Add current timestamp (executes first)"""
    from datetime import datetime
    return f"Current Time: {datetime.now().isoformat()}"

@prompt(priority=10)
def system_status_prompt(context) -> str:
    """Add system status (executes second)"""
    return f"System Status: {get_system_status()}"

@prompt(priority=20)
def user_context_prompt(context) -> str:
    """Add user context (executes third)"""
    user_id = getattr(context, 'user_id', 'anonymous')
    return f"Current User: {user_id}"
```

**Result:** Prompts execute in ascending priority order (5 → 10 → 20).

## Scope-Based Access Control

Control which users see specific prompt content:

```python
@prompt(scope="all")
def public_prompt(context) -> str:
    """Available to all users"""
    return "Public system information"

@prompt(scope="owner")
def owner_prompt(context) -> str:
    """Only for agent owners"""
    return f"Owner Dashboard: {get_owner_stats()}"

@prompt(scope="admin")
def admin_prompt(context) -> str:
    """Admin users only"""
    return f"DEBUG MODE: {get_debug_info()}"

@prompt(scope=["premium", "enterprise"])
def premium_prompt(context) -> str:
    """Multiple scopes"""
    return "Premium features enabled"
```

## Context Access

Access request context for dynamic content:

```python
@prompt(priority=10)
def user_context_prompt(context) -> str:
    """Generate user-specific prompt content"""
    user_id = getattr(context, 'user_id', 'anonymous')
    user_data = get_user_data(user_id)
    
    return f"""User Context:
- Name: {user_data['name']}
- Role: {user_data['role']}
- Preferences: {user_data['preferences']}"""

@prompt(priority=20)
async def dynamic_data_prompt(context) -> str:
    """Async prompt with external data"""
    # Fetch real-time data
    market_data = await fetch_market_data()
    weather_data = await fetch_weather()
    
    return f"""Real-time Context:
- Market: {market_data['status']}
- Weather: {weather_data['condition']}"""
```

## Skill Integration

Use prompts within skills for modular functionality:

```python
from webagents.agents.skills.base import Skill

class AnalyticsSkill(Skill):
    """Skill that adds analytics context to prompts"""
    
    @prompt(priority=15, scope="owner")
    def analytics_prompt(self, context) -> str:
        """Add analytics data to system prompt"""
        stats = self.get_analytics_data()
        return f"""Analytics Summary:
- Active Users: {stats['active_users']}
- Revenue Today: ${stats['daily_revenue']}
- System Load: {stats['cpu_usage']}%"""
    
    @prompt(priority=25)
    def performance_prompt(self, context) -> str:
        """Add performance metrics"""
        metrics = self.get_performance_metrics()
        return f"Performance: {metrics['response_time']}ms avg"
    
    def get_analytics_data(self) -> dict:
        # Fetch real analytics data
        return {"active_users": 1250, "daily_revenue": 5420, "cpu_usage": 23}
    
    def get_performance_metrics(self) -> dict:
        return {"response_time": 150}

# Use in agent
agent = BaseAgent(
    name="analytics-agent",
    model="openai/gpt-4o",
    skills={"analytics": AnalyticsSkill()}
)
```

## Advanced Patterns

### Conditional Prompts

```python
@prompt(priority=10)
def conditional_prompt(context) -> str:
    """Add content based on conditions"""
    user_role = getattr(context, 'user_role', 'guest')
    
    if user_role == 'admin':
        return "ADMIN MODE: Full system access enabled"
    elif user_role == 'premium':
        return "PREMIUM MODE: Enhanced features available"
    else:
        return "STANDARD MODE: Basic features"

@prompt(priority=15)
def time_based_prompt(context) -> str:
    """Different content based on time"""
    from datetime import datetime
    hour = datetime.now().hour
    
    if 6 <= hour < 12:
        return "Good morning! System ready for daily operations."
    elif 12 <= hour < 18:
        return "Good afternoon! Peak usage period - optimized for performance."
    else:
        return "Good evening! Running in power-save mode."
```

### Error Handling

```python
@prompt(priority=5)
def safe_prompt(context) -> str:
    """Prompt with error handling"""
    try:
        external_data = fetch_external_service()
        return f"External Status: {external_data['status']}"
    except Exception as e:
        # Log error but don't break prompt execution
        logger.warning(f"External service unavailable: {e}")
        return "External Status: Offline (using cached data)"

@prompt(priority=10)
async def resilient_async_prompt(context) -> str:
    """Async prompt with timeout handling"""
    try:
        # Set timeout for external calls
        async with asyncio.timeout(2.0):
            data = await fetch_slow_service()
            return f"Live Data: {data['value']}"
    except asyncio.TimeoutError:
        return "Live Data: Timeout (using fallback)"
    except Exception:
        return "Live Data: Service unavailable"
```

## Best Practices

### Keep Prompts Concise
```python
# ✅ Good - concise and focused
@prompt()
def status_prompt(context) -> str:
    return f"Status: {get_status()}"

# ❌ Avoid - too verbose
@prompt()
def verbose_prompt(context) -> str:
    return f"""
    This is a very long prompt that contains way too much information
    and will consume unnecessary tokens in every LLM call. It includes
    redundant details and verbose explanations that don't add value.
    The system status is {get_status()} but this prompt is too long.
    """
```

### Use Appropriate Priorities
```python
# ✅ Good - logical priority order
@prompt(priority=5)   # Core system info first
def system_prompt(context) -> str: ...

@prompt(priority=10)  # User context second  
def user_prompt(context) -> str: ...

@prompt(priority=15)  # Specific features last
def feature_prompt(context) -> str: ...
```

### Handle Failures Gracefully
```python
# ✅ Good - safe error handling
@prompt()
def safe_prompt(context) -> str:
    try:
        return f"Data: {get_data()}"
    except Exception:
        return "Data: Unavailable"

# ❌ Avoid - unhandled exceptions
@prompt()
def unsafe_prompt(context) -> str:
    return f"Data: {get_data()}"  # Could crash prompt execution
```

## Integration Examples

### With Authentication
```python
@prompt(priority=10, scope="owner")
def auth_context_prompt(context) -> str:
    """Add authenticated user context"""
    user = getattr(context, 'authenticated_user', None)
    if user:
        return f"Authenticated as: {user['name']} ({user['email']})"
    return "Authentication: Guest user"
```

### With Payment Skills
```python
@prompt(priority=15, scope="owner")
def billing_context_prompt(context) -> str:
    """Add billing information for owners"""
    balance = get_user_balance(context.user_id)
    usage = get_current_usage(context.user_id)
    
    return f"""Billing Status:
- Balance: ${balance:.2f}
- Usage Today: {usage} credits"""
```

### With Discovery Skills
```python
@prompt(priority=20)
def network_status_prompt(context) -> str:
    """Add network connectivity status"""
    connected_agents = count_connected_agents()
    return f"Network: {connected_agents} agents connected"
```

## See Also

- **[Tools](tools.md)** - Executable functions for agents
- **[Hooks](hooks.md)** - Event-driven processing
- **[Skills](skills.md)** - Modular agent capabilities
- **[Endpoints](endpoints.md)** - HTTP API routes
