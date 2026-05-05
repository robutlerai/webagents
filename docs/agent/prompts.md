---
title: Agent Prompts
description: Dynamic system-prompt contributors via the `@prompt` decorator — priority order, scoping, async, and skill integration.
---

# Agent Prompts

Enhance your agent's system prompt dynamically using the `@prompt` decorator. Prompt functions execute before each LLM call and contribute contextual information to the system message.

Prompts run in priority order (lower runs first) and support scope-based access control. Use them for dynamic context, user-specific information, or system status updates.

## Overview

Prompt functions generate dynamic content that gets appended to the agent's system message before LLM execution. They're perfect for injecting real-time context, user information, or environmental data.

**Key features:**

- Dynamic system-prompt enhancement
- Priority-based execution order
- Scope-based access control
- Context injection
- Automatic string concatenation
- Sync and async support

## Basic Usage

### Simple Prompt

```typescript tab="TypeScript"
import { BaseAgent, Skill, prompt } from 'webagents';
import type { Context } from 'webagents';

class StatusPrompts extends Skill {
  readonly name = 'status-prompts';

  @prompt()
  systemStatus(ctx: Context): string {
    return 'System Status: Online - All services operational';
  }
}

const agent = new BaseAgent({
  name: 'assistant',
  model: 'openai/gpt-4o',
  skills: [new StatusPrompts()],
});
```

```python tab="Python"
from webagents import BaseAgent, prompt

@prompt()
def system_status_prompt(context) -> str:
    """Add current system status to the prompt"""
    return "System Status: Online - All services operational"

agent = BaseAgent(
    name="assistant",
    model="openai/gpt-4o",
    capabilities=[system_status_prompt],
)
```

The agent's effective system message becomes:

```
You are a helpful AI assistant.

System Status: Online - All services operational

Your name is assistant, you are an AI agent in the Internet of Agents.
Current time: 2024-01-15T10:30:00
```

### Priority-Based Execution

```typescript tab="TypeScript"
class ContextPrompts extends Skill {
  readonly name = 'context-prompts';

  @prompt({ priority: 5 })
  timePrompt(ctx: Context): string {
    return `Current Time: ${new Date().toISOString()}`;
  }

  @prompt({ priority: 10 })
  systemStatusPrompt(ctx: Context): string {
    return `System Status: ${getSystemStatus()}`;
  }

  @prompt({ priority: 20 })
  userContextPrompt(ctx: Context): string {
    const userId = ctx.auth?.userId ?? 'anonymous';
    return `Current User: ${userId}`;
  }
}
```

```python tab="Python"
from datetime import datetime

@prompt(priority=5)
def time_prompt(context) -> str:
    """Add current timestamp (executes first)"""
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

Prompts execute in ascending priority order (5 → 10 → 20).

## Scope-Based Access Control

Control which callers see specific prompt content:

```typescript tab="TypeScript"
class ScopedPrompts extends Skill {
  readonly name = 'scoped-prompts';

  @prompt({ scope: 'all' })
  publicPrompt(ctx: Context): string {
    return 'Public system information';
  }

  @prompt({ scope: 'owner' })
  ownerPrompt(ctx: Context): string {
    return `Owner Dashboard: ${getOwnerStats()}`;
  }

  @prompt({ scope: 'admin' })
  adminPrompt(ctx: Context): string {
    return `DEBUG MODE: ${getDebugInfo()}`;
  }

  @prompt({ scope: ['premium', 'enterprise'] })
  premiumPrompt(ctx: Context): string {
    return 'Premium features enabled';
  }
}
```

```python tab="Python"
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

```typescript tab="TypeScript"
class UserPrompts extends Skill {
  readonly name = 'user-prompts';

  @prompt({ priority: 10 })
  async userContextPrompt(ctx: Context): Promise<string> {
    const userId = ctx.auth?.userId ?? 'anonymous';
    const userData = await getUserData(userId);
    return `User Context:
- Name: ${userData.name}
- Role: ${userData.role}
- Preferences: ${userData.preferences}`;
  }

  @prompt({ priority: 20 })
  async dynamicDataPrompt(ctx: Context): Promise<string> {
    const [market, weather] = await Promise.all([
      fetchMarketData(),
      fetchWeather(),
    ]);
    return `Real-time Context:
- Market: ${market.status}
- Weather: ${weather.condition}`;
  }
}
```

```python tab="Python"
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
    market_data = await fetch_market_data()
    weather_data = await fetch_weather()

    return f"""Real-time Context:
- Market: {market_data['status']}
- Weather: {weather_data['condition']}"""
```

## Skill Integration

Use prompts within skills for modular functionality:

```typescript tab="TypeScript"
import { Skill, prompt } from 'webagents';
import type { Context } from 'webagents';

class AnalyticsSkill extends Skill {
  readonly name = 'analytics';

  @prompt({ priority: 15, scope: 'owner' })
  async analyticsPrompt(ctx: Context): Promise<string> {
    const stats = await this.getAnalyticsData();
    return `Analytics Summary:
- Active Users: ${stats.activeUsers}
- Revenue Today: $${stats.dailyRevenue}
- System Load: ${stats.cpuUsage}%`;
  }

  @prompt({ priority: 25 })
  async performancePrompt(ctx: Context): Promise<string> {
    const metrics = await this.getPerformanceMetrics();
    return `Performance: ${metrics.responseTime}ms avg`;
  }

  private async getAnalyticsData() {
    return { activeUsers: 1250, dailyRevenue: 5420, cpuUsage: 23 };
  }
  private async getPerformanceMetrics() {
    return { responseTime: 150 };
  }
}

const agent = new BaseAgent({
  name: 'analytics-agent',
  model: 'openai/gpt-4o',
  skills: [new AnalyticsSkill()],
});
```

```python tab="Python"
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
        return {"active_users": 1250, "daily_revenue": 5420, "cpu_usage": 23}

    def get_performance_metrics(self) -> dict:
        return {"response_time": 150}

agent = BaseAgent(
    name="analytics-agent",
    model="openai/gpt-4o",
    skills={"analytics": AnalyticsSkill()},
)
```

## Advanced Patterns

### Conditional Prompts

```typescript tab="TypeScript"
@prompt({ priority: 10 })
conditionalPrompt(ctx: Context): string {
  const userRole = (ctx.metadata.user_role as string) ?? 'guest';
  if (userRole === 'admin') return 'ADMIN MODE: Full system access enabled';
  if (userRole === 'premium') return 'PREMIUM MODE: Enhanced features available';
  return 'STANDARD MODE: Basic features';
}

@prompt({ priority: 15 })
timeBasedPrompt(ctx: Context): string {
  const hour = new Date().getHours();
  if (hour >= 6 && hour < 12) return 'Good morning! System ready for daily operations.';
  if (hour >= 12 && hour < 18) return 'Good afternoon! Peak usage period - optimized for performance.';
  return 'Good evening! Running in power-save mode.';
}
```

```python tab="Python"
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

```typescript tab="TypeScript"
@prompt({ priority: 5 })
async safePrompt(ctx: Context): Promise<string> {
  try {
    const externalData = await fetchExternalService();
    return `External Status: ${externalData.status}`;
  } catch (e) {
    console.warn('External service unavailable', e);
    return 'External Status: Offline (using cached data)';
  }
}

@prompt({ priority: 10 })
async resilientAsyncPrompt(ctx: Context): Promise<string> {
  try {
    const data = await Promise.race([
      fetchSlowService(),
      new Promise<never>((_, r) => setTimeout(() => r(new Error('timeout')), 2000)),
    ]);
    return `Live Data: ${data.value}`;
  } catch (e) {
    return (e as Error).message === 'timeout'
      ? 'Live Data: Timeout (using fallback)'
      : 'Live Data: Service unavailable';
  }
}
```

```python tab="Python"
import asyncio, logging
logger = logging.getLogger(__name__)

@prompt(priority=5)
def safe_prompt(context) -> str:
    """Prompt with error handling"""
    try:
        external_data = fetch_external_service()
        return f"External Status: {external_data['status']}"
    except Exception as e:
        logger.warning(f"External service unavailable: {e}")
        return "External Status: Offline (using cached data)"

@prompt(priority=10)
async def resilient_async_prompt(context) -> str:
    """Async prompt with timeout handling"""
    try:
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

```typescript tab="TypeScript"
// Good — concise and focused
@prompt()
statusPrompt(ctx: Context): string {
  return `Status: ${getStatus()}`;
}

// Avoid — too verbose, burns tokens on every call
```

```python tab="Python"
# Good - concise and focused
@prompt()
def status_prompt(context) -> str:
    return f"Status: {get_status()}"

# Avoid - too verbose, burns tokens on every call
```

### Use Appropriate Priorities

```typescript tab="TypeScript"
@prompt({ priority: 5 })  // Core system info first
systemPrompt(ctx: Context) { /* ... */ }

@prompt({ priority: 10 }) // User context second
userPrompt(ctx: Context)  { /* ... */ }

@prompt({ priority: 15 }) // Specific features last
featurePrompt(ctx: Context) { /* ... */ }
```

```python tab="Python"
@prompt(priority=5)   # Core system info first
def system_prompt(context) -> str: ...

@prompt(priority=10)  # User context second
def user_prompt(context) -> str: ...

@prompt(priority=15)  # Specific features last
def feature_prompt(context) -> str: ...
```

### Handle Failures Gracefully

Wrap external calls; never let a prompt throw — the agent will fall back to its base instructions but you lose the contextual signal.

## Integration Examples

### With Authentication

```typescript tab="TypeScript"
@prompt({ priority: 10, scope: 'owner' })
authContextPrompt(ctx: Context): string {
  const user = ctx.auth?.user;
  if (user) return `Authenticated as: ${user.name} (${user.email})`;
  return 'Authentication: Guest user';
}
```

```python tab="Python"
@prompt(priority=10, scope="owner")
def auth_context_prompt(context) -> str:
    """Add authenticated user context"""
    user = getattr(context, 'authenticated_user', None)
    if user:
        return f"Authenticated as: {user['name']} ({user['email']})"
    return "Authentication: Guest user"
```

### With Payment Skills

```typescript tab="TypeScript"
@prompt({ priority: 15, scope: 'owner' })
async billingContextPrompt(ctx: Context): Promise<string> {
  const balance = await getUserBalance(String(ctx.auth?.userId));
  const usage = await getCurrentUsage(String(ctx.auth?.userId));
  return `Billing Status:
- Balance: $${balance.toFixed(2)}
- Usage Today: ${usage} credits`;
}
```

```python tab="Python"
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

```typescript tab="TypeScript"
@prompt({ priority: 20 })
async networkStatusPrompt(ctx: Context): Promise<string> {
  const connected = await countConnectedAgents();
  return `Network: ${connected} agents connected`;
}
```

```python tab="Python"
@prompt(priority=20)
def network_status_prompt(context) -> str:
    """Add network connectivity status"""
    connected_agents = count_connected_agents()
    return f"Network: {connected_agents} agents connected"
```

## See Also

- **[Tools](./tools.md)** — Executable functions for agents
- **[Hooks](./hooks.md)** — Event-driven processing
- **[Skills](./skills.md)** — Modular agent capabilities
- **[Endpoints](./endpoints.md)** — HTTP API routes
