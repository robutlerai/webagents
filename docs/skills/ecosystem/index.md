# Ecosystem Skills

A growing collection of third-party integrations and community-contributed skills that extend your agents' capabilities with popular services and platforms.

## Workflow Automation

<div class="grid cards" markdown>

-   🚀 **n8n Skill**

    ---
    
    Connect to n8n instances (self-hosted or cloud) to execute workflows, monitor status, and automate tasks.
    
    **Features**: Workflow execution, status monitoring, secure API key storage
    
    [Learn more →](n8n.md)

-   ⚡ **Zapier Skill**

    ---
    
    Integrate with Zapier to trigger Zaps and automate workflows across 7,000+ supported applications.
    
    **Features**: Zap triggering, task monitoring, 7,000+ app integrations
    
    [Learn more →](zapier.md)

-   🤖 **CrewAI Skill**

    ---
    
    Orchestrate multi-agent crews for collaborative AI workflows and complex task execution.
    
    **Features**: Agent crews, task delegation, process management, execution tracking
    
    [Learn more →](crewai.md)

</div>

## Social Media & Communication

<div class="grid cards" markdown>

-   🐦 **X.com (Twitter) Skill**

    ---
    
    Ultra-minimal X.com integration with OAuth 1.0a authentication, user subscriptions, and real-time notifications.
    
    **Features**: Tweet posting, user subscriptions, webhook monitoring, per-user rate limits
    
    [Learn more →](x_com.md)

</div>

## Cloud Services

<div class="grid cards" markdown>

-   🔍 **Google Skill**

    ---
    
    Integrate with Google services including Search, Gmail, Calendar, and Drive.
    
    **Features**: Google services integration, OAuth authentication
    
    [Learn more →](google.md)

</div>

## AI & Machine Learning

<div class="grid cards" markdown>

-   🤖 **Replicate Skill**

    ---
    
    Execute any machine learning model via Replicate's API - from text generation to image creation and video processing.
    
    **Features**: ML model execution, real-time monitoring, model discovery, prediction management
    
    [Learn more →](replicate.md)

-   🤖 **CrewAI Skill**

    ---
    
    Orchestrate multi-agent workflows using the CrewAI framework.
    
    **Features**: Multi-agent coordination, task delegation
    
    [Learn more →](crewai.md)

</div>

## Data & Storage

<div class="grid cards" markdown>

-   🗃️ **Supabase/PostgreSQL Skill**

    ---
    
    Connect to Supabase and PostgreSQL databases for data operations and real-time functionality.
    
    **Features**: SQL queries, CRUD operations, secure credential storage, dual database support
    
    [Learn more →](database.md)

-   📄 **MongoDB Skill**

    ---
    
    Connect to MongoDB Atlas, local, or self-hosted deployments for document database operations.
    
    **Features**: Document CRUD, aggregation pipelines, flexible deployment support, simple management
    
    [Learn more →](mongodb.md)

-   📁 **Filesystem Skill**

    ---
    
    Read, write, and manage files and directories on the local filesystem.
    
    **Features**: File operations, directory management, metadata access
    
    [Learn more →](filesystem.md)

</div>

## Integration Patterns

### Quick Setup Example

```python
from webagents.agents import BaseAgent
from webagents.agents.skills.ecosystem.x_com import XComSkill
from webagents.agents.skills.ecosystem.n8n import N8nSkill
from webagents.agents.skills.ecosystem.zapier import ZapierSkill
from webagents.agents.skills.ecosystem.replicate import ReplicateSkill
from webagents.agents.skills.ecosystem.crewai import CrewAISkill
from webagents.agents.skills.ecosystem.database import SupabaseSkill
from webagents.agents.skills.ecosystem.mongodb import MongoDBSkill

# All dependencies are automatically resolved
agent = BaseAgent(
    name="automation-agent",
    model="openai/gpt-4o",
    skills={
        "n8n": N8nSkill(),        # Auto-resolves: auth, kv
        "zapier": ZapierSkill(),  # Auto-resolves: auth, kv
        "replicate": ReplicateSkill(),  # Auto-resolves: auth, kv
        "crewai": CrewAISkill(),  # Auto-resolves: auth, kv
        "database": SupabaseSkill(),  # Auto-resolves: auth, kv
        "mongodb": MongoDBSkill(), # Auto-resolves: auth, kv
        "x_com": XComSkill()      # Auto-resolves: auth, kv, notifications
    }
)
```

### Robutler Agentic Use Cases

**Automated Content Generation and Publishing**
- AI agents autonomously generate, edit, and publish content
- Agents collaborate to ensure brand alignment and audience preferences
- CrewAI orchestrates content teams with specialized roles

**Dynamic Supply Chain Management**
- Agents monitor inventory and predict demand fluctuations
- Autonomous coordination with suppliers in real-time
- Database skills track inventory and transaction history

**Personalized Customer Support**
- AI agents handle inquiries, returns, and refunds autonomously
- X.com monitoring for customer service opportunities
- Zapier/n8n workflows for seamless customer journey automation

**Agent-to-Agent Economic Transactions**
- Agents discover intents and coordinate economic exchanges
- Database tracking of agent transactions and performance
- Workflow automation for payment processing and compliance

## Best Practices

All ecosystem skills follow WebAgents best practices:

- **👤 User context management** via auth skill
- **💾 Simple credential storage** via KV skill  
- **✅ API key validation** during setup
- **📝 Clear error handling** with helpful user messages
- **🧪 Comprehensive testing** with automated test suites

## Contributing

Help grow the ecosystem by contributing new skills:

1. **Follow the skill pattern** established by existing ecosystem skills
2. **Include comprehensive tests** covering all functionality
3. **Provide clear documentation** with examples and troubleshooting
4. **Implement simple credential management** using auth/KV skills
5. **Add error handling** with user-friendly messages

## Getting Help

- **Documentation**: Each skill has comprehensive docs with examples
- **Test Suites**: Review test files for usage patterns
- **Community**: Join discussions about skill development
- **Issues**: Report bugs or request features in the main repository

---

*The ecosystem grows with every contribution. Build the skill you need and share it with the community!*
