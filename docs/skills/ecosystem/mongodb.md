# MongoDB Skill

Minimalistic MongoDB integration for document database operations. Connect to MongoDB Atlas, local instances, or self-hosted deployments with secure credential storage.

## Features

- **Flexible Deployment Support**: Works with MongoDB Atlas, local, and self-hosted instances
- **Document Operations**: Complete CRUD operations for MongoDB collections
- **Aggregation Pipelines**: Execute complex data processing and analytics
- **Credential Management**: Connection string storage
- **Per-User Isolation**: Each user has their own isolated MongoDB configuration

## Quick Setup

```python
from webagents.agents import BaseAgent
from webagents.agents.skills.ecosystem.mongodb import MongoDBSkill

agent = BaseAgent(
    name="mongodb-agent",
    model="openai/gpt-4o",
    skills={
        "mongodb": MongoDBSkill()  # Auto-resolves: auth, kv
    }
)
```

Install dependencies:
```bash
pip install pymongo
```

## Core Tools

### `mongodb_setup(config)`
Configure MongoDB connection with Atlas, local, or custom deployment credentials.

### `mongodb_query(database, collection, operation, query, data)`
Execute CRUD operations: find, insert_one, update_one, delete_one, etc.

### `mongodb_aggregate(database, collection, pipeline)`
Execute MongoDB aggregation pipelines for data processing and analytics.

### `mongodb_status()`
Check MongoDB configuration and connection health.

## Usage Example

```python
# Setup MongoDB Atlas and perform operations
messages = [{
    'role': 'user',
    'content': 'Set up MongoDB Atlas with connection string mongodb+srv://user:pass@cluster.net/db, then find all active agents'
}]
response = await agent.run(messages=messages)
```

## Configuration

**MongoDB Atlas**: `mongodb+srv://user:pass@cluster.mongodb.net/database`
**Local MongoDB**: `mongodb://localhost:27017/database`
**Self-hosted**: `mongodb://user:pass@hostname:27017/database`

## Troubleshooting

**"MongoDB not configured"** - Run `mongodb_setup()` with your connection credentials
**"Connection failed"** - Verify MongoDB server is accessible and credentials are correct
**"Authentication failed"** - Check username/password and database privileges
