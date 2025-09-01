# Supabase/PostgreSQL Skill

Minimalistic database integration for Supabase and PostgreSQL operations. Execute queries, manage data, and perform CRUD operations with secure credential storage.

## Features

- **Dual Database Support**: Works with both Supabase and PostgreSQL
- **SQL Query Execution**: Run parameterized SQL queries safely
- **CRUD Operations**: Create, Read, Update, Delete operations on tables
- **Secure Credentials**: Connection string storage
- **Per-User Isolation**: Each user has their own isolated database configuration

## Quick Setup

```python
from webagents.agents import BaseAgent
from webagents.agents.skills.ecosystem.database import SupabaseSkill

agent = BaseAgent(
    name="database-agent",
    model="openai/gpt-4o",
    skills={
        "database": SupabaseSkill()  # Auto-resolves: auth, kv
    }
)
```

Install dependencies:
```bash
pip install supabase psycopg2-binary
```

## Core Tools

### `supabase_setup(config)`
Configure database connection with Supabase or PostgreSQL credentials.

### `supabase_query(sql, params)`
Execute raw SQL queries with parameterization (PostgreSQL mode only).

### `supabase_table_ops(operation, table, data, filters)`
Perform CRUD operations: select, insert, update, delete.

### `supabase_status()`
Check database configuration and connection health.

## Usage Example

```python
# Setup database, create user, and query data
messages = [{
    'role': 'user',
    'content': 'Set up Supabase with URL https://myproject.supabase.co and my API key, then create a new user Alice Smith'
}]
response = await agent.run(messages=messages)
```

## Configuration

**Supabase**: `supabase_url`, `supabase_key`
**PostgreSQL**: Connection string or individual parameters (`host`, `port`, `database`, `user`, `password`)

## Troubleshooting

**"Database not configured"** - Run `supabase_setup()` with your credentials
**"Connection failed"** - Verify database server is accessible and credentials are correct
**"Permission denied"** - Check database user privileges and Row Level Security policies