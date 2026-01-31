# n8n Skill for WebAgents

Minimalistic n8n integration for workflow automation. This skill allows you to securely manage n8n API credentials and execute workflows from your WebAgent.

## Features

- 🔐 **Secure credential storage** via auth and KV skills
- 🚀 **Execute workflows** with custom input data
- 📋 **List workflows** from your n8n instance
- 📊 **Monitor execution status** in real-time
- 🔒 **Per-user isolation** with authentication
- 🛡️ **API key validation** during setup

## Prerequisites

1. **n8n instance** (self-hosted or cloud)
2. **n8n API key** (generated from n8n Settings > n8n API)
3. **WebAgents framework** with auth and kv skills enabled

## Quick Start

### 1. Set up n8n API credentials

```python
# First, configure your n8n credentials
await agent.call_tool("n8n_setup", 
    api_key="n8n_api_key_here",
    base_url="https://your-n8n-instance.com"  # Optional, defaults to localhost:5678
)
```

### 2. List available workflows

```python
# See all workflows in your n8n instance
await agent.call_tool("n8n_list_workflows")
```

### 3. Execute a workflow

```python
# Execute a workflow with optional input data
await agent.call_tool("n8n_execute",
    workflow_id="workflow_id_here",
    data={"input_param": "value"}  # Optional
)
```

### 4. Check execution status

```python
# Monitor workflow execution
await agent.call_tool("n8n_status", 
    execution_id="execution_id_from_execute_response"
)
```

## Tools Reference

### `n8n_setup(api_key, base_url=None)`
Set up n8n API credentials securely.

**Parameters:**
- `api_key` (str): Your n8n API key from Settings > n8n API
- `base_url` (str, optional): n8n instance URL (defaults to http://localhost:5678)

**Returns:**
- Success message with configuration details
- Error message if API key is invalid or instance unreachable

**Example:**
```python
result = await agent.call_tool("n8n_setup",
    api_key="n8n_1a2b3c4d5e6f7g8h9i0j",
    base_url="https://n8n.example.com"
)
# ✅ n8n credentials saved successfully!
# 🌐 Base URL: https://n8n.example.com
# 🔑 API key configured
```

### `n8n_execute(workflow_id, data=None)`
Execute an n8n workflow with optional input data.

**Parameters:**
- `workflow_id` (str): The ID of the workflow to execute
- `data` (dict, optional): Input data to pass to the workflow

**Returns:**
- Success message with execution ID and status
- Error message if workflow not found or execution fails

**Example:**
```python
result = await agent.call_tool("n8n_execute",
    workflow_id="123",
    data={"customer_email": "user@example.com", "order_id": "ORD-456"}
)
# ✅ Workflow executed successfully!
# 📋 Execution ID: exec_789
# 📊 Status: running
```

### `n8n_list_workflows()`
List all available workflows in your n8n instance.

**Parameters:** None

**Returns:**
- List of workflows with names, IDs, status, and tags
- Empty message if no workflows found

**Example:**
```python
result = await agent.call_tool("n8n_list_workflows")
# 📋 Available n8n Workflows:
# 
# 🟢 **Customer Onboarding** (ID: 123)
#    🏷️ Tags: automation, customers
# 
# 🔴 **Data Backup** (ID: 456)
#    🏷️ Tags: maintenance
# 
# 💡 Use n8n_execute(workflow_id, data) to run a workflow
```

### `n8n_status(execution_id)`
Check the status of a workflow execution.

**Parameters:**
- `execution_id` (str): The execution ID returned from n8n_execute

**Returns:**
- Detailed execution status report
- Error message if execution not found

**Example:**
```python
result = await agent.call_tool("n8n_status", execution_id="exec_789")
# 📊 Execution Status Report
# 🆔 Execution ID: exec_789
# 🔧 Workflow ID: 123
# ✅ Status: success
# 🕐 Started: 2024-01-15T10:30:00Z
# 🕑 Finished: 2024-01-15T10:32:15Z
```

## Setup Guide

### Getting your n8n API Key

1. Open your n8n instance
2. Go to **Settings** > **n8n API**
3. Click **Create an API key**
4. Provide a label (e.g., "WebAgent Integration")
5. Set expiration time (or leave blank for no expiration)
6. Copy the generated API key

### Environment Configuration

You can set a default n8n URL via environment variable:

```bash
export N8N_BASE_URL=https://your-n8n-instance.com
```

## Usage Examples

### Simple Workflow Execution
```python
# Setup (one-time)
await agent.call_tool("n8n_setup", api_key="your_api_key")

# Execute a simple workflow
result = await agent.call_tool("n8n_execute", workflow_id="welcome_email")
print(result)  # ✅ Workflow executed successfully!
```

### Workflow with Input Data
```python
# Execute workflow with custom data
customer_data = {
    "name": "John Doe",
    "email": "john@example.com",
    "subscription": "premium"
}

result = await agent.call_tool("n8n_execute", 
    workflow_id="customer_welcome",
    data=customer_data
)

# Monitor execution
execution_id = "exec_123"  # Extract from result
status = await agent.call_tool("n8n_status", execution_id=execution_id)
print(status)
```

### Discover and Execute Workflows
```python
# First, see what workflows are available
workflows = await agent.call_tool("n8n_list_workflows")
print(workflows)

# Execute a specific workflow
result = await agent.call_tool("n8n_execute", 
    workflow_id="data_processing",
    data={"source": "api", "format": "json"}
)
```

## Error Handling

The skill provides clear error messages for common scenarios:

- **❌ Authentication required** - User not authenticated
- **❌ API key is required** - Empty or missing API key
- **❌ Invalid API key** - API key rejected by n8n
- **❌ n8n instance not found** - Base URL unreachable
- **❌ Workflow not found** - Invalid workflow ID
- **❌ Execution not found** - Invalid execution ID

## Security

- **API keys** are stored securely using the KV skill with per-user namespacing
- **User isolation** ensures each user can only access their own credentials
- **Authentication required** for all operations
- **Automatic validation** of API keys during setup
- **Memory fallback** available if KV skill unavailable

## Architecture

```
WebAgent
├── Auth Skill (user context)
├── KV Skill (secure storage)
└── n8n Skill
    ├── Credential Management
    ├── API Communication
    └── Workflow Operations
```

## Dependencies

- `auth` skill - For user authentication and context
- `kv` skill - For secure credential storage
- `httpx` - For HTTP API communication

## Troubleshooting

### "Authentication required"
Ensure your agent has proper authentication configured.

### "Invalid API key" 
- Verify the API key is correct
- Check if the key has expired
- Ensure the key has necessary permissions

### "n8n instance not found"
- Verify the base URL is correct
- Check network connectivity
- Ensure n8n instance is running

### "Workflow not found"
- Use `n8n_list_workflows()` to see available workflows
- Check if the workflow ID is correct
- Verify the workflow is saved in n8n

## Limitations

- Supports n8n API v1 endpoints
- Requires n8n instance with API access enabled
- Limited to workflow execution and status monitoring
- Does not support workflow creation or modification

## Contributing

To extend this skill:

1. Add new tools following the existing pattern
2. Update tests in `tests/test_n8n_skill.py`
3. Update this documentation
4. Ensure proper error handling and user feedback

## License

Part of the WebAgents framework. See main project license.
