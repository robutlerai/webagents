# Zapier Skill for WebAgents

Minimalistic Zapier integration for workflow automation. This skill allows you to securely manage Zapier API credentials and trigger Zaps from your WebAgent.

## Features

- 🔐 **Secure credential storage** via auth and KV skills
- ⚡ **Trigger Zaps** with custom input data
- 📋 **List Zaps** from your Zapier account
- 📊 **Monitor task status** in real-time
- 🔒 **Per-user isolation** with authentication
- 🛡️ **API key validation** during setup

## Prerequisites

1. **Zapier account** (free or paid)
2. **Zapier API key** (generated from Zapier account settings)
3. **WebAgents framework** with auth and kv skills enabled

## Quick Start

### 1. Set up Zapier API credentials

```python
# First, configure your Zapier credentials
await agent.call_tool("zapier_setup", 
    api_key="your_zapier_api_key_here"
)
```

### 2. List available Zaps

```python
# See all Zaps in your Zapier account
await agent.call_tool("zapier_list_zaps")
```

### 3. Trigger a Zap

```python
# Trigger a Zap with optional input data
await agent.call_tool("zapier_trigger",
    zap_id="zap_id_here",
    data={"name": "John", "email": "john@example.com"}  # Optional
)
```

### 4. Check task status

```python
# Monitor Zap execution
await agent.call_tool("zapier_status", 
    task_id="task_id_from_trigger_response"
)
```

## Tools Reference

### `zapier_setup(api_key)`
Set up Zapier API credentials securely.

**Parameters:**
- `api_key` (str): Your Zapier API key from account settings

**Returns:**
- Success message with account information
- Error message if API key is invalid

**Example:**
```python
result = await agent.call_tool("zapier_setup",
    api_key="AK_1a2b3c4d5e6f7g8h9i0j"
)
# ✅ Zapier credentials saved successfully!
# 🔑 API key configured
# 📊 Found 5 Zaps in your account
```

### `zapier_trigger(zap_id, data=None)`
Trigger a Zapier Zap with optional input data.

**Parameters:**
- `zap_id` (str): The ID of the Zap to trigger
- `data` (dict, optional): Input data to pass to the Zap

**Returns:**
- Success message with task ID and status
- Error message if Zap not found or trigger fails

**Example:**
```python
result = await agent.call_tool("zapier_trigger",
    zap_id="12345",
    data={"customer_name": "Jane Doe", "order_total": 99.99}
)
# ✅ Zap triggered successfully!
# 📋 Task ID: task_67890
# 📊 Status: triggered
```

### `zapier_list_zaps()`
List all available Zaps in your Zapier account.

**Parameters:** None

**Returns:**
- List of Zaps with names, IDs, status, and trigger apps
- Empty message if no Zaps found

**Example:**
```python
result = await agent.call_tool("zapier_list_zaps")
# 📋 Available Zapier Zaps:
# 
# 🟢 **Email to Slack** (ID: 12345)
#    📊 Status: on
#    🔗 Trigger: Gmail
# 
# 🔴 **Form to Spreadsheet** (ID: 67890)
#    📊 Status: off
#    🔗 Trigger: Typeform
# 
# 💡 Use zapier_trigger(zap_id, data) to trigger a Zap
```

### `zapier_status(task_id)`
Check the status of a Zap execution.

**Parameters:**
- `task_id` (str): The task ID returned from zapier_trigger

**Returns:**
- Detailed task status report
- Error message if task not found

**Example:**
```python
result = await agent.call_tool("zapier_status", task_id="task_67890")
# 📊 Zap Execution Status Report
# 🆔 Task ID: task_67890
# 🔧 Zap ID: 12345
# ✅ Status: success
# 🕐 Created: 2024-01-15T10:30:00Z
# 🕑 Updated: 2024-01-15T10:32:15Z
```

## Setup Guide

### Getting your Zapier API Key

1. Log into your Zapier account
2. Go to **Account Settings** → **Developer**
3. Click **Manage API Keys**
4. Click **Create API Key**
5. Give it a name (e.g., "WebAgent Integration")
6. Copy the generated API key

### Zapier API Limits

- **Free accounts**: 100 tasks/month
- **Paid accounts**: Based on your plan
- **Rate limits**: 1 request per second per API key

## Usage Examples

### Simple Zap Trigger
```python
# Setup (one-time)
await agent.call_tool("zapier_setup", api_key="your_api_key")

# Trigger a simple Zap
result = await agent.call_tool("zapier_trigger", zap_id="welcome_email_zap")
print(result)  # ✅ Zap triggered successfully!
```

### Zap with Input Data
```python
# Trigger Zap with custom data
lead_data = {
    "name": "John Smith",
    "email": "john@company.com",
    "company": "Acme Corp",
    "source": "website_form"
}

result = await agent.call_tool("zapier_trigger", 
    zap_id="lead_processing_zap",
    data=lead_data
)

# Monitor execution
task_id = "task_123"  # Extract from result
status = await agent.call_tool("zapier_status", task_id=task_id)
print(status)
```

### Discover and Trigger Zaps
```python
# First, see what Zaps are available
zaps = await agent.call_tool("zapier_list_zaps")
print(zaps)

# Trigger a specific Zap
result = await agent.call_tool("zapier_trigger", 
    zap_id="data_sync_zap",
    data={"sync_type": "full", "timestamp": "2024-01-15T10:00:00Z"}
)
```

### Error Handling Example
```python
try:
    result = await agent.call_tool("zapier_trigger", zap_id="test_zap")
    if "✅" in result:
        print("Zap triggered successfully!")
    elif "❌" in result:
        print(f"Error: {result}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Common Use Cases

### 1. Lead Processing
```python
# New lead from website form
lead_info = {
    "name": request.form["name"],
    "email": request.form["email"],
    "message": request.form["message"]
}

# Trigger Zap to add to CRM and send welcome email
await agent.call_tool("zapier_trigger", 
    zap_id="lead_processing", 
    data=lead_info
)
```

### 2. Order Fulfillment
```python
# New order processed
order_data = {
    "order_id": "ORD-123",
    "customer_email": "customer@example.com",
    "items": [{"name": "Product A", "qty": 2}],
    "total": 49.98
}

# Trigger fulfillment workflow
await agent.call_tool("zapier_trigger", 
    zap_id="order_fulfillment", 
    data=order_data
)
```

### 3. Data Synchronization
```python
# Sync data between systems
sync_data = {
    "source": "database",
    "destination": "crm",
    "record_count": 150,
    "sync_timestamp": datetime.now().isoformat()
}

await agent.call_tool("zapier_trigger", 
    zap_id="data_sync", 
    data=sync_data
)
```

## Error Handling

The skill provides clear error messages for common scenarios:

- **❌ Authentication required** - User not authenticated
- **❌ API key is required** - Empty or missing API key
- **❌ Invalid API key** - API key rejected by Zapier
- **❌ API key doesn't have required permissions** - Limited API key
- **❌ Zap not found** - Invalid Zap ID
- **❌ Permission denied** - Zap disabled or no access
- **❌ Task not found** - Invalid task ID

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
└── Zapier Skill
    ├── Credential Management
    ├── API Communication
    └── Zap Operations
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
- Check if the key has been revoked
- Ensure you copied the full key

### "API key doesn't have required permissions"
- Check if your Zapier plan supports API access
- Verify the API key has necessary permissions
- Contact Zapier support if issues persist

### "Zap not found"
- Use `zapier_list_zaps()` to see available Zaps
- Check if the Zap ID is correct
- Verify the Zap exists and is accessible

### "Permission denied"
- Check if the Zap is enabled/turned on
- Verify you have access to the Zap
- Ensure the Zap isn't paused or has errors

## Limitations

- Supports Zapier REST API v1 endpoints
- Limited to triggering existing Zaps (no Zap creation/modification)
- Rate limited by Zapier (1 request/second)
- Task status checking depends on Zapier's API availability
- Some Zap types may not support external triggering

## Zapier Integration Tips

1. **Test your Zaps** in Zapier before triggering via API
2. **Use descriptive Zap names** for easier identification
3. **Enable Zaps** before trying to trigger them
4. **Check Zap history** in Zapier for debugging
5. **Monitor task limits** to avoid hitting quotas

## Contributing

To extend this skill:

1. Add new tools following the existing pattern
2. Update tests in `tests/test_zapier_skill.py`
3. Update this documentation
4. Ensure proper error handling and user feedback

## License

Part of the WebAgents framework. See main project license.
