# X.com (Twitter) OAuth 1.0a Skill

A comprehensive X.com integration skill for WebAgents that implements OAuth 1.0a User Context authentication, enabling multitenant applications with per-user rate limits.

## Features

- **OAuth 1.0a User Context Authentication**: Full 3-legged OAuth flow
- **Per-User Rate Limits**: Scale rate limits with your user base instead of app-wide limits
- **Auth & KV Skill Integration**: Secure credential storage using WebAgents auth and KV skills
- **Real-time Webhook Monitoring**: Monitor posts, mentions, likes, follows, and DMs
- **Agent/LLM-Powered Responses**: Automatic agent/LLM responses to webhook events
- **Smart Notification System**: Configurable notifications with keyword filtering
- **Comprehensive API Coverage**: Post tweets, search, user info, rate limit tracking
- **Rate Limit Management**: Automatic rate limit header tracking and error handling
- **Multitenant Ready**: Designed for applications serving multiple users with isolated data

## Setup

### 1. Environment Variables

Set the following environment variables:

```bash
# Required: X.com API credentials
X_API_KEY=your_api_key_here
X_API_SECRET=your_api_secret_here

# Optional: Base URL for OAuth callbacks (defaults to http://localhost:2224)
AGENTS_BASE_URL=https://your-domain.com
```

### 2. X.com Developer Setup

1. Create a developer account at [developer.x.com](https://developer.x.com)
2. Create a new project and app
3. Get your API Key and API Secret
4. Configure OAuth 1.0a settings in your app:
   - **App permissions**: Read and Write (or as needed)
   - **Callback URLs**: Add your callback URL pattern:
     - `http://localhost:2224/agents/{agent-name}/oauth/x/callback` (development)
     - `https://your-domain.com/agents/{agent-name}/oauth/x/callback` (production)

### 3. Agent Configuration

Add the skill to your agent along with required auth and KV skills:

```python
from webagents.agents.skills.ecosystem.x_com.skill import XComSkill
from webagents.agents.skills.robutler.auth.skill import AuthSkill
from webagents.agents.skills.robutler.kv.skill import KVSkill

# In your agent initialization
agent = BaseAgent(
    name="x-com-agent",
    model="openai/gpt-4o",
    skills={
        "auth": AuthSkill(),  # Required for user authentication
        "kv": KVSkill(),      # Required for secure credential storage
        "x_com": XComSkill()  # X.com integration
    }
)
```

## Usage

### Authentication Flow

1. **Initialize Authentication**:
   ```python
   # User calls this tool to start OAuth flow
   result = await agent.call_tool("init_x_auth")
   # Returns: "🔗 Open this link to authorize X.com access: https://api.x.com/oauth/authorize?..."
   ```

2. **User Authorization**: User opens the provided URL and authorizes your app

3. **Automatic Token Storage**: After authorization, tokens are automatically stored per user

### Available Tools

#### Core Authentication & API Tools

#### `init_x_auth()`
Initialize OAuth authentication flow. Returns authorization URL for the user.

**Returns**: Authorization URL string

#### `post_tweet(text: str)`
Post a tweet to X.com.

**Parameters**:
- `text`: Tweet content (up to 280 characters)

**Returns**: Success message with tweet ID

#### `search_tweets(query: str, max_results: int = 10)`
Search for recent tweets.

**Parameters**:
- `query`: Search query string
- `max_results`: Maximum number of tweets to return (1-100)

**Returns**: Formatted list of tweets with metrics

#### `get_user_info()`
Get information about the authenticated user.

**Returns**: User profile information including followers, tweets, etc.

#### `get_rate_limits()`
Check current rate limit status for the authenticated user.

**Returns**: Rate limit status information

#### Webhook Management Tools (Owner-Only)

#### `setup_webhook_monitoring(keywords: List[str] = None, mentions_only: bool = False, send_notifications: bool = True)`
Set up webhook monitoring for X.com posts and events.

**Parameters**:
- `keywords`: List of keywords to monitor (optional)
- `mentions_only`: Only trigger on mentions of authenticated user
- `send_notifications`: Whether to send notifications for events

**Returns**: Configuration summary and webhook URL

#### `get_webhook_config()`
Get current webhook monitoring configuration.

**Returns**: Current webhook configuration details

#### `update_webhook_config(keywords: List[str] = None, mentions_only: bool = None, send_notifications: bool = None)`
Update webhook monitoring configuration.

**Parameters**: Same as setup_webhook_monitoring (all optional)

**Returns**: Success confirmation

#### `disable_webhook_monitoring()`
Disable webhook monitoring.

**Returns**: Confirmation message

#### `get_notifications(limit: int = 10)`
Get recent notifications from webhook events.

**Parameters**:
- `limit`: Maximum number of notifications to return

**Returns**: List of recent notifications with read status

#### `mark_notifications_read()`
Mark all notifications as read.

**Returns**: Confirmation message

## Rate Limits

### Per-User vs App-Wide Limits

- **OAuth 1.0a User Context** (this skill): ~900 requests per 15-minute window **per user**
- **OAuth 2.0 Bearer Token**: ~450 requests per 15-minute window for entire app

With 100 authenticated users, you get 90,000 requests per 15-minute window instead of 450!

### Rate Limit Headers

The skill automatically tracks rate limit headers from X.com API responses:
- `x-rate-limit-limit`: Total requests allowed
- `x-rate-limit-remaining`: Requests remaining in current window
- `x-rate-limit-reset`: Unix timestamp when limits reset

### Shared User Limits

⚠️ **Important**: Per-user rate limits are shared across ALL applications the user has authorized, including X's own apps. If a user is active on multiple X.com applications, their rate limit is consumed across all of them.

## Security

### Token Storage

- Tokens are stored using the KV skill with namespace "auth"
- Fallback to secure in-memory storage if KV skill unavailable
- All tokens are associated with user IDs from the request context

### OAuth Security

- Uses HMAC-SHA1 signatures for all requests
- Implements proper nonce and timestamp generation
- Secure state parameter for callback correlation
- Request token secrets are temporarily stored for callback processing

## Error Handling

The skill provides comprehensive error handling:

- **Authentication Errors**: Clear messages when tokens are missing or expired
- **Rate Limit Errors**: Specific handling for 429 responses
- **API Errors**: Detailed error messages from X.com API
- **Network Errors**: Timeout and connection error handling

## Testing

Run the test suite:

```bash
cd /home/vs/webagents
python -m pytest tests/test_x_com_skill.py -v
```

The tests cover:
- OAuth 1.0a signature generation
- Token storage and retrieval
- All API endpoints
- Error conditions
- Rate limit handling

## Architecture

### OAuth 1.0a Flow

1. **Request Token**: Get temporary credentials from X.com
2. **User Authorization**: Redirect user to X.com for authorization
3. **Access Token**: Exchange authorized request token for access token
4. **API Requests**: Use access token for all subsequent API calls

### Multitenant Design

- User tokens stored per user ID from request context
- Each user gets their own rate limit allocation
- Secure token isolation between users
- Automatic cleanup of temporary tokens

### Integration with WebAgents

- Follows WebAgents skill architecture patterns
- Uses standard decorators: `@tool`, `@prompt`, `@http`
- Integrates with context system for user identification
- Supports KV skill for persistent storage

### Webhook Architecture

#### Event Flow
1. **X.com sends webhook** → Agent endpoint (`/webhook/x/events`)
2. **Signature verification** → HMAC-SHA256 validation
3. **Event processing** → Extract and categorize event data
4. **Agent/LLM trigger** → Generate contextual response
5. **Notification delivery** → Store and optionally notify user

#### Supported Event Types
- **Tweet Events**: New tweets, retweets, quote tweets
- **Engagement Events**: Likes, replies, mentions
- **Follow Events**: New followers, unfollows
- **Direct Messages**: Incoming DMs
- **User Events**: Profile updates, blocking

#### Security Features
- **Webhook Signature Verification**: HMAC-SHA256 signature validation
- **Challenge Response Check (CRC)**: Automatic webhook validation
- **User Isolation**: Per-user webhook configurations and tokens
- **Scope-Based Access**: Owner-only webhook management tools
- **Secure Storage**: All configurations stored via KV skill

## Examples

### Basic Tweet Posting

```python
# User authenticates first
auth_result = await agent.call_tool("init_x_auth")
# User opens URL and authorizes

# Post a tweet
result = await agent.call_tool("post_tweet", text="Hello from WebAgents! 🤖")
# Returns: "✅ Tweet posted successfully! ID: 1234567890"
```

### Search and Analyze

```python
# Search for tweets
tweets = await agent.call_tool("search_tweets", query="webagents", max_results=5)
# Returns formatted list of tweets with engagement metrics

# Get user info
user_info = await agent.call_tool("get_user_info")
# Returns: "👤 **User Name** (@username) - ✅ Verified\n📝 Bio text\n..."
```

### Webhook Monitoring Setup

```python
# Set up webhook monitoring for specific keywords
result = await agent.call_tool("setup_webhook_monitoring", 
                               keywords=["webagents", "AI", "automation"],
                               mentions_only=False,
                               send_notifications=True)
# Returns: "✅ Webhook monitoring set up successfully! ..."

# Monitor only mentions of the authenticated user
result = await agent.call_tool("setup_webhook_monitoring", 
                               mentions_only=True)
```

### Webhook Configuration Management

```python
# Check current webhook configuration
config = await agent.call_tool("get_webhook_config")
# Returns: "📡 Webhook Monitoring Configuration\nStatus: 🟢 Active..."

# Update configuration
result = await agent.call_tool("update_webhook_config", 
                               keywords=["new", "keywords"],
                               send_notifications=False)

# Disable monitoring
result = await agent.call_tool("disable_webhook_monitoring")
```

### Notification Management

```python
# Get recent notifications
notifications = await agent.call_tool("get_notifications", limit=5)
# Returns: "🔔 Recent Notifications:\n1. 🔴 [tweet_created] ..."

# Mark all notifications as read
result = await agent.call_tool("mark_notifications_read")
# Returns: "✅ All notifications marked as read"
```

### Complete Multitenant Workflow

```python
# 1. User authenticates
auth_url = await agent.call_tool("init_x_auth")
# User completes OAuth flow

# 2. Set up monitoring
webhook_result = await agent.call_tool("setup_webhook_monitoring", 
                                      keywords=["support", "help", "issue"],
                                      send_notifications=True)

# 3. Agent automatically responds to relevant tweets
# (Webhook events trigger agent/LLM responses automatically)

# 4. Check notifications periodically
notifications = await agent.call_tool("get_notifications")

# 5. Post responses or take actions
response = await agent.call_tool("post_tweet", 
                                text="Thanks for reaching out! We're here to help. 🤝")
```

### Rate Limit Monitoring

```python
# Check current rate limits
limits = await agent.call_tool("get_rate_limits")
# Returns rate limit status for monitoring
```

## Troubleshooting

### Common Issues

1. **"❌ X.com API credentials not configured"**
   - Ensure `X_API_KEY` and `X_API_SECRET` environment variables are set

2. **"❌ Authentication expired"**
   - User needs to re-run `init_x_auth()` to refresh tokens

3. **"❌ Rate limit exceeded"**
   - User has hit their per-user rate limit, wait for reset

4. **OAuth callback errors**
   - Verify callback URL is correctly configured in X.com app settings
   - Check `AGENTS_BASE_URL` environment variable

### Debugging

Enable debug logging:

```python
import logging
logging.getLogger('skill.x_com').setLevel(logging.DEBUG)
```

## Contributing

When contributing to this skill:

1. Follow the existing code patterns
2. Add tests for new functionality
3. Update this README for new features
4. Ensure proper error handling
5. Test with real X.com API credentials

## License

This skill is part of the WebAgents framework and follows the same license terms.
