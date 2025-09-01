# Google Skills

Integrate with Google APIs including Calendar, with Gmail coming soon.

## Available Google Skills

### Google Calendar Skill

Comprehensive Google Calendar integration with OAuth 2.0 authentication and event management.

#### Features
- **OAuth 2.0 Authentication**: Secure user authorization flow
- **Calendar Event Listing**: View upcoming events from primary calendar
- **Token Management**: Automatic token refresh and secure storage
- **Multi-User Support**: Per-user authentication and data isolation
- **HTTP Callback Handler**: Seamless OAuth redirect handling

#### Quick Setup

```python
from webagents.agents import BaseAgent
from webagents.agents.skills.ecosystem.google.calendar import GoogleCalendarSkill

agent = BaseAgent(
    name="calendar-agent",
    model="openai/gpt-4o",
    skills={
        "google_calendar": GoogleCalendarSkill()
    }
)
```

#### Prerequisites

**Environment Variables:**
```bash
export GOOGLE_CLIENT_ID="your-google-oauth-client-id"
export GOOGLE_CLIENT_SECRET="your-google-oauth-client-secret"
export AGENTS_BASE_URL="http://localhost:2224"  # Your agent server base URL
```

**Google Cloud Console Setup:**
1. Create a project in [Google Cloud Console](https://console.cloud.google.com)
2. Enable the Google Calendar API
3. Create OAuth 2.0 credentials (Web application)
4. Add your redirect URI: `{AGENTS_BASE_URL}/agents/{agent-name}/oauth/google/calendar/callback`

#### Core Tools

**1. Calendar Event Listing**

**Tool**: `list_events(max_results=10)`

Lists upcoming events from the user's primary Google Calendar.

```python
# Example usage via LLM
messages = [{
    'role': 'user', 
    'content': 'Show me my upcoming calendar events for this week'
}]
response = await agent.run(messages=messages)
```

**Features:**
- Automatic OAuth authorization flow if not authenticated
- Token refresh handling for expired tokens
- Configurable number of results (default: 10)
- Chronological ordering of events
- Support for both datetime and all-day events

#### Authentication Flow

**1. First-Time Authorization:**
- When `list_events()` is called without authentication, it returns an authorization URL
- User opens the URL and grants calendar access permissions
- Google redirects to the callback endpoint with an authorization code
- The skill exchanges the code for access and refresh tokens
- Tokens are securely stored for future use

**2. Automatic Token Refresh:**
- Expired access tokens are automatically refreshed using the refresh token
- No user intervention required for token maintenance
- Seamless re-authentication for API calls

#### Usage Examples

**Calendar Event Summary:**
```python
# The LLM will automatically handle authentication and API calls
messages = [{
    'role': 'user',
    'content': 'What meetings do I have today and tomorrow?'
}]
response = await agent.run(messages=messages)
```

**Event Planning:**
```python
messages = [{
    'role': 'user',
    'content': 'Check my calendar for the next 5 events and suggest optimal times for a 1-hour meeting this week'
}]
response = await agent.run(messages=messages)
```

**Schedule Analysis:**
```python
messages = [{
    'role': 'user',
    'content': 'Analyze my upcoming calendar events and identify any scheduling conflicts or busy periods'
}]
response = await agent.run(messages=messages)
```

#### Configuration

**OAuth Scopes:**
- `https://www.googleapis.com/auth/calendar.readonly`
- `https://www.googleapis.com/auth/calendar.events.readonly`

**Token Storage:**
- Uses KV skill for secure token persistence
- Fallback to in-memory storage if KV skill unavailable
- Per-user token isolation with user ID-based keys

**Callback Endpoint:**
- Path: `/oauth/google/calendar/callback`
- Method: GET
- Handles OAuth authorization code exchange
- Returns user-friendly HTML confirmation page

#### Security Features

- **Secure Token Storage**: Tokens stored via KV skill with proper namespacing
- **User Isolation**: Each user's tokens stored separately by user ID
- **Scope Limitation**: Read-only calendar access only
- **Token Refresh**: Automatic handling of expired tokens
- **Error Handling**: Comprehensive error messages for troubleshooting

#### Troubleshooting

**Common Issues:**

**"Missing user identity"**
- Ensure the agent has access to user context
- Verify authentication middleware is properly configured

**"Not authorized"**
- User needs to complete OAuth flow by opening the provided authorization URL
- Check that Google OAuth credentials are properly configured

**"Permission error (403)"**
- Verify Google Calendar API is enabled in Google Cloud Console
- Check OAuth scopes are correctly configured
- Ensure user has granted necessary permissions

**"Token expired or invalid"**
- The skill will automatically attempt token refresh
- If refresh fails, user needs to re-authorize

#### API Reference

**GoogleCalendarSkill Methods:**
- `list_events(max_results: int = 10) -> str`: List upcoming calendar events
- OAuth callback handler for authorization code exchange
- Automatic token refresh and storage management

## Coming Soon

### Gmail Skill 🚧

Gmail integration is currently in development and will include:

- **Email Management**: Read, send, and organize emails
- **OAuth 2.0 Authentication**: Secure Gmail access
- **Search and Filter**: Advanced email search capabilities  
- **Attachment Handling**: Download and process email attachments
- **Label Management**: Organize emails with Gmail labels
- **Draft Management**: Create and manage email drafts

Stay tuned for the Gmail skill release in upcoming versions!

## Dependencies

**Google Calendar Skill:**
- `httpx`: HTTP client for Google API requests
- `json`: Token serialization and storage
- KV skill (optional): Secure token persistence

**Required Environment:**
- Google Cloud Console project with Calendar API enabled
- OAuth 2.0 web application credentials
- Properly configured redirect URIs

## Best Practices

### Security
- Always use environment variables for OAuth credentials
- Never commit client secrets to version control
- Use HTTPS in production for OAuth callbacks
- Implement proper user session management

### Performance
- Cache calendar data when appropriate
- Use reasonable `max_results` limits for API calls
- Implement error handling and retry logic
- Monitor API quota usage

### User Experience
- Provide clear authorization instructions
- Handle authentication errors gracefully
- Offer helpful error messages for common issues
- Support multiple calendar time zones

## Advanced Features

### Multi-Calendar Support
The current implementation focuses on the primary calendar, with multi-calendar support planned for future releases.

### Event Creation
Write capabilities (event creation, modification) are planned for future versions with appropriate OAuth scopes.

### Webhook Support
Real-time calendar change notifications via Google Calendar webhooks are under consideration for future releases. 