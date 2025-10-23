# File Storage Skill

Comprehensive file management with harmonized API for storing, retrieving, and managing files.

## Overview

The `RobutlerFilesSkill` provides file management capabilities using the harmonized content API. It allows agents to download files from URLs, store files from base64 data, and list accessible files with proper scope-based access control.

## Features

- **URL-based File Storage**: Download and store files directly from URLs
- **Base64 File Upload**: Store files from base64 encoded data
- **File Listing**: List files with scope-based filtering (public/private)
- **Agent Name Prefixing**: Automatically prefixes uploaded files with agent name
- **Visibility Control**: Support for public, private, and shared file visibility
- **Owner-Scoped Uploads**: File upload operations restricted to agent owners
- **Harmonized API**: Uses the new `/api/content/agent` endpoints for efficient operations

## Usage

### Basic Setup

```python
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.storage.files.skill import RobutlerFilesSkill

agent = BaseAgent(
    name="file-agent",
    model="openai/gpt-4o-mini",
    skills={
        "files": RobutlerFilesSkill()
    }
)
```

### File Operations

```python
# Download and store a file from URL
response = await agent.run(messages=[
    {"role": "user", "content": "Download and store the image from https://example.com/image.jpg"}
])

# List all accessible files
response = await agent.run(messages=[
    {"role": "user", "content": "Show me all my files"}
])
```

## Tool Reference

### `store_file_from_url`

Download and store a file from a URL.

**Parameters:**

- `url` (str, required): URL to download file from
- `filename` (str, optional): Custom filename (auto-detected if not provided)
- `description` (str, optional): Description of the file
- `tags` (List[str], optional): List of tags for the file
- `visibility` (str, optional): File visibility - "public", "private", or "shared" (default: "private")

**Returns:**

JSON string with storage result including:
- `success`: Boolean indicating success/failure
- `id`: File ID in the system
- `filename`: Stored filename (with agent prefix)
- `url`: Public URL for accessing the file
- `size`: File size in bytes
- `content_type`: MIME type of the file
- `visibility`: File visibility setting
- `source_url`: Original URL the file was downloaded from

**Scope:** `owner` - Only the agent owner can upload files

### `store_file_from_base64`

Store a file from base64 encoded data.

**Parameters:**

- `filename` (str, required): Name of the file
- `base64_data` (str, required): Base64 encoded file content
- `content_type` (str, optional): MIME type of the file (default: "application/octet-stream")
- `description` (str, optional): Description of the file
- `tags` (List[str], optional): List of tags for the file
- `visibility` (str, optional): File visibility - "public", "private", or "shared" (default: "private")

**Returns:**

JSON string with storage result including:
- `success`: Boolean indicating success/failure
- `id`: File ID in the system
- `filename`: Stored filename (with agent prefix)
- `url`: Public URL for accessing the file
- `size`: File size in bytes
- `content_type`: MIME type of the file
- `visibility`: File visibility setting

**Scope:** `owner` - Only the agent owner can upload files

### `list_files`

List files accessible by the current agent with scope-based filtering.

**Parameters:**

- `scope` (str, optional): Scope filter - "public", "private", or None (all files for owner)

**Returns:**

JSON string with file list including:
- `success`: Boolean indicating success/failure
- `agent_name`: Name of the agent
- `total_files`: Number of files returned
- `files`: Array of file objects with details

**Scope:** Available to all users, but results filtered based on ownership:
- **Agent owner**: Can see all files (public + private) or filter by scope
- **Non-owner**: Only sees public files regardless of scope parameter

**Pricing:** 0.005 credits per call

## File Visibility Levels

- **private**: Only visible to the agent owner
- **public**: Visible to anyone who can access the agent
- **shared**: Visible to authorized users (implementation-dependent)

## Configuration

### Environment Variables

- `ROBUTLER_API_URL`: Portal API base URL (default: "http://localhost:3000")
- `ROBUTLER_CHAT_URL`: Chat server base URL for public content (default: "http://localhost:3001")
- `WEBAGENTS_API_KEY`: Default API key if not provided in config

### Skill Configuration

```python
config = {
    "portal_url": "https://robutler.ai",
    "chat_base_url": "https://chat.robutler.ai",
    "api_key": "your-api-key"
}

files_skill = RobutlerFilesSkill(config)
```

## Example Integration

```python
from webagents import Skill, tool

class ImageProcessingSkill(Skill):
    @tool
    async def process_image_from_url(self, image_url: str) -> str:
        # Download and store the image
        store_result = await self.discover_and_call(
            "files", 
            "store_file_from_url",
            image_url,
            description="Image for processing"
        )
        
        # Parse the result
        import json
        result = json.loads(store_result)
        
        if result["success"]:
            # Process the stored image
            file_url = result["url"]
            return f"✅ Image stored and ready for processing: {file_url}"
        else:
            return f"❌ Failed to store image: {result['error']}"
    
    @tool
    async def list_my_images(self) -> str:
        # List only public image files
        files_result = await self.discover_and_call("files", "list_files", "public")
        
        import json
        result = json.loads(files_result)
        
        if result["success"]:
            image_files = [f for f in result["files"] 
                          if f["content_type"].startswith("image/")]
            return f"Found {len(image_files)} image files"
        else:
            return f"❌ Failed to list files: {result['error']}"
```

## Security

- **Owner-Only Uploads**: File upload operations are restricted to agent owners
- **Scope-Based Access**: File listing respects ownership and visibility settings
- **Agent Isolation**: Files are associated with specific agents
- **API Key Authentication**: Uses secure agent API keys for all operations
- **Automatic Prefixing**: Agent names are automatically prefixed to prevent conflicts

## Error Handling

The skill provides comprehensive error handling:

- **Download Failures**: Returns detailed error messages for URL download issues
- **Upload Failures**: Handles API upload errors with descriptive messages
- **Authentication Errors**: Manages missing or invalid API keys
- **Network Issues**: Provides meaningful error responses for connectivity problems
- **Invalid Data**: Handles malformed base64 data and other input validation

## Advanced Features

### Agent Name Prefixing

All uploaded files are automatically prefixed with the agent name to prevent conflicts:
- Original: `image.jpg`
- Stored as: `my-agent_image.jpg`

### URL Rewriting

Public content URLs are automatically rewritten to point to the chat server for optimal delivery:
- Portal URL: `http://localhost:3000/api/content/public/...`
- Rewritten: `http://localhost:3001/api/content/public/...`

### Ownership Detection

The skill automatically detects whether the current user is the actual owner of the agent for proper access control, not just checking admin privileges.

## Dependencies

- **Agent API Key**: Requires valid agent API key for portal authentication
- **Portal Connectivity**: Requires network access to Robutler portal API endpoints
- **RobutlerClient**: Uses the official Robutler client for API operations
- **Agent Context**: Requires proper agent initialization and context
