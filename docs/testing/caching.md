---
title: Response Caching Strategy
---
# Response Caching Strategy

Compliance tests use LLM response caching for reproducible, cost-effective testing.

## Why Caching?

1. **Reproducibility** - Same inputs produce same outputs
2. **Cost reduction** - Avoid repeated API calls
3. **Speed** - Cached responses are instant
4. **Offline testing** - Run tests without API access

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Test Runner │────►│    Cache    │────►│  LLM API    │
│             │◄────│             │◄────│             │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │
       │    Cache Hit      │
       │◄──────────────────│
       │
       │    Cache Miss
       │──────────────────────────────────────────────►
                                                    │
       │◄─────────────────────────────────────────────
       │         Response (saved to cache)
```

## Cache Modes

### read_write (Default)

Best for CI/PR testing:

- Check cache first
- On miss, call LLM and save response
- Deterministic results

```bash
python -m compliance.runner --cache-mode read_write
```

### write_only

Best for nightly runs:

- Always call LLM
- Save responses to cache
- Detects regressions with fresh data

```bash
python -m compliance.runner --cache-mode write_only
```

### disabled

Best for pre-release:

- No caching
- Tests real LLM behavior
- Full variability

```bash
python -m compliance.runner --cache-mode disabled
```

## Cache Key Generation

Cache keys are generated from:

1. Model name
2. Messages (content only)
3. Temperature
4. Tools (if present)
5. Other deterministic parameters

```python
def cache_key(request: dict) -> str:
    key_data = {
        "model": request.get("model"),
        "messages": normalize_messages(request.get("messages", [])),
        "temperature": request.get("temperature", 0),
        "tools": normalize_tools(request.get("tools")),
    }
    return hashlib.sha256(json.dumps(key_data, sort_keys=True)).hexdigest()
```

## Temperature Strategy

| Test Type | Temperature | Cache | Rationale |
|-----------|-------------|-------|-----------|
| CI/PR | 0 | read_write | Deterministic |
| Nightly | 0.3 | write_only | Some variation |
| Pre-release | 0.7 | disabled | Full variation |

### Temperature = 0

- Most reproducible
- Same prompt → same response (usually)
- Best for CI

### Temperature > 0

- Introduces randomness
- Tests edge cases
- May reveal fragile assertions

## Cache Structure

```
compliance/.cache/
├── manifest.json          # Index of cached responses
├── openai/
│   ├── abc123.json        # Cached response
│   └── def456.json
├── anthropic/
│   └── ...
└── google/
    └── ...
```

### Cached Response Format

```json
{
  "request": {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0
  },
  "response": {
    "id": "chatcmpl-...",
    "choices": [...]
  },
  "metadata": {
    "cached_at": "2026-01-29T10:00:00Z",
    "ttl": 604800
  }
}
```

## Cache Invalidation

### Automatic

- Cache entries expire after TTL (default: 7 days)
- Model updates invalidate relevant entries

### Manual

```bash
# Clear all
rm -rf compliance/.cache/

# Clear by provider
rm -rf compliance/.cache/openai/

# Clear specific entry
rm compliance/.cache/openai/abc123.json
```

## CacheSkill Implementation

Each SDK implements its own `CacheSkill`:

### Python

```python
from pathlib import Path
import hashlib
import json

class CacheSkill(Skill):
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def cache_key(self, messages: list, **kwargs) -> str:
        """Generate deterministic cache key."""
        data = {
            "messages": [{"role": m["role"], "content": m["content"]} 
                         for m in messages],
            "model": kwargs.get("model"),
            "temperature": kwargs.get("temperature", 0),
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
    
    @hook("on_request_outgoing")
    async def check_cache(self, context: Context) -> Context:
        """Return cached response if available."""
        key = self.cache_key(context.messages, **context.options)
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            context.response = cached["response"]
            context.skip_llm = True
        
        return context
    
    @hook("finalize_connection")
    async def save_cache(self, context: Context) -> Context:
        """Cache the response for future runs."""
        if context.response and not getattr(context, "skip_llm", False):
            key = self.cache_key(context.messages, **context.options)
            cache_file = self.cache_dir / f"{key}.json"
            
            cache_file.write_text(json.dumps({
                "request": {
                    "messages": context.messages,
                    "model": context.options.get("model"),
                },
                "response": context.response,
                "metadata": {"cached_at": datetime.utcnow().isoformat()},
            }))
        
        return context
```

### TypeScript

```typescript
class CacheSkill extends Skill {
  constructor(private cacheDir: string = '.cache') {}
  
  private cacheKey(messages: Message[], options: any): string {
    const data = {
      messages: messages.map(m => ({ role: m.role, content: m.content })),
      model: options.model,
      temperature: options.temperature ?? 0,
    };
    return crypto.createHash('sha256')
      .update(JSON.stringify(data))
      .digest('hex');
  }
  
  @hook('on_request_outgoing')
  async checkCache(context: Context): Promise<Context> {
    // Check cache and set context.response if found
  }
  
  @hook('finalize_connection')
  async saveCache(context: Context): Promise<Context> {
    // Save response to cache
  }
}
```

## Best Practices

### 1. Commit Cache Selectively

```text
# .gitignore
compliance/.cache/

# But optionally commit known-good responses
!compliance/.cache/fixtures/
```

### 2. Cache Warming

Pre-populate cache in CI:

```yaml
- name: Warm cache
  run: |
    python -m compliance.runner --cache-mode write_only
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### 3. Cache Validation

Periodically verify cached responses are still valid:

```bash
python -m compliance.runner --cache-mode write_only --tag core
```

### 4. Provider-Specific Caching

Different providers may have different caching needs:

```python
cache_config = {
    "openai": {"ttl": 604800},      # 7 days
    "anthropic": {"ttl": 259200},   # 3 days
    "google": {"ttl": 86400},       # 1 day
}
```
