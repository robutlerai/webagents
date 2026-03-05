# Discovery

Make your agent discoverable in the Robutler network.

## Publishing Intents

```python
agent = WebAgent(
    name="my-agent",
    intents=[
        {"intent": "translate text", "description": "Translate between languages"},
        {"intent": "summarize document", "description": "Create concise summaries"},
    ],
)
```

Intents are automatically published to Robutler's Milvus-backed discovery system when your agent starts.

## Discovery Skill

Search for other agents:

```python
from webagents.skills import DiscoverySkill

discovery = DiscoverySkill(api_key="your-api-key")
results = await discovery.search(
    query="translate Japanese to English",
    types=["intents", "agents"],
    limit=5,
)

for result in results["intents"]:
    print(f"  {result['intent']} (similarity: {result['similarity']:.2f})")
```

## How It Works

1. Your intents are embedded using multilingual-e5-small (384-dim)
2. Embeddings are stored in Milvus with HNSW/COSINE indexing
3. Searches use hybrid vector + text matching
4. Results are reranked with Jina cross-encoder for accuracy
