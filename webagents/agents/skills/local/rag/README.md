# RAG Skill

The RAG (Retrieval Augmented Generation) Skill provides local vector search capabilities for agents.

## Features

- **Local Vector Store**: Uses ChromaDB to store embeddings locally in `~/.webagents/agents/{name}/rag`.
- **Embeddings**: Uses `sentence-transformers` for local, private embedding generation.
- **Metadata Filtering**: Supports searching with metadata filters (date, type, etc.).

## Configuration

In your `AGENT.md`:

```yaml
skills:
  - rag
```

## Tools

### `index_file`
 indexes a text file into the knowledge base.
- `file_path`: Path to file.
- Automatically chunks content and extracts metadata.

### `search_knowledge`
Searches the knowledge base.
- `query`: Search text.
- `n_results`: Number of results.
- `where`: Optional metadata filter (e.g., `{"extension": ".md"}`).

### `clear_knowledge`
Wipes the agent's knowledge base.
