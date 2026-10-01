/**
 * The five memory tools, as the model sees them, the same in both SDKs
 * (gap-closure plan item 2.1, 2026-09-26; `memory_read` and the notes'
 * descriptions since 2026-09-29, when the prompt's notes became an index).
 * Pinned word for word by `python/tests/fixtures/memory_tool/definition.json`,
 * which the Python skill (`skills/local/memory/caller_scoped.py`) is checked
 * against too, so an agent file naming `memory` gives its model the same tools
 * under either CLI.
 */

export interface MemoryToolDefinition {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export const MEMORY_TOOL_DEFINITIONS: readonly MemoryToolDefinition[] = [
  {
    "type": "function",
    "function": {
      "name": "memory_search",
      "description": "Search your memory: the notes you saved and the summaries of earlier conversations. Results come only from the memory of the caller you are talking to, the notes shared with every caller, and, when the caller is your owner, every namespace. Each result has key, namespace, description, content and updated_at.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "What to look for, in a few words."
          },
          "namespace": {
            "type": "string",
            "description": "Owner only: search one namespace (owner, shared, or a caller namespace as memory_list names it)."
          },
          "limit": {
            "type": "integer",
            "description": "Most results to return (default 10, at most 50)."
          }
        },
        "required": [
          "query"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "memory_read",
      "description": "Read one note in full, by the key your memory index gives it. Answers with key, namespace, description, content and updated_at.",
      "parameters": {
        "type": "object",
        "properties": {
          "key": {
            "type": "string",
            "description": "The note's name."
          },
          "namespace": {
            "type": "string",
            "description": "Owner only: owner, shared, or a caller namespace as memory_list names it."
          }
        },
        "required": [
          "key"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "memory_write",
      "description": "Save a note to remember across conversations. key names the note (a short slug such as preferences or project-status: letters, digits, dots, dashes and underscores); writing an existing key replaces it. Give it a one-line description: that line is what your memory index shows in every conversation. The note is filed in the memory of the caller you are talking to. Only your owner may file under owner (the default in the owner's conversations) or shared (read by every caller).",
      "parameters": {
        "type": "object",
        "properties": {
          "key": {
            "type": "string",
            "description": "The note's name, a slug such as preferences.",
            "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
          },
          "content": {
            "type": "string",
            "description": "The note, in Markdown."
          },
          "description": {
            "type": "string",
            "description": "One line saying what the note is for, shown in your memory index. Without it, the note's first line is shown."
          },
          "namespace": {
            "type": "string",
            "enum": [
              "owner",
              "shared"
            ],
            "description": "Owner only: owner (the default) or shared."
          }
        },
        "required": [
          "key",
          "content"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "memory_forget",
      "description": "Delete a note by key from the memory of the caller you are talking to. Your owner may name the namespace to delete from. Answers with how many notes were removed.",
      "parameters": {
        "type": "object",
        "properties": {
          "key": {
            "type": "string",
            "description": "The note's name."
          },
          "namespace": {
            "type": "string",
            "description": "Owner only: owner, shared, or a caller namespace as memory_list names it."
          }
        },
        "required": [
          "key"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "memory_list",
      "description": "List the notes in memory, most recently updated first: key, namespace, description and updated_at. Callers see their own notes and the shared ones; your owner sees every namespace.",
      "parameters": {
        "type": "object",
        "properties": {
          "prefix": {
            "type": "string",
            "description": "Only keys starting with this."
          },
          "namespace": {
            "type": "string",
            "description": "Owner only: list one namespace."
          },
          "limit": {
            "type": "integer",
            "description": "Most notes to return (default 50, at most 200)."
          }
        },
        "required": []
      }
    }
  }
];
