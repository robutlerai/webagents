"""
A usage-only stream chunk does not end the turn in an error (2026-09-24).

OpenAI sends a last chunk with `choices: []` and the usage when a request asks
for it (`stream_options.include_usage`), and OpenAI-compatible servers send
one anyway. `_reconstruct_response_from_chunks` read `choices[0]` on it, so
every such turn failed with "list index out of range": `webagents -p`, the
chat and `webagents serve` alike. Found by the first-time-user journey.
"""

from webagents.agents.core.base_agent import BaseAgent


def test_a_usage_only_chunk_is_read_past():
    agent = BaseAgent(name="t", instructions="x")
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello "}}]},
        {"choices": [{"index": 0, "delta": {"content": "there."}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
    ]
    response = agent._reconstruct_response_from_chunks(chunks)
    assert response["choices"][0]["message"]["content"] == "Hello there."
