from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.auth.skill import AuthSkill
from webagents.server.core.app import create_server
import uvicorn


def build_agent() -> BaseAgent:
    return BaseAgent(
        name="assistant",
        instructions="You are a helpful AI assistant.",
        model="openai/gpt-4o-mini",
    )


if __name__ == "__main__":
    agent = build_agent()
    server = create_server(agents=[agent])
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
