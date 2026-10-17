"""A producer agent for the cross-SDK exchange probe: publishes a folder and
serves it from the producer endpoint to one audience, printing where the
manifest is. Run by the TypeScript probe (``scripts/artifact-exchange-probe.ts
--producer=python``), useful on its own to hand a folder to another agent.

    python -m webagents.artifacts.probe_server --folder DIR --port 8810 \\
        --public-url http://127.0.0.1:8810 --agent producer \\
        --audience https://consumer.example/agents/reda --task t1
"""

from __future__ import annotations

import argparse
import json
import sys

import uvicorn

from webagents.access.policy import parse_access
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.local.access.skill import AccessSkill
from webagents.artifacts import ArtifactProducer, ArtifactRegistry, ArtifactsSkill, files
from webagents.server.core.app import create_server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--public-url", required=True)
    parser.add_argument("--agent", default="producer")
    parser.add_argument("--audience", action="append", default=[])
    parser.add_argument("--task", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)

    registry = ArtifactRegistry()
    agent = BaseAgent(
        name=args.agent,
        instructions="Serves published artifacts.",
        skills={
            "access": AccessSkill({"policy": parse_access({"default": "public"})}),
            "artifacts": ArtifactsSkill(registry=registry),
        },
    )
    agent_url = f"{args.public_url.rstrip('/')}/{args.agent}"
    published = files.publish(
        args.folder,
        producer=ArtifactProducer(agent=agent_url, task=args.task),
        registry=registry,
        base_url=agent_url,
        audience=args.audience or None,
        allow_http=True,
    )
    server = create_server(agents=[agent], public_url=args.public_url, heartbeat=False, quiet=True)
    print("PUBLISHED " + json.dumps({"manifestUrl": published.manifest_url, "sha256": published.sha256, "agentUrl": agent_url}), flush=True)
    uvicorn.run(server.app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
