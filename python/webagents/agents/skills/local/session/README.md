# Session Skill

An agent keeps its conversations.

```yaml
skills:
  - session                        # conversations kept on this machine
  - session: {backend: robutler}   # and yours on Robutler too, as chats with the agent
```

- **In the chat** (`webagents`, and `-p`), conversations are kept under
  `~/.webagents/sessions/<folder>/<agent>/` and `/resume` continues them. With
  `backend: robutler` and the agent published, each is also your chat with
  the agent on Robutler, which `/resume` on another machine finds.
- **Served** (`webagents serve`, `webagents daemon`), the skill keeps each
  verified caller's conversation when the request names it with
  `metadata.session_id`: yours with the chat's, anyone else's under
  `callers/<hash>/`, one namespace per caller. Anonymous callers are not kept.

See `skill.py` for the rules; the TypeScript skill is the same.
