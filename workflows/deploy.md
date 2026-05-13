# Deploy Workflow

Use this when changing deployment, infrastructure, env vars, build config,
Docker, CI, or hosting.

1. Identify the target environment.
2. Check required env vars.
3. Confirm build and runtime commands.
4. Check logs or failure output.
5. Make one infra change at a time.
6. Verify with a build, health check, or runtime logs.

Rules:

- Do not commit secrets.
- Do not hardcode local paths.
- Do not change production config casually.
- Document new env vars in `agent-memory/PROJECT.md`.
