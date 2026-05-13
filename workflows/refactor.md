# Refactor Workflow

Use this only when the task is explicitly a refactor or when a refactor is required for the requested change.

1. State the behavior that must remain unchanged.
2. Identify files in scope.
3. Make small mechanical changes first.
4. Run tests or checks.
5. Make semantic cleanup only after behavior is protected.

Rules:

- No behavior change unless explicitly requested.
- No large rewrite without a migration plan.
- Preserve public APIs unless instructed.
- Prefer boring clarity over clever abstraction.
