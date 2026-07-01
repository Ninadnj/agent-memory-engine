# Debug Workflow

Use this only when fixing a bug.

1. Reproduce or identify the failure.
2. Locate the smallest likely cause.
3. Explain the cause before editing.
4. Make the smallest safe fix.
5. Add or update a regression test if practical.
6. Run verification.

Rules:

- Do not patch symptoms without identifying the cause.
- Do not introduce broad refactors while debugging.
- Do not hide errors with empty catch blocks or fallback defaults.
- Do not change unrelated behavior.
