# HOWTOUSE.md

Short human guide for using this repository with Claude Code, Codex, or another AI coding agent.

## Start Session

Use `/start` if your agent supports repo slash commands. If not, paste `commands/start.md`.

Reusable prompt:

```text
Read AGENTS.md first, then follow its read policy.
Summarize the project context, active task, and next safe step.
Inspect relevant files before editing.
```

## End Session

Use `/end` if your agent supports repo slash commands. If not, paste `commands/end.md`.

Reusable prompt:

```text
Follow AGENTS.md end-of-session rules.
Compact agent-memory/STATE.md.
Rewrite agent-memory/HANDOFF.md with the exact next step.
Append agent-memory/WORKLOG.md only if the repo changed meaningfully.
```

## Memory Files

- `AGENTS.md` is the primary rules file.
- `agent-memory/PROJECT.md` stores verified stable project facts.
- `agent-memory/STATE.md` stores the current active task only.
- `agent-memory/HANDOFF.md` stores continuation state for the next agent.
- `agent-memory/DECISIONS.md`, `KNOWN_ISSUES.md`, and `WORKLOG.md` are read only when relevant.
- `workflows/` docs are loaded only for matching debug, refactor, or deploy tasks.
