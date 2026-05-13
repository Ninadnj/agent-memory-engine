# AGENTS.md

You are working inside this repository as an AI coding agent.

Make the smallest safe change that solves the real problem, preserve the existing system, verify your work, and leave clear handoff state for future sessions.

## Priorities

1. Correctness
2. Safety
3. Minimal change
4. Existing project style
5. Verification
6. Speed

## Read Policy

Always read:

- `AGENTS.md`

For non-trivial work, also read:

- `agent-memory/PROJECT.md`
- `agent-memory/STATE.md`
- `agent-memory/HANDOFF.md`

Read only when relevant:

- `agent-memory/DECISIONS.md` for architecture, data, security, deployment, or product behavior changes
- `agent-memory/KNOWN_ISSUES.md` when touching related areas
- `agent-memory/WORKLOG.md` only when historical context is needed
- `workflows/debug.md` only for bug fixing
- `workflows/refactor.md` only for refactors
- `workflows/deploy.md` only for deployment, infrastructure, CI, env vars, or hosting work

Code is the source of truth. Load the minimum useful context.

## Core Rules

- Do exactly what was asked.
- Do not add extra features.
- Do not refactor unrelated code.
- Do not reformat unrelated files.
- Do not change architecture unless explicitly asked.
- Match existing project style.
- Prefer the simplest implementation that solves the task.
- Every changed line must be explainable by the task.
- Ask only when ambiguity blocks safe implementation.
- State assumptions when they affect the solution.
- Be careful with auth, permissions, database, APIs, secrets, deployment, payments, and destructive actions.
- Never expose, log, or commit secrets.
- Do not add dependencies unless necessary.
- Do not claim verification passed unless it actually ran and passed.
- If verification cannot be run, say why.

## Code Quality Rules

- Write simple, maintainable code.
- Prefer boring, readable code over clever abstractions.
- Reuse existing helpers, utilities, components, constants, and patterns before creating new ones.
- Avoid duplicated validation, formatting, API calls, database queries, and business logic.
- Do not write 200 lines when 50 clear lines solve the problem.
- Keep functions small and focused.
- Keep files focused.
- Avoid deeply nested logic; use early returns when they improve clarity.
- Avoid hidden side effects.
- Keep business logic separate from UI, transport, persistence, and external APIs.
- Validate inputs at system boundaries.
- Use explicit names instead of comments explaining confusing code.
- Add comments only to explain why something exists, not what obvious code does.
- Avoid unnecessary abstractions, dependencies, and future-proofing for hypothetical requirements.

## Change Size Rule

- Choose the smallest safe implementation before editing.
- Reuse existing code first.
- Modify existing code before adding new files.
- Add a small helper only when it is justified by real duplication or clarity.
- Add a new file only when the existing structure clearly needs it.
- Add a dependency only as a last resort.
- If a change requires many files, explain why before editing.

## Before Editing

Briefly state:

- Task
- Relevant files to inspect
- Plan
- Assumptions
- Risk level

## During Work

Update `agent-memory/STATE.md` only when useful:

- task status changes
- scoped files change
- blocker appears
- next step changes

Do not write long progress logs.

## Memory Updates

Use memory to preserve useful state, not history for its own sake.

Update only the right file:

- `agent-memory/PROJECT.md`: verified stable project facts, commands, architecture map, env vars
- `agent-memory/STATE.md`: current active task only
- `agent-memory/HANDOFF.md`: exact continuation state for the next agent
- `agent-memory/DECISIONS.md`: durable architecture, product, data, security, or deployment decisions
- `agent-memory/KNOWN_ISSUES.md`: unresolved bugs, risks, or technical debt
- `agent-memory/WORKLOG.md`: compact completed-task history only when useful

Do not store secrets, raw logs, speculation, duplicate summaries, trivial changes, stale guesses, or personal commentary.

## End Of Session

Before stopping after non-trivial work:

1. Compact `agent-memory/STATE.md`.
2. Rewrite `agent-memory/HANDOFF.md`.
3. Append to `agent-memory/WORKLOG.md` only if the repo changed meaningfully.

## Self-Review

Before the final report, check the diff:

- Every changed line is related to the task.
- No duplicated existing logic.
- No unnecessary abstraction.
- No behavior changes outside the requested scope.
- No debug logs, dead code, temporary comments, or unused imports.
- Verification was actually run, or the reason it was not run is clearly stated.

## Final Report

After editing, report:

- Summary
- Files changed
- Verification
- Intentionally not changed
- Risks / follow-ups
- Memory updates
