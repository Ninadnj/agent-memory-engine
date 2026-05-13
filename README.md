# Lightweight AI Agent Memory Scaffold

A lightweight file-based memory and handoff scaffold for AI coding agents.

## Repository Description

Switch between AI coding agents without losing context, duplicating instructions, or loading unnecessary history.

This repo is for developers who use tools like Codex, Claude Code, Gemini, or other coding agents in the same project. It gives agents a small set of files for project facts, current state, handoff notes, decisions, known issues, and compact work history.

It is not a framework. It does not run code, install dependencies, manage a database, or automate your workflow.

## What Problem It Solves

AI coding agents often lose context when you switch tools or start a new session. The usual workaround is to paste long summaries or reload too much history.

This scaffold keeps the important context in plain Markdown files so agents can read only what they need:

- stable project facts
- current active task
- next handoff step
- durable decisions
- unresolved issues
- compact completed history

`AGENTS.md` remains the source of truth for agent behavior. The other files provide memory, prompts, and optional task workflows.

## Folder Structure

```text
AGENTS.md
CLAUDE.md
HOWTOUSE.md
commands/
  start.md
  end.md
agent-memory/
  PROJECT.md
  STATE.md
  HANDOFF.md
  DECISIONS.md
  KNOWN_ISSUES.md
  WORKLOG.md
workflows/
  debug.md
  refactor.md
  deploy.md
```

## Quick Start

1. Copy these files into your project.
2. Fill in `agent-memory/PROJECT.md` with verified stable facts from your repo.
3. Start a session with `/start` if your agent supports repo commands.
4. If `/start` is not supported, paste `commands/start.md` into the agent.
5. End a session with `/end` if your agent supports repo commands.
6. If `/end` is not supported, paste `commands/end.md` into the agent.

## Start A Session

At the beginning of a session, the agent should:

- read `AGENTS.md`
- follow the Read Policy in `AGENTS.md`
- read `agent-memory/PROJECT.md`, `STATE.md`, and `HANDOFF.md` for non-trivial work
- summarize project context, active task, and next safe step
- inspect relevant files before editing

Use `commands/start.md` as a portable prompt when native slash commands are not available.

## End A Session

Before stopping or switching agents, the agent should:

- compact `agent-memory/STATE.md`
- rewrite `agent-memory/HANDOFF.md`
- append `agent-memory/WORKLOG.md` only if the repo changed meaningfully
- avoid updating `PROJECT.md` unless stable verified facts changed
- avoid updating `DECISIONS.md` unless a durable decision was made

Use `commands/end.md` as a portable prompt when native slash commands are not available.

## Memory Model

`STATE.md` and `HANDOFF.md` are current memory. They should be rewritten and compacted as work changes.

`PROJECT.md`, `DECISIONS.md`, `KNOWN_ISSUES.md`, and `WORKLOG.md` are collected memory over time:

- `PROJECT.md` stores only verified stable project facts.
- `DECISIONS.md` stores only durable architecture, product, data, security, or deployment decisions.
- `KNOWN_ISSUES.md` stores only unresolved problems.
- `WORKLOG.md` stores compact completed-task history and should stay short.

Do not load `WORKLOG.md` by default. Use it only when historical context is needed.

## What Not To Store

Do not store:

- secrets, tokens, passwords, or private keys
- raw logs
- speculative notes
- duplicate summaries
- stale guesses
- personal commentary
- full chat transcripts
- sensitive client, production, or operational details

Use placeholders for examples. Store only context that helps the next agent work safely.

## Adapting This In Another Repo

Keep the structure simple:

- Make `AGENTS.md` the primary rules file.
- Update `agent-memory/PROJECT.md` with verified project facts.
- Keep `STATE.md` focused on the current active task.
- Keep `HANDOFF.md` focused on the exact next step.
- Load workflow docs only for matching debug, refactor, or deploy work.
- Add project-specific rules only when they are durable and useful.

Avoid adding scripts, databases, dependencies, or automation unless your own project clearly needs them.
