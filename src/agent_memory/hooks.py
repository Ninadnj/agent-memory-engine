"""Claude Code hooks: make memory happen without the agent remembering to.

Every tool in this package depends on the model *choosing* to call it. Models
forget, especially at the end of a session — the session simply ends. Hooks are
the deterministic half: the client runs them itself, so the read side needs no
model discipline at all, and the write side gets a floor of accurate history
even when nothing was written explicitly.

Three events are used (schemas: https://code.claude.com/docs/en/hooks):

* ``SessionStart`` — inject the previous session's handoff plus orientation
  memories, and record a marker of where the repository stood.
* ``UserPromptSubmit`` — inject memories relevant to what was just asked.
  Opt-in: it fires on every message and costs a model load each time.
* ``SessionEnd`` — diff against the marker and save what actually changed.

Two rules hold everywhere in this module:

1. **A hook must never break a session.** Every entry point catches everything
   and falls back to empty output.
2. **stdout is a protocol channel.** Only the hook's JSON goes there; anything
   informational goes to stderr.
"""

from __future__ import annotations

import json
import hashlib
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .embeddings import default_min_score
from .rendering import pack_blocks, recall_context
from .store import (
    PROJECT_STORE_DIR,
    MemoryStore,
    default_store_path,
    find_project_root,
)

# What SessionStart injects when there is no task to match against yet.
ORIENTATION_TYPES = ("project", "decision")
SESSION_START_BUDGET = 400
PROMPT_RECALL_BUDGET = 200

HOOK_COMMAND = "agent-memory hook"


# ---- shared helpers --------------------------------------------------------
def _sessions_dir(store_path: Path) -> Path:
    return store_path.parent / "sessions"


def _git(root: Path, *args: str) -> Optional[str]:
    """Run a read-only git command, or return None if git/the repo is unusable."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.rstrip("\r\n") if out.returncode == 0 else None


def _open_store(payload: dict) -> Optional[MemoryStore]:
    """Resolve the store for the directory the session is running in."""
    cwd = payload.get("cwd") or os.getcwd()
    try:
        path = default_store_path(cwd)
        return MemoryStore(path=path)
    except Exception as exc:  # a broken store must not take the session with it
        print(f"[agent-memory] could not open store: {exc}", file=sys.stderr)
        return None


def _context_output(event: str, context: str) -> dict:
    """A hook response that injects text into the model's context."""
    if not context:
        return {}
    return {
        "additionalContext": context,
        "hookSpecificOutput": {
            "hookEventName": event,
            "additionalContext": context,
        },
    }


# ---- SessionStart ----------------------------------------------------------
def session_start(payload: dict) -> dict:
    """Inject the last handoff plus orientation memories, and mark the repo state.

    No task is known yet — the user has not typed anything — so there is nothing
    to match against. What is always relevant at session start is the previous
    agent's handoff and a few durable project facts.
    """
    store = _open_store(payload)
    if store is None:
        return {}

    _write_marker(payload, store)

    blocks = []
    handoff = store.latest("handoff", fresh=True)
    if handoff is not None:
        blocks.append(f"Last handoff [{handoff.agent or 'unknown'}]: {handoff.text}")
    note = store.latest("worklog", fresh=True)
    if note is not None:
        blocks.append(f"Last session: {note.text}")
    orientation = [entry for entry in reversed(store.all())
                   if entry.type in ORIENTATION_TYPES and entry.status == "active"]
    blocks.extend(f"- [{entry.type}] {entry.text}" for entry in orientation)
    return _context_output("SessionStart", pack_blocks(blocks, SESSION_START_BUDGET, limit=7))


def _write_marker(payload: dict, store: MemoryStore) -> None:
    """Record where the repository stood, so SessionEnd can diff against it."""
    session_id = payload.get("session_id")
    root = find_project_root(payload.get("cwd") or os.getcwd())
    if not session_id or root is None:
        return
    marker = {
        "head": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "dirty_state": _dirty_state(root),
    }
    try:
        directory = _sessions_dir(store.path)
        directory.mkdir(parents=True, exist_ok=True)
        marker_file = _marker_file(store.path, session_id)
        marker_file.write_text(json.dumps(marker))
    except OSError:
        pass  # a missing marker only costs us the autosave


# ---- UserPromptSubmit ------------------------------------------------------
def user_prompt(payload: dict) -> dict:
    """Inject memories relevant to the prompt the user just submitted."""
    # `prompt` is the documented client field. Keep the previous alias for
    # scripts built against v0.3; the real field always takes precedence.
    prompt = payload.get("prompt", payload.get("user_input", ""))
    if not isinstance(prompt, str):
        print("[agent-memory] prompt field must be text", file=sys.stderr)
        return {}
    prompt = prompt.strip()
    if len(prompt) < 12:  # "yes", "continue" — nothing to match on
        return {}
    store = _open_store(payload)
    if store is None:
        return {}
    context = recall_context(store, prompt, k=3, budget=PROMPT_RECALL_BUDGET,
                             min_score=default_min_score(store.embedder))
    if not context:
        return {}
    return _context_output("UserPromptSubmit", context)


# ---- SessionEnd ------------------------------------------------------------
def session_end(payload: dict) -> dict:
    """Save what actually happened, derived from git rather than from the model.

    A model can write a better handoff than this, but only if it remembers to.
    This is the floor: accurate, boring, and always written.
    """
    store = _open_store(payload)
    if store is None:
        return {}
    session_id = payload.get("session_id")
    marker_file = _marker_file(store.path, session_id) if session_id else None

    marker = {}
    if marker_file is not None and marker_file.exists():
        try:
            marker = json.loads(marker_file.read_text())
        except (OSError, ValueError):
            marker = {}

    root = Path(marker.get("root") or "") if marker.get("root") else find_project_root(
        payload.get("cwd") or os.getcwd()
    )
    summary = _describe_session(root, marker) if root else None

    if marker_file is not None:
        try:
            marker_file.unlink(missing_ok=True)
        except OSError:
            pass

    if not summary:
        return {}  # nothing changed; do not pollute the store
    try:
        store.write(summary, type="worklog", agent=_agent_name(),
                    source={"event": "SessionEnd", "commit": _git(root, "rev-parse", "HEAD") or ""})
    except Exception:
        return {}
    return {"systemMessage": "agent-memory: saved a session note."}


def _describe_session(root: Path, marker: dict) -> Optional[str]:
    """One sentence about what changed, or None if nothing did."""
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD") or marker.get("branch")
    start_head = marker.get("head")
    head = _git(root, "rev-parse", "HEAD")

    commits: list[str] = []
    if start_head and head and start_head != head:
        log = _git(root, "log", "--format=%s", f"{start_head}..{head}")
        commits = [line for line in (log or "").splitlines() if line]

    state = _dirty_state(root)
    before = marker.get("dirty_state", {})
    dirty = sorted(path for path, fingerprint in state.items() if before.get(path) != fingerprint)

    if not commits and not dirty:
        return None

    bits = [f"Session on branch {branch}." if branch else "Session."]
    if commits:
        shown = "; ".join(commits[:3])
        more = f" (+{len(commits) - 3} more)" if len(commits) > 3 else ""
        bits.append(f"Committed: {shown}{more}.")
    if dirty:
        shown = ", ".join(dirty[:6])
        more = f" (+{len(dirty) - 6} more)" if len(dirty) > 6 else ""
        bits.append(f"Working-tree changes observed in: {shown}{more}.")
    return " ".join(bits)


def _marker_file(store_path: Path, session_id: str) -> Path:
    if not isinstance(session_id, str) or not session_id or len(session_id) > 200:
        raise ValueError("invalid hook session_id")
    # Keep conventional IDs readable; arbitrary client IDs cannot escape the
    # sessions directory or inject path components.
    safe = session_id if all(c.isalnum() or c in "-_" for c in session_id) else hashlib.sha256(session_id.encode()).hexdigest()
    return _sessions_dir(store_path) / f"{safe}.json"


def _dirty_state(root: Path) -> dict[str, str]:
    raw = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all") or ""
    records = iter(raw.split("\0"))
    state = {}
    for record in records:
        if len(record) < 4:
            continue
        status, path = record[:2], record[3:]
        if "R" in status or "C" in status:
            next(records, None)  # in -z form, the destination appears first
        if _is_store_path(path):
            continue
        full = root / path
        try:
            # Metadata avoids reading large/binary files or following symlinks.
            st = full.lstat()
            signature = f"{status}:{st.st_size}:{st.st_mtime_ns}"
        except OSError:
            signature = status + ":missing"
        state[path] = signature
    return state


def _is_store_path(path: str) -> bool:
    """Is this git path our own store rather than the user's work?

    The store lives inside the repository, so without this every session would
    report its own bookkeeping as changes the user made. Note git reports a
    wholly untracked directory as ``.agent_memory/``, with the trailing slash.
    """
    cleaned = path.strip('"')
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = cleaned.rstrip("/")
    return cleaned == PROJECT_STORE_DIR or cleaned.startswith(PROJECT_STORE_DIR + "/")


def _agent_name() -> str:
    return os.environ.get("AGENT_MEMORY_AGENT", "claude-code")


# ---- settings.json wiring --------------------------------------------------
EVENT_HANDLERS = {
    "SessionStart": session_start,
    "UserPromptSubmit": user_prompt,
    "SessionEnd": session_end,
}

_CLI_NAME = {
    "SessionStart": "session-start",
    "UserPromptSubmit": "user-prompt",
    "SessionEnd": "session-end",
}


def _executable() -> str:
    """Absolute path to this installation's CLI, falling back to the bare name.

    Hooks are spawned by the client, which may not have the virtualenv that
    `agent-memory` lives in on its PATH. Pinning the path we are currently
    running from makes the hook work regardless.
    """
    import shutil

    found = shutil.which("agent-memory")
    if found:
        return found
    candidate = Path(sys.executable).with_name("agent-memory")
    return str(candidate) if candidate.exists() else "agent-memory"


def _hook_entry(event: str) -> dict:
    entry = {
        "type": "command",
        "command": f"{subprocess.list2cmdline([_executable()]) if os.name == 'nt' else shlex.quote(_executable())} hook {_CLI_NAME[event]}",
    }
    if event == "UserPromptSubmit":
        entry["timeout"] = 20  # the event's own limit is 30s
    return entry


def _is_ours(hook: dict) -> bool:
    """Match our hooks whether they were written bare or as an absolute path."""
    if not isinstance(hook, dict):
        return False
    return HOOK_COMMAND in str(hook.get("command", ""))


def install(settings_path: Path, events: list[str]) -> list[str]:
    """Add our hooks to a settings file, preserving everything already there.

    Idempotent: re-running replaces our own entries rather than stacking copies,
    and hooks belonging to other tools are never touched.
    """
    settings = _load_settings(settings_path)
    hooks = settings.setdefault("hooks", {})
    changes: list[str] = []

    for event in EVENT_HANDLERS:
        groups = hooks.get(event, [])
        if not isinstance(groups, list):
            continue
        # Drop any previous version of our own hook for this event.
        cleaned = []
        for group in groups:
            if not isinstance(group, dict):
                cleaned.append(group)
                continue
            kept = [h for h in group.get("hooks", []) if not _is_ours(h)]
            if kept:
                cleaned.append({**group, "hooks": kept})
            elif not group.get("hooks"):
                cleaned.append(group)
        if event in events:
            entry = _hook_entry(event)
            cleaned.append({"hooks": [entry]})
            changes.append(f"{event} -> {entry['command']}")
        if cleaned:
            hooks[event] = cleaned
        else:
            hooks.pop(event, None)

    if not hooks:
        settings.pop("hooks", None)
    _save_settings(settings_path, settings)
    return changes


def uninstall(settings_path: Path) -> list[str]:
    """Remove only our hooks, leaving any others in place."""
    return install(settings_path, events=[])


def installed_events(settings_path: Path) -> list[str]:
    settings = _load_settings(settings_path)
    found = []
    for event, groups in (settings.get("hooks") or {}).items():
        if not isinstance(groups, list):
            continue
        if any(
            _is_ours(h)
            for group in groups
            if isinstance(group, dict)
            for h in group.get("hooks", [])
        ):
            found.append(event)
    return found


def _load_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"{path} is not valid JSON ({exc}). Fix or move it before installing hooks."
        )
    return data if isinstance(data, dict) else {}


def _save_settings(path: Path, settings: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(settings, indent=2) + "\n")
    os.replace(tmp, path)


# ---- entry point used by the CLI ------------------------------------------
def run(event: str, stdin=None, stdout=None) -> int:
    """Read the hook payload, emit the response, and never fail loudly.

    A raised exception here would surface as a hook error in the user's session
    for something as minor as an unreadable store, so everything is swallowed
    and reported on stderr.
    """
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    try:
        raw = stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}

    try:
        result = EVENT_HANDLERS[event](payload) or {}
    except Exception as exc:  # never break the session
        print(f"[agent-memory] {event} hook failed: {exc!r}", file=sys.stderr)
        result = {}

    if result:
        json.dump(result, stdout)
        stdout.write("\n")
    return 0
