"""Tests for the MCP server.

This is the headline feature and it used to have no coverage at all, which is
how it came to be broken against `mcp` 2.x: the package renamed the server
class, the import failed, and nothing noticed. These tests exercise the tools
through the real server object on whatever `mcp` version is installed.
"""

import asyncio
import json

import pytest

mcp = pytest.importorskip("mcp", reason="needs the optional 'mcp' extra")

from agent_memory.mcp_server import build_server  # noqa: E402


def call(server, name: str, **args) -> str:
    """Invoke a tool and return its text, across mcp 1.x and 2.x result shapes."""
    result = asyncio.run(server.call_tool(name, args))
    content = getattr(result, "content", None)  # mcp 2.x: CallToolResult
    if content is None:  # mcp 1.x: list, or (list, dict)
        content = result[0] if isinstance(result, tuple) else result
    return content[0].text


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_EMBEDDER", "hashing")
    return build_server(store_path=tmp_path / "store.json", agent="claude-code")


def test_server_builds_and_exposes_the_documented_tools(server):
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {
        "memory_write",
        "memory_recall",
        "memory_boot",
        "memory_handoff",
        "memory_update",
        "memory_forget",
        "memory_list",
        "memory_stats",
        "memory_get",
        "memory_supersede",
    } <= names


def test_write_then_recall_roundtrip(server):
    assert "Saved" in call(server, "memory_write", text="Bookings are stored in UTC.", type="decision")
    out = call(server, "memory_recall", query="how are timezones handled for bookings")
    assert "UTC" in out


def test_write_reports_duplicates_instead_of_claiming_a_save(server):
    text = "Admin routes are guarded by requireAdmin in server/auth.ts."
    first = call(server, "memory_write", text=text, type="decision")
    second = call(server, "memory_write", text=text, type="decision")
    assert first.startswith("Saved")
    assert "Not saved" in second and "exact duplicate" in second


def test_write_rejects_unknown_type(server):
    assert "Error" in call(server, "memory_write", text="something", type="nope")


def test_recall_filters_out_unrelated_memories(server):
    call(server, "memory_write", text="Bookings are stored in UTC.", type="decision")
    call(server, "memory_write", text="The chatbot uses Google Gemini.", type="decision")
    assert call(server, "memory_recall", query="how do I bake sourdough bread") == (
        "No relevant memories."
    )


def test_boot_never_exceeds_its_budget(server):
    call(
        server,
        "memory_handoff",
        done="Finished the booking refactor across the router, the email helper "
        "and the calendar sync, and re-ran the integration suite twice.",
        next_steps="Rate limit the chat endpoint before deploying to production.",
    )
    for text in [
        "Bookings are stored in UTC and converted in the UI layer only.",
        "The chatbot uses Google Gemini and its prompt lives in gemini-chat.ts.",
        "Admin routes are guarded by requireAdmin in server/auth.ts.",
        "Uploaded images go to object storage with ACLs in objectAcl.ts.",
    ]:
        call(server, "memory_write", text=text, type="decision")

    from agent_memory.tokens import count_tokens

    for budget in (30, 60, 120, 300):
        out = call(server, "memory_boot", task="continue the booking work", budget_tokens=budget)
        # All rendered labels, IDs and source references count toward the cap.
        assert count_tokens(out) <= budget, f"budget {budget} exceeded: {out!r}"



def test_boot_on_an_empty_store(server):
    assert "Empty store" in call(server, "memory_boot", task="anything")


def test_update_and_forget(server):
    call(server, "memory_write", text="The API listens on port 5002.", type="project")
    listed = call(server, "memory_list")
    entry_id = listed.split()[1]

    assert "Updated" in call(server, "memory_update", id=entry_id, text="The API listens on port 8080.", expected_revision=1)
    assert "8080" in call(server, "memory_recall", query="which port does the API listen on")

    assert "Forgot" in call(server, "memory_forget", id=entry_id, expected_revision=2)
    assert "No memory with id" in call(server, "memory_forget", id=entry_id, expected_revision=2)
    assert call(server, "memory_list") == "No memories stored."


def test_update_and_forget_report_missing_ids(server):
    assert "No memory with id" in call(server, "memory_update", id="mem_9999", text="x", expected_revision=1)
    assert "No memory with id" in call(server, "memory_forget", id="mem_9999", expected_revision=1)


@pytest.mark.parametrize("tool", ["memory_update", "memory_forget"])
def test_stale_id_cannot_modify_replacement_through_mcp(server, tool):
    saved = call(server, "memory_write", text="The retired staging server uses port 5002.")
    old_id = saved.split()[1]
    assert "Forgot" in call(server, "memory_forget", id=old_id, expected_revision=1)
    saved = call(server, "memory_write", text="Customer invoices are archived monthly.")
    replacement_id = saved.split()[1]
    before = call(server, "memory_list")
    args = {"id": old_id, "expected_revision": 1}
    if tool == "memory_update":
        args["text"] = "Incorrect stale update."

    assert call(server, tool, **args) == f"No memory with id {old_id}."
    assert replacement_id != old_id
    assert call(server, "memory_list") == before
    assert "Customer invoices" in call(server, "memory_recall", query="invoices archived")


def test_handoff_is_picked_up_by_a_second_agent_on_the_same_store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_EMBEDDER", "hashing")
    path = tmp_path / "shared.json"
    claude = build_server(store_path=path, agent="claude-code")
    call(claude, "memory_handoff", done="Fixed double emails.", next_steps="Add rate limiting.")

    codex = build_server(store_path=path, agent="codex")
    out = call(codex, "memory_boot", task="continue the rate limiting work")
    assert "rate limiting" in out.lower()
    assert "claude-code" in out  # provenance survives the handoff


def test_server_ships_usage_instructions(server):
    """Clients surface these to the model, so agents without hooks still learn
    the workflow. This is the vendor-neutral half of not relying on the model
    to remember on its own."""
    instructions = getattr(server, "instructions", None)
    if instructions is None:  # older mcp releases have no such field
        pytest.skip("installed mcp version does not carry server instructions")
    for tool in ("memory_boot", "memory_write", "memory_handoff"):
        assert tool in instructions


def test_stats_reports_the_store(server):
    call(server, "memory_write", text="Bookings are stored in UTC.", type="decision")
    out = call(server, "memory_stats")
    assert "1 memories" in out and "decision=1" in out


def test_literal_tokenizer_marker_survives_mcp_roundtrip(server):
    text = "The parser must preserve the literal <|endoftext|> marker."
    assert "Saved" in call(server, "memory_write", text=text)
    assert "1 memories" in call(server, "memory_stats")
    assert text in call(server, "memory_recall", query="parser literal marker")
    assert text in call(server, "memory_boot", task="parser literal marker")


@pytest.mark.parametrize("tool", ["memory_update", "memory_forget", "memory_supersede"])
def test_correction_tools_require_revision_and_reject_stale_read(server, tool):
    saved = call(server, "memory_write", text="Retry three times.")
    entry_id = saved.split()[1]
    args = {"id": entry_id}
    if tool != "memory_forget":
        args["text"] = "Retry seven times."
    schema = next(t.model_dump(by_alias=True)["inputSchema"] for t in asyncio.run(server.list_tools()) if t.name == tool)
    assert "expected_revision" in schema["required"]
    with pytest.raises(Exception, match="expected_revision"):
        call(server, tool, **args)
    call(server, "memory_update", id=entry_id, text="Retry five times.", expected_revision=1)
    before = call(server, "memory_get", id=entry_id)
    with pytest.raises(Exception, match="revision"):
        call(server, tool, **args, expected_revision=1)
    assert call(server, "memory_get", id=entry_id) == before


def test_supersession_exposes_history_but_recalls_only_current_decision(server):
    old = call(server, "memory_write", text="Retry three times.", source={"path": "policy.py"}).split()[1]
    new = call(server, "memory_supersede", id=old, text="Retry five times.", expected_revision=1).split()[1]
    retired = json.loads(call(server, "memory_get", id=old))
    assert retired["status"] == "superseded" and retired["superseded_by"] == new
    assert retired["history"][0]["source"] == {"path": "policy.py"}
    out = call(server, "memory_recall", query="Retry", budget_tokens=300)
    assert "five" in out and "three" not in out
