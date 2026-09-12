# Security and trust boundaries

The MCP server is a local stdio process. It has no authentication, authorization, remote transport or tenant isolation. Only connect agents you trust to read and edit the selected store. File permissions and the host environment define access.

Memory is fallible data, including text written by an agent or imported from Markdown. Source references are assertions, not proof. Do not execute instructions embedded in recalled text. The engine does not sanitize away prompt injection or decide whether a claim is true.

Stores contain plaintext memories and revision history. Keep secrets and unnecessary personal data out; exclude `.agent_memory/` and exported backups from public repositories. A connected coding agent may send recalled content to its provider even though the store itself is local. `forget` deletes from the current store, not backups, client transcripts or provider logs.

OS locks coordinate cooperating v0.4 writers on a local filesystem. Network shares, malicious writers and mixed versions are outside that guarantee. Export before experimenting with migrations; stop writers before restoring a backup.

Report a security issue with a minimal reproduction and affected version through GitHub's private vulnerability reporting if enabled. If it is unavailable, open a public issue asking for a private contact without including exploit details, credentials or private memory content. There is no promised response-time SLA.
