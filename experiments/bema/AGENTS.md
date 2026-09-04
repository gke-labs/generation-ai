# Bema - Agentic Session Server

Bema's core role is session storage.

## Scaling Goals

One of the core goals is scaling Bema to a **billion sessions** (not all of which will be active at once!).

## Pluggable Storage

To support this scale, Bema uses a pluggable `SessionStore` interface. This allows for different storage backends:

- **FileSessionStore**: Uses the local filesystem (default), suitable for development and small-scale use.
- **SQL / NoSQL Backends**: Future implementations will support distributed databases to achieve the billion-session goal.

## Key Principles

- **Efficiency**: Don't load all sessions into memory. Use the `SessionStore` for efficient retrieval and updates.
- **Asynchronicity**: Long-running LLM generation and tool execution are handled asynchronously, updating the session and notifying watchers.
- **Persistence**: Every session update is persisted to the store.
