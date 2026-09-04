# AgentFS - Agent-Centric Filesystem

AgentFS provides a simple ephemeral filesystem for agents, designed with scaling and snapshotting in mind.

## Scaling Goals

AgentFS aims to scale with thousands of agents, providing each with its own ephemeral workspace.

## Key Principles

- **Ephemeral First**: The filesystem is ephemeral by default, providing fast, local storage.
- **Snapshotted Persistence**: Persistence is achieved through periodic snapshots, rather than continuous synchronization.
- **Lightweight CSI Implementation**: The CSI driver is designed to be minimal and easy to deploy on GKE.

## Development Status

- Currently implementing the node daemon to serve ephemeral storage.
- Implemented node daemon snapshotting (push/pull) to the controller.
- Implemented AgentFS Controller to serve snapshots and blobs.
- Future work: Advanced volume management and multi-node coordination.
