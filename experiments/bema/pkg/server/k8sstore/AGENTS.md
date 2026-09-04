# K8s Session Store
This package provides a Kubernetes-backed implementation of the `SessionStore` interface.

## Architecture
The store uses two Custom Resource Definitions (CRDs):
1. **ChatSession**: Represents the top-level session metadata and configuration.
2. **ChatSessionMessage**: Represents individual messages within a session.

### Dual CRD / Aggregated API Server Strategy
Scaling to a billion sessions is a core goal of Bema. While CRDs are excellent for small to medium-scale deployments and provide a standard Kubernetes experience (working with `kubectl`, standard RBAC, etc.), they might face performance challenges at extreme scales due to etcd limitations.
The strategy is:
1. **Initial Implementation with CRDs**: Provides an immediate, functional, and familiar storage backend.
2. **Aggregated API Server**: In the future, we will implement an aggregated API server that is **API compatible** with these CRDs. This server will back the data with a more scalable storage (e.g., a dedicated SQL or NoSQL database) while still allowing users to interact with sessions via `kubectl` and other standard tools.

## Key Considerations
- **Immutability**: `ChatSessionMessage` objects are designed to be mostly immutable, representing the progression of a conversation.
- **Filtering**: `ChatSessionMessage` objects include a label (`bema.labs.gke.io/session-id`) to allow efficient filtering when reconstructing a session.
- **Ordering**: Messages are ordered by timestamp to ensure they are reconstructed in the correct sequence.
- **Data Conversion**: The store handles conversion between Protobuf types (used by the Bema gRPC service) and Kubernetes-native types.
