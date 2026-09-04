# AGENTS.md

This file provides context and instructions for LLM coding agents working on the `generation-ai` project.

## Project Vision

`generation-ai` provides end-to-end solutions demonstrating application of AI to real-world business tasks, including ongoing fine-tuning / reinforcement learning and model serving and evaluation.

## Key Principles for Agents

- **End-to-End Solutions**: Focus on complete, functional examples that solve real business problems.
- **Modern AI Practices**: Employ best practices for fine-tuning, serving, and evaluating AI models.
- **GKE Integration**: Solutions should be designed to run effectively on Google Kubernetes Engine (GKE).
- **Clarity and Documentation**: Code should be well-documented and easy to follow, serving as a reference for users.
- **Structured Logging**: Use `log := klog.FromContext(ctx)` for structured logging to ensure operational visibility. Always pass context down and use structured fields (e.g., `log.Info("message", "key", value)`) instead of formatted strings.
- **Testability**: Ensure that implementations are well-tested, including end-to-end tests where appropriate.
- **Modern Go**: Use `any` instead of `interface{}` in Go code.

## Development Workflow

- Adhere to the project's coding style and structure.
- Follow the PR hygiene mentioned in the project's instructions:
    - Solve only the specific issue.
    - One idea per PR.
    - Well-structured commits.
    - Reference issues in the commit body.
    - **Binary Files**: Do not check in binary files or large data files into the repository. Agents MUST confirm they have not checked in any binaries before sending a PR!

### Commands

The project uses the `ap` tool for various tasks. Since `ap` is a custom tool, it should be run using `go run`:

- `go run github.com/gke-labs/gke-labs-infra/ap@latest generate //...`: Regenerate any code and format.
- `go run github.com/gke-labs/gke-labs-infra/ap@latest test`: Run unit tests.
- `go run github.com/gke-labs/gke-labs-infra/ap@latest e2e`: Run e2e tests.
- `go run github.com/gke-labs/gke-labs-infra/ap@latest lint`: For deeper static analysis.

**Reminder**: Coding agents MUST run at least `ap generate //...` and `ap lint` before sending PRs, and preferably `ap e2e` as well!
**CRITICAL**: Running `ap generate //...` is absolutely mandatory, otherwise the CI presubmit `ap-verify-generate` will fail.
