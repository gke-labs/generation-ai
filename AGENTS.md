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
- **Testing**: prefer an end-to-end test over a unit test per file. In Go, correctness comes mostly from compiling, vetting and linting plus the e2e suites under `tests/e2e`; do not write unit tests that restate the main code. In Python, which has no strict compiler, a unit test may be needed for behaviour no e2e reaches; keep it to that behaviour. Test code (`*_test.py`, `*_test.go`) must be correct and readable but is held to a lower standard than production code: no input validation beyond what catches real mistakes, simple structure is fine, and test code needs no tests of its own.
- **Python types**: annotate function parameters and returns, dataclass fields, and every document that crosses a process or language boundary (the kernel table, the plan file, the run record) as a `TypedDict` or a dataclass with its schema version. New modules start with `# pyright: strict`; an existing function is annotated when a change touches it; do not open annotation-only PRs over old code. Tensors are `torch.Tensor`; use `typing.Protocol` for the few duck-typed seams rather than `Any`. The presubmit runs pyright (strict on files that opt in, basic elsewhere), so annotations are checked, not decorative.
- **Modern Go**: Use `any` instead of `interface{}` in Go code.

## Writing

Code comments, docstrings, commit messages, PR descriptions and docs should read clearly to someone new to the code. Follow Google's [Technical Writing One](https://developers.google.com/tech-writing/one) and, for details, the [Google developer documentation style guide](https://developers.google.com/style). For doc comments, see [Go doc comments](https://go.dev/doc/comment) and the docstring section of the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

The rules that matter most here:
- **Use active voice.** Say who does what: "the router claims the executor before it asks it to serve", not "the executor is claimed before it is asked to serve".
- **Write full sentences, one idea each.** Don't chain clauses with colons, semicolons and parenthetical asides.
- **Define a term before you use it,** and use the project's existing words. Don't coin shorthand.
- **Say what the code does and why,** not how it got that way.
- **Check all the prose in your change** before you send it. Start from the diff, and check every file that holds prose: docs (such as `design.md` and `README.md`), code comments, docstrings, proto and script comments, error and log messages, flag help, the commit message and the PR description.

For example, rewrite this:

> The inputs of this engine's range that are numbers, not tensors (a prefill's length one segment hands another): from the network they come as tensors, and are turned back.

as this:

> The names of this range's inputs that are plain numbers rather than tensors, such as a prefill's length. The wire format carries only tensors, so the range server turns these back into numbers when they arrive.

## Development Workflow

- Adhere to the project's coding style and structure.
- Follow the PR hygiene mentioned in the project's instructions:
    - Solve only the specific issue.
    - One idea per PR.
    - Well-structured commits (see Commit Messages and PR Descriptions below).
    - Reference issues in the commit body.
    - **Design changes in vxpu**: update `vxpu/docs/design.md` in the same PR. GitHub issues hold the backlog; the design itself lives in the repo.
    - **Binary Files**: Do not check in binary files or large data files into the repository. Agents MUST confirm they have not checked in any binaries before sending a PR!

### Commit Messages and PR Descriptions

**The code and its commit history must stand on their own.** PR descriptions do not become part of the repository; commit messages do. Anything a future reader needs to understand a change belongs in the commit messages, and the PR description should simply repeat them.

- **Subject line:** `area: imperative summary`, about 72 characters at most, where `area` is the package or component touched (for example `vxpu/engine`, `vxpu/plan`, `vxpu/proto`, `vxpu/router`, `vxpu/dev/vm`, `modelstore/registry`, `docs`).
- **Body: explain why, not just what.** Include whatever the diff alone doesn't make obvious:
    - **Bug fixes:** the symptom, the root cause, why the change fixes it, and the test that reproduces it.
    - **Design changes:** the approach taken, and the alternatives considered and why they were rejected.
    - **Performance changes:** the numbers, how they were measured, and in what configuration.
    - **Review answers:** if a reviewer had to ask, the answer probably belongs in the message.
- **One logical change per commit**, each building and passing tests where practical (for example: proto changes, then implementation, then tests or CI).
- **Keep messages true to the final code.** When review changes the approach, rewrite the affected commit messages (amend, or `fixup!` commits squashed with `git rebase --autosquash`) so they describe what the code does now, not the history of attempts. If you add follow-up commits instead, give each one a body explaining what changed and why; a bare `fixup!` subject is not enough. Never leave a message describing code that was later removed.
- **Reference issues** in the commit body with `Fixes #N` on the commit that completes an issue. Do not write `Part of #N`: a PR that links an issue it does not close makes codebot-robot treat that issue as in progress and leaves nothing to close; split the issue instead so each PR closes one.
- **PR description = the commit messages**, in order, for multi-commit PRs. If a reviewer would need information that isn't in the commits, add it to the commits rather than only to the PR description.

### Commands

The project uses the `ap` tool for various tasks. Since `ap` is a custom tool, it should be run using `go run`:

- `go run github.com/gke-labs/gke-labs-infra/ap@latest generate //...`: Regenerate any code and format.
- `go run github.com/gke-labs/gke-labs-infra/ap@latest test`: Run unit tests.
- `go run github.com/gke-labs/gke-labs-infra/ap@latest e2e`: Run e2e tests.
- `go run github.com/gke-labs/gke-labs-infra/ap@latest lint`: For deeper static analysis.

**Reminder**: Coding agents MUST run at least `ap generate //...` and `ap lint` before sending PRs, and preferably `ap e2e` as well!
**CRITICAL**: Running `ap generate //...` is absolutely mandatory, otherwise the CI presubmit `ap-verify-generate` will fail.
