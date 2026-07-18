"""Long-lived generator actor + demo client.

The actor is a thin Monarch shim over engine.Engine — swap this file
for a gRPC servicer and the engine doesn't change; that's the point of
keeping the engine transport-agnostic (Monarch is under evaluation, not
assumed).

Heavy imports (torch, transformers, the engine) happen inside the actor
so the client process stays thin: it ships prompts and receives text.

Demo: two concurrent sessions on one worker — session A gets a
follow-up question that only makes sense if A's KV cache carried the
first turn, and B's answers prove it never saw A's conversation.
"""

import argparse
import os

from monarch.actor import Actor, endpoint, this_host


class GeneratorActor(Actor):
    def __init__(self, manifest_path, device="cpu", compile_decode=False):
        from engine import Engine  # worker-side only

        self.engine = Engine(
            manifest_path, device=device, compile_decode=compile_decode
        )

    @endpoint
    def new_session(self) -> str:
        return self.engine.new_session()

    @endpoint
    def chat(self, session_id: str, text: str, max_new_tokens: int = 96) -> dict:
        return self.engine.chat(session_id, text, max_new_tokens)

    @endpoint
    def stats(self) -> dict:
        return self.engine.stats()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="HuggingFaceTB--SmolLM2-135M-Instruct.manifest.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    print(f"[client] pid={os.getpid()} spawning generator worker")
    procs = this_host().spawn_procs(per_host={"procs": 1})
    generator = procs.spawn(
        "generator", GeneratorActor, args.manifest, args.device, args.compile
    )

    a = generator.new_session.call_one().get()
    b = generator.new_session.call_one().get()
    print(f"[client] sessions: {a}, {b}")

    def turn(session, text):
        reply = generator.chat.call_one(session, text).get()
        print(f"[{session}] >>> {text}")
        print(f"[{session}] {reply['text']}")
        print(f"[{session}]     ({reply['generated']} tokens, "
              f"{reply['ms_per_token']} ms/token, prefill "
              f"{reply['new_prompt_tokens']} new tokens in "
              f"{reply['prefill_ms']} ms, session total "
              f"{reply['session_tokens']})\n")

    turn(a, "Why is the sky blue?")
    turn(b, "What is 2 + 2?")
    # Only answerable if session A's cache still holds turn one:
    turn(a, "Summarize your previous answer in one short sentence.")
    turn(b, "What number did I just ask you about?")

    print(f"[client] stats: {generator.stats.call_one().get()}")
    procs.stop().get()


if __name__ == "__main__":
    main()
