# Chat Loop

A chat loop is an interactive conversation cycle where an LLM agent responds to a user's messages iteratively until the user signals the end of the conversation. Unlike the [validator loop](./validator_loop.md), which is designed for quality refinement, a chat loop requires **full control over how the next prompt is built** — the standard "please correct your response" template is replaced by a conversation history.

```
Initial prompt
      │
      v
  LLM Agent  ──── calls tool ───► print_response()
      │
      │  AgentOutput
      v
 get_and_check_input()   ◄── blocks on input()
      │
      ├── is_valid=True  ("/bye")  ──► exit cycle
      │
      └── is_valid=False
                │
                v
          augment_fn(history)  ──► LLM Agent [next iteration]
```

---

## Key design decisions

### 1. Merge input collection and exit check into one block

The executor tracks `last_producer_output` as **the last non-condition block**. In a 3-block chain `agent → get_user_input → check_done`, the user's typed text becomes the "last producer output" — not the agent's response — which corrupts the augmented prompt.

The fix is to merge both responsibilities into a single condition block:

```
agent → get_and_check_input   (condition_block)
 ↑               ↓
 └───────────────┘
```

Now `agent` is the sole producer and its response is correctly tracked.

### 2. Use `augment_fn` to control the next prompt

By default, `add_cycle()` wraps the feedback in a "--- Attempt N (rejected) ---" template aimed at quality refinement. For a chat loop, the next prompt should simply be the **conversation history**. Pass `augment_fn` to override this behaviour:

```python
graph.add_cycle(
    ...
    augment_fn=lambda orig, iteration, producer, feedback: feedback
)
```

The `feedback` value returned by `get_and_check_input` is the full formatted chat history, so the LLM receives clean conversational context on every new turn.

---

## Complete example

```python
import asyncio
import os

from agenticblocks import as_tool
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput, AgentOutput
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor

MODEL = os.getenv("AGENTICBLOCKS_MODEL", "ollama/mistral-nemo:latest")

# ── Shared conversation history ─────────────────────────────────────────────
chat_history: list[str] = []


# ── Tool: print the agent's response and record it in history ───────────────
@as_tool(name="print_researcher_response",
         description="Prints the researcher's response and adds it to the chat history.")
def print_researcher_response(response: str) -> None:
    print(f"Researcher: {response}")
    chat_history.append(f"Researcher: {response}")


# ── Condition block: collect user input and decide whether to exit ───────────
@as_tool(name="get_and_check_input")
def get_and_check_input() -> dict:
    """
    Blocks until the user types a message.
    Returns is_valid=True when the user types /bye, ending the cycle.
    The feedback field carries the full chat history for the next LLM turn.
    """
    print("Você: ", end="", flush=True)
    user_input = input()
    chat_history.append(f"User: {user_input}")
    is_done = user_input.strip().startswith("/bye")
    return {
        "is_valid": is_done,
        "feedback": "\n".join(chat_history),   # full history as next prompt
    }


# ── Optional: observable subclass to inspect the prompt each iteration ───────
class ObservableLLMAgent(LLMAgentBlock):
    async def run(self, input: AgentInput) -> AgentOutput:
        print(f"\n[{self.name}] prompt → {input.prompt[:120]}...")
        return await super().run(input)


async def main():
    agent = ObservableLLMAgent(
        name="research_agent",
        model=MODEL,
        system_prompt=(
            "Você é um assistente de pesquisa especializado em ajudar pesquisadores "
            "sobre os mais diversos tópicos. Você vai ser o primeiro a falar: "
            "dê as boas-vindas e pergunte sobre o que o usuário quer saber."
        ),
        tools=[print_researcher_response],
        max_tool_calls=1,
        litellm_kwargs={
            "temperature": 0.7,
            # Force the agent to always call print_researcher_response
            "tool_choice": {"type": "function",
                            "function": {"name": "print_researcher_response"}},
        },
    )

    graph = WorkflowGraph()
    graph.add_block(agent)
    graph.add_block(get_and_check_input)

    graph.add_cycle(
        name="chat_loop",
        sequence=["research_agent", "get_and_check_input"],
        condition_block="get_and_check_input",
        max_iterations=1000,
        # Replace the default "rejected/please correct" template with the
        # plain conversation history — the LLM sees clean chat context.
        augment_fn=lambda orig, iteration, producer, feedback: feedback,
    )

    executor = WorkflowExecutor(graph, verbose=False)

    ctx = await executor.run(initial_input={
        "prompt": (
            "Você está em uma conversa contínua. "
            "Responda à mensagem mais recente do histórico."
        )
    })

    cr = ctx.cycle_results.get("chat_loop")
    if cr:
        print(f"\n[Chat encerrado após {cr.iterations} turno(s).]")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## How `augment_fn` is called

On every iteration where `is_valid=False`, the executor calls:

```python
new_prompt = augment_fn(original_prompt, iteration, producer_text, feedback)
```

| Parameter | Type | Value in this example |
|---|---|---|
| `original_prompt` | `str` | The `initial_input["prompt"]` — never changes |
| `iteration` | `int` | Current iteration number (1-based) |
| `producer_text` | `str` | Text extracted from `research_agent`'s last output |
| `feedback` | `str` | Value of `feedback` returned by `get_and_check_input` |

In this example the lambda simply returns `feedback` (the full chat history), discarding the other parameters. You can use all four to build richer prompts — for instance, explicitly highlighting the user's last message:

```python
def build_chat_prompt(orig, iteration, producer, feedback):
    lines = feedback.splitlines()
    last_user = next((l for l in reversed(lines) if l.startswith("User:")), "")
    return (
        f"{feedback}\n\n"
        f"[Focus on answering: {last_user}]"
    )

graph.add_cycle(..., augment_fn=build_chat_prompt)
```

---

## Ending the conversation

Type `/bye` to exit. `get_and_check_input` returns `is_valid=True`, the executor stores the final `CycleResult` and the loop terminates cleanly — no exception is raised.

```python
cr = ctx.cycle_results.get("chat_loop")
print(f"Ended after {cr.iterations} turn(s). Validated: {cr.validated}")
```

---

## Related

- [Validator Loop](./validator_loop.md) — quality-refinement cycle using the default augmentation template
- `WorkflowGraph.add_cycle()` — full parameter reference in `validator_loop.md § add_cycle() reference`
