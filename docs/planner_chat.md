# PlannerChatBlock

`PlannerChatBlock` is a high-level, reusable block that encapsulates the **plan-then-execute** conversational pattern. It combines a planner LLM with a [`PlanExecutorBlock`](./chatexample.md) to handle one full chat turn — planning what to do and then executing it — while automatically managing the conversation history.

```
get_user_input()
      │  user_message
      v
PlannerChatBlock
  │
  ├─► planner LLM  ──► JSON plan
  │
  └─► PlanExecutorBlock
            │
            ├─► tool_1()
            ├─► tool_2()
            └─► executor LLM ──► print_agent_response()
                                         │
                                         v
                                   response string
      │
      v
check_done()  ──► is_valid=True  ──► exit cycle
                  is_valid=False ──► next iteration
```

---

## Motivation

The recurring "planner + executor" pattern inside a chat loop previously required a hand-crafted closure:

```python
# Old pattern — boilerplate closure wired by hand
def make_turn_block(planner_agent, plan_executor):
    @as_tool(name="plan_and_execute_turn", description="...")
    async def plan_and_execute_turn(user_message: str) -> str:
        chat_history.append(f"User: {user_message}")
        history_str = "\n".join(chat_history[-8:])
        # ... call planner, parse JSON, call executor ...
        return output.response
    return plan_and_execute_turn

turn_block = make_turn_block(planner_agent, plan_executor)
```

`PlannerChatBlock` replaces the closure with a first-class, inspectable, configurable `Block`:

```python
# New pattern — single instantiation, no closure
turn_block = PlannerChatBlock(
    planner=planner_agent,
    executor=plan_executor,
    history=chat_history,
)
```

---

## Quick start

```python
import asyncio
from agenticblocks import as_tool, PlannerChatBlock
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks.blocks.patterns.plan_executor import PlanExecutorBlock
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor

# ── Shared conversation history ────────────────────────────────────────────
chat_history: list[str] = []

# ── Output tool (called by the executor agent) ─────────────────────────────
@as_tool(name="print_agent_response", description="Delivers the final reply.")
def print_agent_response(response: str) -> str:
    print(f"Agent: {response}")
    chat_history.append(f"Agent: {response}")
    return "ok"

# ── Chat flow controls ─────────────────────────────────────────────────────
@as_tool(name="get_user_input")
def get_user_input() -> dict:
    print("You: ", end="", flush=True)
    return {"user_message": input().strip()}

@as_tool(name="check_done")
def check_done(last_message: str = "") -> dict:
    for line in reversed(chat_history):
        if line.startswith("User:"):
            if line[len("User:"):].strip().lower() in {"exit", "quit", "/bye"}:
                return {"is_valid": True, "feedback": "done"}
            break
    return {"is_valid": False, "feedback": "continue"}

def build_chat_prompt(orig, iteration, producer, feedback):
    return feedback or ""


async def main():
    # ── Planner: outputs JSON plan only, no tool calls ─────────────────────
    planner = LLMAgentBlock(
        name="planner",
        model="ollama/mistral-nemo:latest",
        system_prompt=(
            "You are a planning agent. Output ONLY a JSON plan.\n"
            'Format: {"thought": "...", "steps": [{"action": "...", "args": {...}}]}\n'
            'Always end with a "reply" step.'
        ),
        tools=[],
        max_iterations=1,
        model_kargs={"temperature": 0.0},
    )

    # ── Executor: receives briefing + observations, calls print_agent_response
    executor_agent = LLMAgentBlock(
        name="executor_agent",
        model="ollama/mistral-nemo:latest",
        system_prompt="Call 'print_agent_response' exactly once with your final answer.",
        tools=[print_agent_response],
        max_tool_calls=1,
        model_kargs={"temperature": 0.3},
    )

    plan_executor = PlanExecutorBlock(
        executor_agent=executor_agent,
        tools=[],           # add domain tools here
        max_reply_retries=2,
    )

    # ── PlannerChatBlock: replaces make_turn_block ─────────────────────────
    turn_block = PlannerChatBlock(
        planner=planner,
        executor=plan_executor,
        history=chat_history,       # block appends "User:" and "Agent:" lines
    )

    graph = WorkflowGraph()
    graph.add_block(get_user_input)
    graph.add_block(turn_block)
    graph.add_block(check_done)

    graph.add_cycle(
        name="chat_loop",
        sequence=["get_user_input", "plan_and_execute_turn", "check_done"],
        condition_block="check_done",
        max_iterations=1000,
        augment_fn=build_chat_prompt,
    )

    executor = WorkflowExecutor(graph, verbose=False)
    await executor.run(initial_input={"prompt": "start"})


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Constructor reference

```python
PlannerChatBlock(
    name="plan_and_execute_turn",   # default — must match the cycle sequence entry
    planner=...,                    # LLMAgentBlock that returns a JSON plan
    executor=...,                   # PlanExecutorBlock that runs the plan
    history=chat_history,           # shared list; None = block owns private list
    history_window=8,               # how many recent lines to send to the planner
    user_prefix="User",             # label prepended to user lines
    agent_prefix="Agent",           # label prepended to agent lines
    planner_prompt_template=...,    # format string with {history} and {user_message}
    fallback_plan={...},            # JSON plan used when planner returns invalid JSON
    on_plan_ready=my_callback,      # optional fn(plan: dict) for logging / debugging
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"plan_and_execute_turn"` | Block name — must match the `sequence` entry in `add_cycle()`. |
| `planner` | `LLMAgentBlock` | required | LLM that produces the JSON plan. |
| `executor` | `PlanExecutorBlock` | required | Runs the plan steps and synthesises the final reply. |
| `history` | `list[str] \| None` | `None` | Shared mutable list. When `None`, a private list is created internally. |
| `history_window` | `int` | `8` | Number of history lines passed to the planner prompt. |
| `user_prefix` | `str` | `"User"` | Prefix for user messages recorded in `history`. |
| `agent_prefix` | `str` | `"Agent"` | Prefix for agent messages recorded in `history`. |
| `planner_prompt_template` | `str` | *(see source)* | Format string for the planner prompt. Available keys: `{history}`, `{user_message}`. |
| `fallback_plan` | `dict \| None` | `None` | Plan used when `extract_json_plan` fails. Defaults to a single `reply` step with an apology. |
| `on_plan_ready` | `Callable[[dict], Any] \| None` | `None` | Called with the parsed plan just before execution. Useful for logging. |

---

## How a turn is processed

On each cycle iteration, the block performs four steps:

### Step 1 — Normalise user input

`PlannerChatInput.user_message` may arrive as a plain string or as a dict
(when the WorkflowGraph passes the raw output of `get_user_input`). The block
handles both transparently:

```python
# Both are accepted:
PlannerChatInput(user_message="What's on the menu?")
PlannerChatInput(user_message={"user_message": "What's on the menu?"})
```

### Step 2 — Record and build the planner prompt

The user message is appended to `history` and the last `history_window` lines
are formatted into the planner prompt via `planner_prompt_template`:

```
RECENT HISTORY:
User: Hello!
Agent: Hi! How can I help?
User: What's on the menu?

USER MESSAGE: What's on the menu?

Produce ONLY the JSON plan. No text before or after.
```

### Step 3 — Plan

The planner LLM is called and its response is parsed with `extract_json_plan`.
If no valid JSON is found, `fallback_plan` is used (or the built-in apology
plan if `fallback_plan` is `None`).

### Step 4 — Execute

`PlanExecutorBlock.run()` receives the parsed plan and the history string,
runs each step (calling domain tools as needed), and delegates the final
`reply` step to the executor agent, which calls the output tool
(e.g. `print_agent_response`).

The agent response is appended to `history` and returned as
`PlannerChatOutput.response`.

---

## Customising the planner prompt

Override `planner_prompt_template` to adapt the prompt to your domain or
language:

```python
turn_block = PlannerChatBlock(
    ...
    planner_prompt_template=(
        "HISTÓRICO RECENTE:\n{history}\n\n"
        "MENSAGEM DO USUÁRIO: {user_message}\n\n"
        "Produza APENAS o JSON do plano. Sem texto antes ou depois."
    ),
)
```

The only required placeholders are `{history}` and `{user_message}`.

---

## Observing the generated plan

Pass `on_plan_ready` to inspect or log the plan without modifying the block's
internals:

```python
import json

def log_plan(plan: dict) -> None:
    print(f"[Plan]\n{json.dumps(plan, indent=2, ensure_ascii=False)}")

turn_block = PlannerChatBlock(
    ...
    on_plan_ready=log_plan,
)
```

---

## Using a custom fallback plan

When the planner LLM produces invalid JSON, the block uses `fallback_plan`.
Override it to match your domain's toolset:

```python
turn_block = PlannerChatBlock(
    ...
    fallback_plan={
        "thought": "fallback",
        "steps": [
            {"action": "get_cardapio", "args": {}},
            {"action": "reply", "args": {
                "message": "Não entendi o pedido. Posso mostrar o cardápio?"
            }},
        ],
    },
)
```

---

## The old closure pattern

`make_turn_block` still works and is not deprecated. `PlannerChatBlock` is the
recommended approach for new code because it is:

- **Inspectable** — a Pydantic model with declared fields, not a closure.
- **Configurable** — all knobs are constructor parameters.
- **Reusable** — can be instantiated multiple times with different settings.
- **Testable** — `await block.run(PlannerChatInput(user_message="..."))` works
  in isolation without a WorkflowGraph.

```python
# Both produce a block named "plan_and_execute_turn" — they are interchangeable
# in the graph sequence.

# Old way
turn_block = make_turn_block(planner_agent, plan_executor)

# New way
turn_block = PlannerChatBlock(planner=planner_agent, executor=plan_executor, history=chat_history)
```

---

## Complete real-world example

See [`examples/08_planner_v2.py`](../examples/08_planner_v2.py) for a full
working chatbot (TasteFast snack bar) that uses `PlannerChatBlock` alongside
domain tools, a validator, and a custom planner prompt in Brazilian Portuguese.

---

## Related

- [Chat Loop](./chatexample.md) — simpler single-agent chat without planning
- [Validator Loop](./validator_loop.md) — quality-refinement cycle
- [`PlanExecutorBlock`](./chatexample.md) — executes JSON plans step by step
