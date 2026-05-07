# Token Usage Statistics

This document explains how `LLMAgentBlock` exposes per-step token statistics and how you can consume them in your workflows.

---

## Overview

Every time `LLMAgentBlock` calls the LLM it emits a **`TokenUsage`** record that contains:

| Field | Type | Description |
|---|---|---|
| `block_name` | `str` | Name of the block that produced the call |
| `step` | `int` | Iteration number within the block's own loop (1-based) |
| `prompt_tokens` | `int` | Tokens consumed by the input prompt |
| `completion_tokens` | `int` | Tokens produced in the model response |
| `total_tokens` | `int` | Sum of prompt + completion tokens |

The data comes directly from the `usage` object returned by the LiteLLM response — **no extra API calls are made**.

---

## Strategy 1 — `ExecutionContext.token_stats` (recommended)

When a block runs inside a `WorkflowExecutor`, every `TokenUsage` record is automatically appended to the shared `ExecutionContext.token_stats` list.

```python
import asyncio
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor

async def main():
    agent = LLMAgentBlock(
        name="analyst",
        model="gpt-4o-mini",
        max_iterations=3,
    )

    graph = WorkflowGraph()
    graph.add_block(agent)

    executor = WorkflowExecutor(graph)
    ctx = await executor.run(initial_input={"prompt": "Summarise the history of AI."})

    # ── Per-step records ───────────────────────────────────────────────────
    for record in ctx.token_stats:
        print(
            f"[{record.block_name}] step={record.step} "
            f"prompt={record.prompt_tokens} "
            f"completion={record.completion_tokens} "
            f"total={record.total_tokens}"
        )

    # ── Aggregated helpers ─────────────────────────────────────────────────
    print("Grand total tokens:", ctx.total_tokens())
    print("By block:", ctx.tokens_by_block())

asyncio.run(main())
```

### Helper methods on `ExecutionContext`

| Method | Returns | Description |
|---|---|---|
| `ctx.token_stats` | `list[TokenUsage]` | Ordered list of every per-step record |
| `ctx.total_tokens()` | `int` | Sum of all `total_tokens` across the entire run |
| `ctx.tokens_by_block()` | `dict[str, dict]` | Per-block aggregation with `prompt`, `completion`, and `total` keys |

---

## Strategy 2 — `on_token_usage` callback

Use this when you want to react to each LLM call in real time (logging, streaming a UI counter, writing to a database, etc.) without needing a `WorkflowExecutor`.

```python
import asyncio
from agenticblocks import TokenUsage
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput

def log_usage(usage: TokenUsage) -> None:
    print(
        f"[{usage.block_name}] step {usage.step} → "
        f"in={usage.prompt_tokens} out={usage.completion_tokens} "
        f"total={usage.total_tokens}"
    )

async def main():
    agent = LLMAgentBlock(
        name="writer",
        model="gpt-4o-mini",
        on_token_usage=log_usage,   # ← attach the callback
    )
    output = await agent.run(AgentInput(prompt="Write a haiku about data."))
    print(output.response)

asyncio.run(main())
```

The callback also accepts `async` functions:

```python
async def async_log(usage: TokenUsage) -> None:
    await my_database.insert(usage.__dict__)
```

---

## Strategy 3 — Both simultaneously

Both mechanisms coexist. When a block runs inside a `WorkflowExecutor` **and** has an `on_token_usage` callback, every call triggers both.

```python
agent = LLMAgentBlock(
    name="researcher",
    model="gpt-4o-mini",
    on_token_usage=log_usage,  # immediate real-time feedback
)

executor = WorkflowExecutor(graph)
ctx = await executor.run(...)

# ctx.token_stats is also fully populated
print("Total:", ctx.total_tokens())
```

---

## Where stats are captured

Stats are emitted at **every** LiteLLM call inside `LLMAgentBlock.run()`:

| Scenario | `step` value |
|---|---|
| Normal iteration of the reasoning loop | `iteration_count` at time of call |
| Synthesis call (`on_max_iterations="return_last"`) | `iteration_count` (same as last iteration) |
| Forced final call after `max_tool_calls` is reached | `iteration_count` at time of call |

> **Note:** Tool executions (sub-blocks called as tools) are **not** counted in the parent
> block's `step`. Each subordinate `LLMAgentBlock` emits its own `TokenUsage` records
> under its own `block_name`, allowing per-agent attribution.

---

## Multi-agent / cycle workflows

In workflows with cycles or multiple LLM blocks the records accumulate sequentially in `ctx.token_stats`. Use `tokens_by_block()` to separate their contributions:

```python
ctx = await executor.run(initial_input={"prompt": "..."})

for block_name, totals in ctx.tokens_by_block().items():
    print(
        f"{block_name}: "
        f"prompt={totals['prompt']} "
        f"completion={totals['completion']} "
        f"total={totals['total']}"
    )
```

Example output for a `writer → validate_email` cycle that ran 3 iterations:

```
writer:         prompt=4210  completion=823  total=5033
validate_email: prompt=0     completion=0    total=0    # pure-Python @as_tool, no LLM
```

---

## Impact summary

| Aspect | Impact |
|---|---|
| **Latency** | Zero — data is extracted from the response object already in memory |
| **Accuracy** | Exact values reported by the provider via LiteLLM's `response.usage` |
| **Streaming** | `response.usage` is **not** populated in streaming mode; use non-streaming calls if token stats are required |
| **Concurrency** | `ctx.token_stats` is protected by an `asyncio.Lock` — safe for parallel waves |
| **Standalone use** | The `on_token_usage` callback works even without a `WorkflowExecutor` |
| **Provider support** | Most providers (OpenAI, Anthropic, Gemini, Cohere) populate `usage`; local models via Ollama may not |
