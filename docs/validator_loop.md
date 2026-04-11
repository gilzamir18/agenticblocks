# Validator Loop

The validator loop is a self-correcting execution cycle that pairs a **producer** block with a **validator** block. The producer generates output; the validator checks it. If validation fails, the validator's feedback is automatically appended to the next prompt so the producer can correct itself. The cycle repeats until the output passes or `max_iterations` is reached.

```
Input (y)
   │
   v
Producer(y) --> Output (x)
                   │
                   v
             Validator(x) --> is_valid=True  --> exit cycle, return x
                   │
                   └── is_valid=False
                             │
                             v
                        y = y + x + feedback
                             │
                             └--> Producer(y) [next iteration]
```

---

## Approach 1 — Native graph cycle *(recommended)*

Cycles are first-class citizens in `WorkflowGraph`. Declare the cycle with `add_cycle()` and the executor handles the loop transparently, including:

- Collecting initial inputs from upstream nodes
- Building the augmented prompt on each rejected iteration
- Storing the cycle output in `ExecutionContext` so downstream blocks can consume it normally

### Quickstart

```python
import asyncio, os
from agenticblocks import as_tool
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor

MODEL = os.getenv("AGENTICBLOCKS_MODEL", "gpt-4o-mini")

# 1. Validator — plain function, reusable in any graph
@as_tool
def validate_email(content: str) -> dict:
    """Checks for at least 3 paragraphs and no informal language."""
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if len(paragraphs) < 3:
        return {
            "is_valid": False,
            "feedback": f"Only {len(paragraphs)} paragraph(s). Write at least 3.",
        }
    return {"is_valid": True, "feedback": ""}

# 2. Producer — standard LLMAgentBlock
writer = LLMAgentBlock(
    name="writer",
    model=MODEL,
    system_prompt="You are a professional corporate writer.",
    max_tool_calls=0,
)

# 3. Build the graph with a declared cycle
graph = WorkflowGraph()
graph.add_block(writer)
graph.add_block(validate_email)

graph.add_cycle(
    name="refine_email",
    sequence=["writer", "validate_email"],  # linear chain shorthand
    condition_block="validate_email",        # controls is_valid / feedback
    max_iterations=3,
)

# 4. Connect the cycle to downstream nodes as usual
graph.add_block(my_publisher)
graph.connect("refine_email", "my_publisher")

async def main():
    executor = WorkflowExecutor(graph)
    ctx = await executor.run(initial_input={
        "prompt": "Write a formal email postponing Friday's meeting to Monday at 10 AM."
    })

    cycle_result = ctx.cycle_results["refine_email"]
    print(f"Validated: {cycle_result.validated} | Iterations: {cycle_result.iterations}")
    print(ctx.get_output("refine_email"))  # the producer's last (validated) output

asyncio.run(main())
```

### `graph.add_cycle()` reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Unique cycle identifier; acts as a virtual node in the graph. |
| `condition_block` | `str` | required | Name of the block whose output controls continuation. Must return `is_valid` and optionally `feedback`. |
| `edges` | `list[tuple[str, str]] \| None` | `None` | Explicit directed edges `(from, to)` inside the cycle. Use for non-linear topologies (fanout, merge). Mutually exclusive with `sequence`. |
| `sequence` | `list[str] \| None` | `None` | Shorthand for linear chains. `sequence=["A","B","C"]` expands to `edges=[("A","B"),("B","C")]`. Mutually exclusive with `edges`. |
| `max_iterations` | `int` | `5` | Maximum iterations before giving up. Result is returned with `validated=False`. |
| `prompt_field` | `str` | `"prompt"` | Field on the entry block's input schema that receives the augmented feedback prompt. |

> **Note:** Exactly one of `edges` or `sequence` must be provided. Supplying both raises `ValueError`.

The **entry block** (first to receive input) is auto-detected as the cycle member with no incoming internal edges.

### Reading the cycle result

```python
ctx = await executor.run(...)

# Option A — via ExecutionContext helper (same API as any node)
output = ctx.get_output("refine_email")   # returns the producer's last output

# Option B — full cycle metadata
cr = ctx.cycle_results["refine_email"]
print(cr.validated)    # bool
print(cr.iterations)   # int
print(cr.output)       # BaseModel — producer's last output
```

### Multi-block cycles

Cycles are not limited to a producer + validator pair. Any linear chain works.

#### Using `sequence` (recommended for linear chains)

Pass the ordered list of block names; edges are derived automatically via consecutive pairs:

```python
graph.add_block(writer)
graph.add_block(formatter)
graph.add_block(validator)

graph.add_cycle(
    name="full_pipeline",
    sequence=["writer", "formatter", "validator"],  # same as edges below
    condition_block="validator",
    max_iterations=4,
)
```

#### Using `edges` (required for non-linear topologies)

Use `edges` when the internal data flow is not a simple chain — for example, a fanout or a merge:

```python
# writer branches to two reviewers, both feed a merger
graph.add_cycle(
    name="review_pipeline",
    edges=[
        ("writer", "reviewer_a"),
        ("writer", "reviewer_b"),
        ("reviewer_a", "merger"),
        ("reviewer_b", "merger"),
    ],
    condition_block="merger",
    max_iterations=3,
)
```

The executor runs `writer → formatter → validator` (or your custom topology) on each iteration. The cycle output is the last non-condition block's output.

---

## Approach 2 — `ValidatorLoopBlock` *(standalone convenience)*

If you do not need a full `WorkflowGraph` — for example in a simple script or when composing cycles within a custom block — `ValidatorLoopBlock` provides the same pattern as a self-contained block:

```python
from agenticblocks.blocks.flow.validator_loop import ValidatorLoopBlock, ValidatorLoopInput

loop = ValidatorLoopBlock(
    name="email_loop",
    producer=writer,
    validator=validate_email,
    max_iterations=3,
)

result = await loop.run(input=ValidatorLoopInput(
    prompt="Write a formal email..."
))
print(result.validated, result.iterations, result.result)
```

`ValidatorLoopBlock` is itself a `Block` and can be added to a `WorkflowGraph` as a single node, but its internal blocks (`producer`, `validator`) will not be individually visible or addressable in the graph. Use Approach 1 when that matters.

---

## Validator return formats

Both approaches accept the same three validator output formats:

### A — `@as_tool` returning `dict` *(recommended)*

```python
@as_tool
def my_validator(content: str) -> dict:
    ok = len(content) > 100
    return {
        "is_valid": ok,
        "feedback": "" if ok else "Too short — elaborate more.",
    }
```

### B — `@as_tool` returning a typed model

```python
from agenticblocks.blocks.flow.validator_loop import ValidationResult

@as_tool
def my_validator(content: str) -> ValidationResult:
    return ValidationResult(is_valid=True, feedback="")
```

### C — `LLMAgentBlock` returning JSON

The executor parses the JSON out of the model's text response automatically:

```python
judge = LLMAgentBlock(
    name="judge",
    model="gpt-4o",
    system_prompt=(
        "You are a strict code reviewer. "
        'Respond ONLY with JSON: {"is_valid": true/false, "feedback": "reason"}'
    ),
    max_tool_calls=0,
)
```

> **Tip:** For the native graph approach, the executor uses `_get_text_field()` to automatically detect whether the condition block's input schema expects `content`, `prompt`, `text`, or another string field — no manual mapping needed.

---

## Practical examples

### Enforce output length

```python
@as_tool
def length_validator(content: str) -> dict:
    word_count = len(content.split())
    if word_count < 200:
        return {"is_valid": False, "feedback": f"Only {word_count} words. Need at least 200."}
    return {"is_valid": True, "feedback": ""}
```

### JSON schema validation

```python
import json

@as_tool
def json_validator(content: str) -> dict:
    try:
        data = json.loads(content)
        missing = {"title", "summary", "tags"} - data.keys()
        if missing:
            return {"is_valid": False, "feedback": f"Missing keys: {missing}"}
        return {"is_valid": True, "feedback": ""}
    except json.JSONDecodeError as e:
        return {"is_valid": False, "feedback": f"Invalid JSON: {e}"}
```

### LLM-as-Judge pipeline

```python
judge = LLMAgentBlock(
    name="judge",
    model="gpt-4o",
    system_prompt=(
        "You are a senior editor. Evaluate clarity, grammar, and tone. "
        'Respond ONLY with JSON: {"is_valid": true/false, "feedback": "..."}'
    ),
    max_tool_calls=0,
)

graph.add_cycle(
    name="draft_loop",
    edges=[("writer", "judge")],
    condition_block="judge",
    max_iterations=3,
)
```

---

## How the feedback prompt is built

When validation fails, the executor automatically constructs an augmented prompt for the next iteration:

```
{original prompt}

--- Attempt {N} (rejected) ---
Your previous response was:
{rejected output}

Validator feedback:
{feedback message}

Please correct your response taking the feedback above into account.
```

The producer receives the full context of what it generated and why it was rejected — no extra orchestration code needed.

---

## When `validated=False`

If all iterations are exhausted without validation passing, no exception is raised. The last producer output is returned with `validated=False`:

```python
# Native graph approach
cr = ctx.cycle_results["refine_email"]
if not cr.validated:
    print(f"Warning: not validated after {cr.iterations} attempts.")

# ValidatorLoopBlock approach
result = await loop.run(input=ValidatorLoopInput(prompt="..."))
if not result.validated:
    print(f"Warning: not validated after {result.iterations} attempts.")
```
