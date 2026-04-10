# Validator Loop

The **ValidatorLoopBlock** implements a self-correcting production cycle that pairs any generator block (the *producer*) with any validation function (the *validator*). It repeatedly refines the output until it passes validation or a maximum number of attempts is reached.

## How it works

```
Input (y)
   │
   ▼
Producer(y) ──► Output (x)
                   │
                   ▼
             Validator(x) ──► is_valid=True  ──► return x ✅
                   │
                   └── is_valid=False
                             │
                             ▼
                        y = y + x + feedback
                             │
                             └──► Producer(y) [next iteration]
```

1. The **producer** receives the current prompt `y` and generates output `x`.
2. The **validator** receives `x` and returns `{"is_valid": bool, "feedback": str}`.
3. If valid → the loop ends and returns `x`.
4. If not valid → the feedback is appended to the prompt so the producer knows what went wrong.
5. The cycle repeats until validation passes or `max_iterations` is reached.

## Installation

`ValidatorLoopBlock` lives in the `agenticblocks.blocks.flow` module:

```python
from agenticblocks.blocks.flow.validator_loop import ValidatorLoopBlock, ValidatorLoopInput
```

## Quickstart

The minimal setup requires only two things: an existing `LLMAgentBlock` as the producer and an `@as_tool` decorated function as the validator.

```python
import asyncio
import os

from agenticblocks import as_tool
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks.blocks.flow.validator_loop import ValidatorLoopBlock, ValidatorLoopInput

MODEL = os.getenv("AGENTICBLOCKS_MODEL", "gpt-4o-mini")

# 1. Validator — plain Python function, no extra class needed
@as_tool
def validate_email(content: str) -> dict:
    """Check that the email has at least 3 paragraphs and no informal language."""
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if len(paragraphs) < 3:
        return {
            "is_valid": False,
            "feedback": f"Only {len(paragraphs)} paragraph(s). Write at least 3: introduction, body, and closing.",
        }
    return {"is_valid": True, "feedback": ""}

# 2. Producer — standard LLMAgentBlock
writer = LLMAgentBlock(
    name="EmailWriter",
    model=MODEL,
    system_prompt="You are a professional corporate writer. Write formal, well-structured emails.",
    max_tool_calls=0,
)

# 3. Orchestrate the loop
loop = ValidatorLoopBlock(
    name="email_loop",
    producer=writer,
    validator=validate_email,
    max_iterations=3,
)

async def main():
    result = await loop.run(input=ValidatorLoopInput(
        prompt="Write an email to the team informing that Friday's meeting has been postponed to Monday at 10 AM, room 204."
    ))
    print(f"Validated: {result.validated} | Iterations: {result.iterations}")
    print(result.result)

asyncio.run(main())
```

## API Reference

### `ValidatorLoopBlock`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Unique block identifier. |
| `producer` | `Block` | required | Any block whose `input_schema` exposes a `prompt: str` field. |
| `validator` | `Block` | required | Block (or `@as_tool` function) whose `input_schema` exposes a `content: str` field. |
| `max_iterations` | `int` | `5` | Maximum number of produce→validate cycles before giving up. |

### `ValidatorLoopInput`

```python
class ValidatorLoopInput(BaseModel):
    prompt: str
```

### `ValidatorLoopOutput`

```python
class ValidatorLoopOutput(BaseModel):
    result: str       # final text output from the producer
    iterations: int   # how many cycles ran
    validated: bool   # True if the validator accepted the result
```

## Validator Return Formats

The `ValidatorLoopBlock` is flexible about what the validator returns. All three formats are supported:

### Option A — `@as_tool` returning a `dict` *(recommended)*

The simplest approach. Return a plain dictionary with `is_valid` and `feedback` keys:

```python
@as_tool
def my_validator(content: str) -> dict:
    ok = len(content) > 100
    return {
        "is_valid": ok,
        "feedback": "" if ok else "Response is too short, elaborate more.",
    }
```

### Option B — `@as_tool` returning a `ValidationResult` model

If you prefer typed outputs, import and return `ValidationResult` directly:

```python
from agenticblocks.blocks.flow.validator_loop import ValidationResult

@as_tool
def my_validator(content: str) -> ValidationResult:
    return ValidationResult(is_valid=True, feedback="")
```

### Option C — `LLMAgentBlock` returning JSON

You can use another LLM as the validator. The loop will extract the JSON from the model's text response automatically:

```python
llm_validator = LLMAgentBlock(
    name="validator_llm",
    model=MODEL,
    system_prompt=(
        'You are a strict code reviewer. '
        'Respond ONLY with valid JSON: {"is_valid": true/false, "feedback": "reason if rejected"}'
    ),
    max_tool_calls=0,
)

loop = ValidatorLoopBlock(
    name="code_loop",
    producer=code_writer,
    validator=llm_validator,
    max_iterations=4,
)
```

## Practical Examples

### Example 1: Enforcing output length

```python
@as_tool
def length_validator(content: str) -> dict:
    """Ensure the response is at least 200 words."""
    word_count = len(content.split())
    if word_count < 200:
        return {
            "is_valid": False,
            "feedback": f"Response has {word_count} words. It must be at least 200 words long.",
        }
    return {"is_valid": True, "feedback": ""}
```

### Example 2: JSON schema validation

```python
import json

@as_tool
def json_validator(content: str) -> dict:
    """Ensure the producer returns parseable JSON with required keys."""
    try:
        data = json.loads(content)
        required = {"title", "summary", "tags"}
        missing = required - data.keys()
        if missing:
            return {"is_valid": False, "feedback": f"Missing required keys: {missing}"}
        return {"is_valid": True, "feedback": ""}
    except json.JSONDecodeError as e:
        return {"is_valid": False, "feedback": f"Invalid JSON: {e}"}
```

### Example 3: LLM-as-Judge pipeline

Use a stronger model to evaluate the output of a smaller, faster one:

```python
judge = LLMAgentBlock(
    name="judge",
    model="gpt-4o",  # stronger model as evaluator
    system_prompt=(
        "You are a senior editor. Evaluate the provided text for clarity, grammar, and tone. "
        'Respond only with JSON: {"is_valid": true/false, "feedback": "specific improvements if rejected"}'
    ),
    max_tool_calls=0,
)

loop = ValidatorLoopBlock(
    name="draft_loop",
    producer=LLMAgentBlock(name="writer", model="gpt-4o-mini", ...),
    validator=judge,
    max_iterations=3,
)
```

## How the feedback prompt is built

When validation fails, the loop automatically constructs an augmented prompt for the next iteration:

```
{original prompt}

--- Attempt {N} (rejected) ---
Your previous response was:
{rejected output}

Validator feedback:
{feedback message}

Please correct your response taking the feedback above into account.
```

The producer receives the full context of the failure — what it generated and why it was rejected — so it can correct itself without any extra code on your side.

## When `validated=False` in the output

If the loop exhausts all `max_iterations` without the validator accepting the result, `ValidatorLoopBlock` **does not raise an exception**. Instead, it returns the last output with `validated=False`. You can decide what to do with it:

```python
result = await loop.run(input=ValidatorLoopInput(prompt="..."))

if not result.validated:
    print(f"Warning: output was not validated after {result.iterations} attempts.")
    # fallback logic, log, escalate, etc.
else:
    print(result.result)
```
