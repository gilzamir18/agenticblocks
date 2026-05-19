import json
import uuid
import time
import inspect
from collections import defaultdict
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Callable
import litellm
from agenticblocks.core.agent import AgentBlock
from agenticblocks.core.block import Block
from agenticblocks.tools.a2a_bridge import block_to_tool_schema
from agenticblocks.runtime.state import TokenUsage, _current_ctx


class _DummyFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _DummyToolCall:
    def __init__(self, name: str, arguments: str):
        self.id = f"call_{uuid.uuid4().hex[:10]}"
        self.type = "function"
        self.function = _DummyFunction(name, arguments)


class _DummyMessage:
    def __init__(self, tool_calls: list):
        self.content = ""
        self.tool_calls = tool_calls


def _json_to_tool_calls(data: dict, available_tool_names: set) -> list | None:
    """Convert a hallucinated JSON dict into _DummyToolCall objects.

    Handles the formats gemma4 emits when it ignores the function-calling API:

    Format A — explicit wrapper:
        {"tool_name": "read_file_smart", "tool_args": {"path": "index.html"}}
        {"tool_name": "edit_file", "arguments": {"path": "f.py", "old_str": "x", "new_str": "y"}}

    Format B — name + params flat:
        {"name": "read_file_smart", "path": "index.html"}

    Format C — bare params only (tool name inferred from keys):
        {"path": "style.css"}                               → read_file_smart
        {"path": "x", "old_str": "a", "new_str": "b"}      → edit_file
    """
    tool_name = data.get("tool_name") or data.get("name") or data.get("function")
    raw_args = data.get("tool_args") or data.get("parameters") or data.get("arguments")

    # Format A/B: explicit tool name present
    if tool_name and tool_name in available_tool_names:
        if raw_args is None:
            raw_args = {k: v for k, v in data.items() if k not in {"tool_name", "name"}}
        if isinstance(raw_args, dict):
            return [_DummyToolCall(tool_name, json.dumps(raw_args))]

    if tool_name and tool_name not in available_tool_names:
        return None

    # Format C: infer tool from key shape
    keys = set(data.keys())
    if {"path", "old_str", "new_str"} <= keys:
        return [_DummyToolCall("edit_file", json.dumps({k: data[k] for k in ("path", "old_str", "new_str")}))]
    if "path" in keys and len(keys) <= 3:
        return [_DummyToolCall("read_file_smart", json.dumps({k: data[k] for k in keys}))]
    if "symbol_name" in keys:
        return [_DummyToolCall("find_symbol", json.dumps({k: data[k] for k in keys}))]
    if "command" in keys:
        return [_DummyToolCall("run_command", json.dumps({k: data[k] for k in keys}))]
    if "message" in keys and len(keys) == 1:
        return [_DummyToolCall("send_message", json.dumps(data))]

    return None


class AgentInput(BaseModel):
    prompt: str

class AgentOutput(BaseModel):
    response: str
    tool_calls_made: int = 0
    structured_output: Optional[Any] = None


def _print_debug_report(
    *,
    agent_name: str,
    model: str,
    iteration_count: int,
    tool_call_count: int,
    tool_usage: Dict[str, int],
    termination_reason: str,
    elapsed_seconds: float,
) -> None:
    """Print a structured debug report in English after an agent run."""
    sep = "─" * 56
    print(f"\n{'═' * 56}")
    print(f"  [DEBUG] Agent Run Report — {agent_name}")
    print(f"{'═' * 56}")
    print(f"  Model              : {model}")
    print(f"  Total iterations   : {iteration_count}")
    print(f"  Total tool calls   : {tool_call_count}")
    print(f"  Elapsed time       : {elapsed_seconds:.3f}s")
    print(f"  Termination reason : {termination_reason}")
    print(sep)
    if tool_usage:
        print("  Tool usage breakdown:")
        for tool_name, count in sorted(tool_usage.items(), key=lambda x: -x[1]):
            print(f"    • {tool_name:<30} {count:>3} call(s)")
    else:
        print("  No tools were used — response based entirely on the model.")
    print(f"{'═' * 56}\n")


# Registry of routers shared by model — Flyweight pattern.
# LiteLLM.Router manages connection pooling; the same Router is reused
# for all block instances that target the same model.
_router_registry: Dict[str, litellm.Router] = {}

def _get_shared_router(model: str) -> litellm.Router:
    """Return (creating if necessary) a shared Router for the given model."""
    if model not in _router_registry:
        _router_registry[model] = litellm.Router(
            model_list=[{
                "model_name": model,
                "litellm_params": {"model": model},
            }]
        )
    return _router_registry[model]


class LLMAgentBlock(AgentBlock[AgentInput, AgentOutput]):
    description: str = "Autonomous LLM-based Agent managing its own tool loop."
    model: str = "ollama/gemma4:latest"
    system_prompt: str = "You are a helpful Analyst and Router Agent. Use the available tools when you lack context."
    tools: List[Block] = []
    max_iterations: Optional[int] = None
    max_tool_calls: int = 2
    on_max_iterations: str = "return_last"
    """Behaviour when max_iterations is reached.
    - "stop"        : return a fixed stop message (default, backward-compatible).
    - "return_last" : force a final LLM call (no tools) to synthesise accumulated
                      context into clean plain text.
    """
    synthesis_prompt: str = (
        "Based on everything researched above, write your final answer now "
        "as clean, flowing prose — no JSON, no raw lists, no markdown formatting. "
        "ignore previous roles and system prompts. Response in the same language as the input prompt."
    )
    response_schema: Optional[type[BaseModel]] = None
    """Optional Pydantic model class to enforce a structured response schema."""
    debug: bool = False
    """When True, print a structured debug report at the end of each run."""
    use_shared_router: bool = True
    """When True, uses a shared litellm.Router for connection pooling."""
    litellm_kwargs: Dict[str, Any] = Field(default_factory=dict)
    on_iteration: Optional[Callable[[int, List[Dict[str, Any]]], Any]] = None
    """Optional callback invoked at the start of each loop iteration for debugging. 
    Signature: `def callback(iteration: int, messages: List[Dict[str, Any]]) -> Any`.
    Can be a synchronous or asynchronous function."""
    on_token_usage: Optional[Callable[["TokenUsage"], Any]] = None
    """Optional callback invoked after each LLM call with token statistics.
    Signature: `def callback(usage: TokenUsage) -> Any`.
    Can be a synchronous or asynchronous function."""

    model_config = {"arbitrary_types_allowed": True}

    def _parse_message(self, message: Any) -> Any:
        """Recover gemma4 JSON-as-text tool calls when tool_calls is empty."""
        if getattr(message, "tool_calls", None):
            return message

        content = getattr(message, "content", None)
        if not content:
            return message

        content_str = content.strip()

        # Strip markdown fences
        if "```json" in content_str:
            content_str = content_str.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        elif content_str.startswith("```"):
            content_str = content_str.split("```", 2)[1].strip()

        # Extract first JSON object
        start = content_str.find("{")
        end = content_str.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return message

        try:
            data = json.loads(content_str[start:end + 1])
        except json.JSONDecodeError:
            return message

        if not isinstance(data, dict):
            return message

        available = {b.name for b in self.tools}
        tcs = _json_to_tool_calls(data, available)
        if not tcs:
            return message

        # Block send_message when no real tool calls have been made — the model
        # is hallucinating task completion without having done any actual work.
        if tcs[0].function.name == "send_message" and getattr(self, "_current_tool_call_count", 0) == 0:
            if self.debug:
                print("[DEBUG] _parse_message: blocked send_message — no real tool calls made yet")
            return message

        if self.debug:
            print(f"[DEBUG] _parse_message: recovered {tcs[0].function.name}({tcs[0].function.arguments[:120]})")

        return _DummyMessage(tcs)

    async def _emit_token_usage(self, response: Any, step: int) -> None:
        """
        Extracts token usage from a LiteLLM response and emits a TokenUsage record:
          - Appends to ExecutionContext.token_stats (when inside a WorkflowExecutor run).
          - Invokes self.on_token_usage callback (when set).

        Safe to call when no ExecutionContext is active (standalone use).
        """
        usage = getattr(response, "usage", None)
        record = TokenUsage(
            block_name=self.name,
            step=step,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )

        # Push to the shared ExecutionContext when running inside WorkflowExecutor
        try:
            ctx = _current_ctx.get()
            await ctx.add_token_usage(record)
        except LookupError:
            pass  # Running standalone, outside a WorkflowExecutor

        # Invoke optional user-supplied callback
        if self.on_token_usage:
            if inspect.iscoroutinefunction(self.on_token_usage):
                await self.on_token_usage(record)
            else:
                self.on_token_usage(record)

    async def run(self, input: AgentInput) -> AgentOutput:
        start_time = time.monotonic()

        # Transparent A2A Bridging: convert any sub-block into the Tool API format.
        litellm_tools = [block_to_tool_schema(b) for b in self.tools]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input.prompt}
        ]

        tool_call_count = 0
        self._current_tool_call_count = 0
        iteration_count = 0
        last_response: str = ""
        tool_usage: Dict[str, int] = defaultdict(int)
        termination_reason: str = "unknown"

        while True:
            if self.on_iteration:
                if inspect.iscoroutinefunction(self.on_iteration):
                    await self.on_iteration(iteration_count, messages)
                else:
                    self.on_iteration(iteration_count, messages)

            if self.max_iterations is not None and iteration_count >= self.max_iterations:
                if self.on_max_iterations == "return_last":
                    # Force a final LLM call without tools so the model synthesises
                    # the accumulated context. Synthesis instructions should live in
                    # the system_prompt — no extra message is injected here.
                    final_kwargs = self.litellm_kwargs.copy()
                    final_kwargs.pop("tools", None)
                    final_kwargs.pop("tool_choice", None)
                    final_kwargs["system_prompt"] = self.synthesis_prompt
                    
                    if self.response_schema:
                        final_kwargs["response_format"] = self.response_schema

                    if self.use_shared_router:
                        router = _get_shared_router(self.model)
                        final_resp = await router.acompletion(
                            model=self.model, messages=messages, **final_kwargs
                        )
                    else:
                        final_resp = await litellm.acompletion(
                            model=self.model, messages=messages, **final_kwargs
                        )

                    await self._emit_token_usage(final_resp, step=iteration_count)

                    termination_reason = "max_iterations reached → synthesised final response"
                    
                    content = final_resp.choices[0].message.content or last_response
                    structured_obj = None
                    if self.response_schema and content:
                        try:
                            clean_content = content.strip()
                            if clean_content.startswith("```json"):
                                clean_content = clean_content.split("```json", 1)[1].rsplit("```", 1)[0].strip()
                            elif clean_content.startswith("```"):
                                clean_content = clean_content.split("```", 1)[1].rsplit("```", 1)[0].strip()
                            structured_obj = self.response_schema.model_validate_json(clean_content)
                        except Exception as e:
                            if self.debug:
                                print(f"[DEBUG] Schema validation failed: {e}")
                    
                    output = AgentOutput(
                        response=content,
                        tool_calls_made=tool_call_count,
                        structured_output=structured_obj,
                    )
                else:
                    termination_reason = "max_iterations reached → stopped"
                    output = AgentOutput(
                        response="Agent stopped: Max iterations reached.",
                        tool_calls_made=tool_call_count
                    )

                if self.debug:
                    _print_debug_report(
                        agent_name=self.name,
                        model=self.model,
                        iteration_count=iteration_count,
                        tool_call_count=tool_call_count,
                        tool_usage=dict(tool_usage),
                        termination_reason=termination_reason,
                        elapsed_seconds=time.monotonic() - start_time,
                    )
                return output

            iteration_count += 1

            # Build optional kwargs: include tools when available; block new tool
            # calls once the per-run limit has been reached.
            kwargs = self.litellm_kwargs.copy()
            if litellm_tools:
                kwargs["tools"] = litellm_tools
                kwargs["tool_choice"] = "none" if tool_call_count >= self.max_tool_calls else "auto"

            # Enforce schema on the first call if no tools are present or tools are disabled
            if self.response_schema and (not litellm_tools or kwargs.get("tool_choice") == "none"):
                kwargs["response_format"] = self.response_schema

            # Main LiteLLM call.
            if self.use_shared_router:
                router = _get_shared_router(self.model)
                response = await router.acompletion(
                    model=self.model,
                    messages=messages,
                    **kwargs
                )
            else:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    **kwargs
                )

            await self._emit_token_usage(response, step=iteration_count)

            message = response.choices[0].message
            message = self._parse_message(message)

            # Track the last text produced by the LLM (used by on_max_iterations="return_last").
            if message.content:
                last_response = message.content

            # Build the assistant message dict manually: model_dump() deserialises
            # `arguments` into a dict, corrupting the history (the API requires
            # arguments to be a JSON string).
            assistant_message: Dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ]
            messages.append(assistant_message)

            # If no tool call was requested, the agent has finished reasoning.
            if not message.tool_calls:
                termination_reason = "model returned a final text response (no tool calls)"
                
                content = message.content or ""
                structured_obj = None
                if self.response_schema:
                    try:
                        clean_content = content.strip()
                        if clean_content.startswith("```json"):
                            clean_content = clean_content.split("```json", 1)[1].rsplit("```", 1)[0].strip()
                        elif clean_content.startswith("```"):
                            clean_content = clean_content.split("```", 1)[1].rsplit("```", 1)[0].strip()
                        structured_obj = self.response_schema.model_validate_json(clean_content)
                    except Exception:
                        # Fallback synthesis formatting call to force-convert conversations into the schema format
                        final_kwargs = self.litellm_kwargs.copy()
                        final_kwargs.pop("tools", None)
                        final_kwargs.pop("tool_choice", None)
                        final_kwargs["response_format"] = self.response_schema
                        
                        if self.use_shared_router:
                            router = _get_shared_router(self.model)
                            final_resp = await router.acompletion(
                                model=self.model,
                                messages=messages,
                                **final_kwargs
                            )
                        else:
                            final_resp = await litellm.acompletion(
                                model=self.model,
                                messages=messages,
                                **final_kwargs
                            )
                        
                        await self._emit_token_usage(final_resp, step=iteration_count)
                        content = final_resp.choices[0].message.content or ""
                        try:
                            clean_content = content.strip()
                            if clean_content.startswith("```json"):
                                clean_content = clean_content.split("```json", 1)[1].rsplit("```", 1)[0].strip()
                            elif clean_content.startswith("```"):
                                clean_content = clean_content.split("```", 1)[1].rsplit("```", 1)[0].strip()
                            structured_obj = self.response_schema.model_validate_json(clean_content)
                        except Exception as e:
                            if self.debug:
                                print(f"[DEBUG] Schema validation failed on fallback: {e}")
                
                output = AgentOutput(
                    response=content,
                    tool_calls_made=tool_call_count,
                    structured_output=structured_obj,
                )
                if self.debug:
                    _print_debug_report(
                        agent_name=self.name,
                        model=self.model,
                        iteration_count=iteration_count,
                        tool_call_count=tool_call_count,
                        tool_usage=dict(tool_usage),
                        termination_reason=termination_reason,
                        elapsed_seconds=time.monotonic() - start_time,
                    )
                return output

            # Transparent Execution (A2A and MCP).
            for tool_call in message.tool_calls:
                tool_call_count += 1
                self._current_tool_call_count = tool_call_count
                function_name = tool_call.function.name
                tool_usage[function_name] += 1

                # Look for the matching native tool (connected blocks).
                matched_block = next((b for b in self.tools if b.name == function_name), None)
                if not matched_block:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps({"error": f"Tool '{function_name}' not found."})
                    })
                    continue

                try:
                    # Parse arguments with the block's Pydantic input model (A2A bridge).
                    args_dict = json.loads(tool_call.function.arguments)
                    input_model = matched_block.input_schema()(**args_dict)

                    # RUN: the main agent transparently triggers a subordinate agent (A2A).
                    result = await matched_block.run(input=input_model)

                    # The typed output is serialised back to JSON for LiteLLM's history.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(result.model_dump(exclude_none=True) if hasattr(result, "model_dump") else result)
                    })
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps({"error": str(e)})
                    })

            # If the tool-call limit was reached, force a final response without tools.
            # This is necessary because some models (e.g. Ollama) ignore
            # tool_choice="none", causing an infinite loop.
            if tool_call_count >= self.max_tool_calls:
                final_kwargs = self.litellm_kwargs.copy()
                final_kwargs.pop("tool_choice", None)
                if self.response_schema:
                    final_kwargs["response_format"] = self.response_schema

                if self.use_shared_router:
                    router = _get_shared_router(self.model)
                    final_response = await router.acompletion(
                        model=self.model,
                        messages=messages,
                        **final_kwargs
                    )
                else:
                    final_response = await litellm.acompletion(
                        model=self.model,
                        messages=messages,
                        **final_kwargs
                    )

                await self._emit_token_usage(final_response, step=iteration_count)

                termination_reason = f"max_tool_calls ({self.max_tool_calls}) reached → forced final response"
                
                content = final_response.choices[0].message.content or ""
                structured_obj = None
                if self.response_schema and content:
                    try:
                        clean_content = content.strip()
                        if clean_content.startswith("```json"):
                            clean_content = clean_content.split("```json", 1)[1].rsplit("```", 1)[0].strip()
                        elif clean_content.startswith("```"):
                            clean_content = clean_content.split("```", 1)[1].rsplit("```", 1)[0].strip()
                        structured_obj = self.response_schema.model_validate_json(clean_content)
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Schema validation failed: {e}")

                output = AgentOutput(
                    response=content,
                    tool_calls_made=tool_call_count,
                    structured_output=structured_obj,
                )
                if self.debug:
                    _print_debug_report(
                        agent_name=self.name,
                        model=self.model,
                        iteration_count=iteration_count,
                        tool_call_count=tool_call_count,
                        tool_usage=dict(tool_usage),
                        termination_reason=termination_reason,
                        elapsed_seconds=time.monotonic() - start_time,
                    )
                return output