import json
import time
import inspect
from collections import defaultdict
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Callable
import litellm
from agenticblocks.core.agent import AgentBlock
from agenticblocks.core.block import Block
from agenticblocks.tools.a2a_bridge import block_to_tool_schema


class AgentInput(BaseModel):
    prompt: str

class AgentOutput(BaseModel):
    response: str
    tool_calls_made: int = 0


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
    model: str = "gpt-4o-mini"
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
    debug: bool = False
    """When True, print a structured debug report at the end of each run."""
    use_shared_router: bool = True
    """When True, uses a shared litellm.Router for connection pooling."""
    litellm_kwargs: Dict[str, Any] = Field(default_factory=dict)
    on_iteration: Optional[Callable[[int, List[Dict[str, Any]]], Any]] = None
    """Optional callback invoked at the start of each loop iteration for debugging. 
    Signature: `def callback(iteration: int, messages: List[Dict[str, Any]]) -> Any`.
    Can be a synchronous or asynchronous function."""

    model_config = {"arbitrary_types_allowed": True}

    def _parse_message(self, message: Any) -> Any:
        """
        Hook for subclasses to manipulate the LiteLLM message before it is processed.
        Useful for intercepting hallucinated JSON tool calls in plain text.
        """
        return message

    async def run(self, input: AgentInput) -> AgentOutput:
        start_time = time.monotonic()

        # Transparent A2A Bridging: convert any sub-block into the Tool API format.
        litellm_tools = [block_to_tool_schema(b) for b in self.tools]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input.prompt}
        ]

        tool_call_count = 0
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
                    
                    if self.use_shared_router:
                        router = _get_shared_router(self.model)
                        final_resp = await router.acompletion(
                            model=self.model, messages=messages, **final_kwargs
                        )
                    else:
                        final_resp = await litellm.acompletion(
                            model=self.model, messages=messages, **final_kwargs
                        )
                    
                    termination_reason = "max_iterations reached → synthesised final response"
                    output = AgentOutput(
                        response=final_resp.choices[0].message.content or last_response,
                        tool_calls_made=tool_call_count,
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
                output = AgentOutput(
                    response=message.content or "",
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

            # Transparent Execution (A2A and MCP).
            for tool_call in message.tool_calls:
                tool_call_count += 1
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
                    
                termination_reason = f"max_tool_calls ({self.max_tool_calls}) reached → forced final response"
                output = AgentOutput(
                    response=final_response.choices[0].message.content or "",
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