import json
import time
import inspect
from collections import defaultdict
from pydantic import Field
from typing import List, Dict, Any, Optional
import litellm

from agenticblocks.core.agent import AgentBlock
from agenticblocks.blocks.llm.agent import AgentInput, AgentOutput, _get_shared_router, _print_debug_report
from agenticblocks.tools.a2a_bridge import block_to_tool_schema
from agenticblocks.core.block import Block
from agenticblocks.core.function_block import as_tool
from agenticblocks.runtime.state import TokenUsage, _current_ctx

class MemGPTAgentBlock(AgentBlock[AgentInput, AgentOutput]):
    """
    An Autonomous LLM Agent that strictly follows the MemGPT Heartbeat paradigm.
    
    In this block:
    1. The LLM MUST use the `send_message` tool to communicate with the user.
    2. Any tool call (including search) consumes 1 heartbeat.
    3. The LLM is explicitly informed of its remaining heartbeats.
    4. The loop only terminates if the LLM explicitly returns `request_heartbeat=false` 
       via the `send_message` tool, or if the `max_heartbeats` limit is reached.
    """
    description: str = "MemGPT style Agent with strict heartbeat limits."
    model: str = "gpt-4o-mini"
    system_prompt: str = (
        "You are an agent with advanced memory capabilities.\n"
        "RULES:\n"
        "1. You MUST ALWAYS communicate with the user by calling the `send_message` tool. NEVER reply with plain text.\n"
        "2. If you need to search memory or use tools, just call them.\n"
        "3. Every tool call you make (including `send_message` with `request_heartbeat=true`) consumes 1 heartbeat.\n"
    )
    tools: List[Block] = []
    max_heartbeats: int = 10
    debug: bool = False
    use_shared_router: bool = True
    litellm_kwargs: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    async def _emit_token_usage(self, response: Any, step: int) -> None:
        usage = getattr(response, "usage", None)
        record = TokenUsage(
            block_name=self.name,
            step=step,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )
        try:
            ctx = _current_ctx.get()
            await ctx.add_token_usage(record)
        except LookupError:
            pass

    async def run(self, input: AgentInput) -> AgentOutput:
        start_time = time.monotonic()

        agent_tools = self.tools.copy()
        
        @as_tool(name="send_message", description="Sends a message to the user. Set request_heartbeat=true if you want to perform more actions (like searching memory) before giving control back to the user.")
        def send_message(message: str, request_heartbeat: bool = False) -> str:
            return "Message recorded."
            
        agent_tools.append(send_message)
        litellm_tools = [block_to_tool_schema(b) for b in agent_tools]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input.prompt}
        ]

        heartbeats_used = 0
        tool_call_count = 0
        tool_usage: Dict[str, int] = defaultdict(int)
        termination_reason = "unknown"
        accumulated_responses = []

        while True:
            heartbeats_left = self.max_heartbeats - heartbeats_used
            kwargs = self.litellm_kwargs.copy()
            kwargs["tools"] = litellm_tools
            
            # Se não houver mais heartbeats, força a enviar a mensagem final
            if heartbeats_left <= 0:
                kwargs["tool_choice"] = {"type": "function", "function": {"name": "send_message"}}
                messages.append({
                    "role": "system", 
                    "content": "SYSTEM ALERT: 0 heartbeats remaining. You MUST call send_message with request_heartbeat=false now to finish the turn."
                })
            else:
                kwargs["tool_choice"] = "auto"
                
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

            await self._emit_token_usage(response, step=heartbeats_used)

            message = response.choices[0].message
            
            assistant_message: Dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ]
            messages.append(assistant_message)

            if not message.tool_calls:
                # O modelo falhou em usar ferramentas e respondeu em texto livre.
                if message.content:
                    accumulated_responses.append(message.content)
                termination_reason = "model returned plain text (violated tool-only rule)"
                break

            heartbeats_used += 1
            wants_heartbeat = False
            
            for tool_call in message.tool_calls:
                tool_call_count += 1
                function_name = tool_call.function.name
                tool_usage[function_name] += 1

                if function_name == "send_message":
                    try:
                        args = json.loads(tool_call.function.arguments)
                        msg_text = args.get("message", "")
                        if msg_text:
                            accumulated_responses.append(msg_text)
                        
                        hb_req = args.get("request_heartbeat", False)
                        if hb_req:
                            wants_heartbeat = True
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": f"Message recorded. Heartbeats remaining: {self.max_heartbeats - heartbeats_used}."
                        })
                    except Exception as e:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps({"error": str(e)})
                        })
                else:
                    # Executa outras ferramentas
                    wants_heartbeat = True # Outras ferramentas implicitamente pedem heartbeat
                    matched_block = next((b for b in agent_tools if b.name == function_name), None)
                    if not matched_block:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps({"error": f"Tool '{function_name}' not found."})
                        })
                        continue

                    try:
                        args_dict = json.loads(tool_call.function.arguments)
                        input_model = matched_block.input_schema()(**args_dict)
                        result = await matched_block.run(input=input_model)
                        content_str = json.dumps(result.model_dump(exclude_none=True) if hasattr(result, "model_dump") else result)
                        
                        hb_left = self.max_heartbeats - heartbeats_used
                        content_str += f"\n[System: You have {hb_left} heartbeats remaining.]"
                            
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": content_str
                        })
                    except Exception as e:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps({"error": str(e)})
                        })

            if not wants_heartbeat:
                termination_reason = "send_message called with request_heartbeat=false"
                break
            
            if heartbeats_used >= self.max_heartbeats:
                termination_reason = f"max_heartbeats ({self.max_heartbeats}) reached"
                break

        final_text = "\n".join(accumulated_responses)
        output = AgentOutput(
            response=final_text,
            tool_calls_made=tool_call_count
        )

        if self.debug:
            _print_debug_report(
                agent_name=self.name,
                model=self.model,
                iteration_count=heartbeats_used,
                tool_call_count=tool_call_count,
                tool_usage=dict(tool_usage),
                termination_reason=termination_reason,
                elapsed_seconds=time.monotonic() - start_time,
            )
        return output
