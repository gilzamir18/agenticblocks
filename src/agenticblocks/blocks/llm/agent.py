import json
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import litellm
from agenticblocks.core.agent import AgentBlock
from agenticblocks.core.block import Block
from agenticblocks.tools.a2a_bridge import block_to_tool_schema


class AgentInput(BaseModel):
    prompt: str

class AgentOutput(BaseModel):
    response: str
    tool_calls_made: int = 0

class LLMAgentBlock(AgentBlock[AgentInput, AgentOutput]):
    description: str = "Agente Autônomo Baseado em LLM gerenciando seu próprio Tool Loop."
    model: str = "gpt-4o-mini"
    system_prompt: str = "Você é um Agente Analista e Roteador prestativo. Use as ferramentas caso não possua contexto."
    tools: List[Block] = []
    max_iterations: Optional[int] = None
    max_tool_calls: int = 2
    litellm_kwargs: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"arbitrary_types_allowed": True}
    
    async def run(self, input: AgentInput) -> AgentOutput:
        # Transparent A2A Bridging 
        # Converter qualquer Sub-Bloco para Tool API Formats
        litellm_tools = [block_to_tool_schema(b) for b in self.tools]
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input.prompt}
        ]
        
        tool_call_count = 0
        iteration_count = 0
        
        while True:
            if self.max_iterations is not None and iteration_count >= self.max_iterations:
                return AgentOutput(
                    response="Agent stopped: Max iterations reached.",
                    tool_calls_made=tool_call_count
                )
                
            iteration_count += 1
            
            # Argumentos opcionais caso ferramentas existam no escopo do Agente e os args persistentes base
            kwargs = self.litellm_kwargs.copy()
            if litellm_tools:
                kwargs["tools"] = litellm_tools
                # Após atingir o limite de chamadas, proíbe novas ferramentas
                kwargs["tool_choice"] = "none" if tool_call_count >= self.max_tool_calls else "auto"
            
            # Chamada principal com LiteLLM
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                **kwargs
            )
            
            message = response.choices[0].message
            # Append dict format rather than object back to history
            messages.append(message.model_dump(exclude_none=True))
            
            # Se não tomou decisão de chamar ferramentas, finalizamos iterando o raciocinio e extraindo a reposta.
            if not message.tool_calls:
                return AgentOutput(
                    response=message.content or "",
                    tool_calls_made=tool_call_count
                )
                
            # Transparent Execution! (A2A e MCP)
            for tool_call in message.tool_calls:
                tool_call_count += 1
                function_name = tool_call.function.name
                
                # Procura a ferramenta nativa (Blocos conectados)
                matched_block = next((b for b in self.tools if b.name == function_name), None)
                if not matched_block:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps({"error": f"Tool {function_name} not found."})
                    })
                    continue
                    
                try:
                    # Roda o Pydantic Parse para o Bloco A2A dinamicamente
                    args_dict = json.loads(tool_call.function.arguments)
                    input_model = matched_block.input_schema()(**args_dict)
                    
                    # RUN: O Agente principal engatilha um Agente Subordinado de forma transparente (A2A)!
                    result = await matched_block.run(input=input_model)
                    
                    # O output tipado retorna ao escopo original do LiteLLM como JSON
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(result.model_dump())
                    })
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps({"error": str(e)})
                    })


# Registry de Routers compartilhados por modelo — padrão Flyweight.
# LiteLLM.Router gerencia pooling de conexões; o mesmo Router é reutilizado
# para todas as instâncias de bloco que usam o mesmo modelo.
_router_registry: Dict[str, litellm.Router] = {}

def _get_shared_router(model: str) -> litellm.Router:
    """Retorna (criando se necessário) um Router compartilhado para o modelo."""
    if model not in _router_registry:
        _router_registry[model] = litellm.Router(
            model_list=[{
                "model_name": model,
                "litellm_params": {"model": model},
            }]
        )
    return _router_registry[model]

class SharedLLMAgentBlock(AgentBlock[AgentInput, AgentOutput]):
    description: str = "Agente Autônomo Baseado em LLM gerenciando seu próprio Tool Loop."
    model: str = "gpt-4o-mini"
    system_prompt: str = "Você é um Agente Analista e Roteador prestativo. Use as ferramentas caso não possua contexto."
    tools: List[Block] = []
    max_iterations: Optional[int] = None
    max_tool_calls: int = 2
    litellm_kwargs: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"arbitrary_types_allowed": True}
    
    async def run(self, input: AgentInput) -> AgentOutput:
        # Transparent A2A Bridging 
        # Converter qualquer Sub-Bloco para Tool API Formats
        litellm_tools = [block_to_tool_schema(b) for b in self.tools]
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input.prompt}
        ]
        
        tool_call_count = 0
        iteration_count = 0
        
        while True:
            if self.max_iterations is not None and iteration_count >= self.max_iterations:
                return AgentOutput(
                    response="Agent stopped: Max iterations reached.",
                    tool_calls_made=tool_call_count
                )
                
            iteration_count += 1
            
            # Argumentos opcionais caso ferramentas existam no escopo do Agente e os args persistentes base
            kwargs = self.litellm_kwargs.copy()
            if litellm_tools:
                kwargs["tools"] = litellm_tools
                # Após atingir o limite de chamadas, proíbe novas ferramentas
                kwargs["tool_choice"] = "none" if tool_call_count >= self.max_tool_calls else "auto"
            
            # Chamada principal via Router compartilhado (instância única por modelo)
            router = _get_shared_router(self.model)
            response = await router.acompletion(
                model=self.model,
                messages=messages,
                **kwargs
            )
            
            message = response.choices[0].message
            # Append dict format rather than object back to history
            messages.append(message.model_dump(exclude_none=True))
            
            # Se não tomou decisão de chamar ferramentas, finalizamos iterando o raciocinio e extraindo a reposta.
            if not message.tool_calls:
                return AgentOutput(
                    response=message.content or "",
                    tool_calls_made=tool_call_count
                )
                
            # Transparent Execution! (A2A e MCP)
            for tool_call in message.tool_calls:
                tool_call_count += 1
                function_name = tool_call.function.name
                
                # Procura a ferramenta nativa (Blocos conectados)
                matched_block = next((b for b in self.tools if b.name == function_name), None)
                if not matched_block:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps({"error": f"Tool {function_name} not found."})
                    })
                    continue
                    
                try:
                    # Roda o Pydantic Parse para o Bloco A2A dinamicamente
                    args_dict = json.loads(tool_call.function.arguments)
                    input_model = matched_block.input_schema()(**args_dict)
                    
                    # RUN: O Agente principal engatilha um Agente Subordinado de forma transparente (A2A)!
                    result = await matched_block.run(input=input_model)
                    # O output tipado retorna ao escopo original do LiteLLM como JSON
                    #if isinstance(result, AgentOutput):
                    #    result = result.response
                    #else:
                    #    result = result.model_dump(exclude_none=True) if hasattr(result, "model_dump") else result
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

            # Se atingiu o limite de tool calls, força uma resposta final sem ferramentas.
            # Necessário porque alguns modelos (ex: Ollama) ignoram tool_choice="none"
            # causando loop infinito.
            if tool_call_count >= self.max_tool_calls:
                final_kwargs = self.litellm_kwargs.copy()
                final_kwargs.pop("tool_choice", None)
                final_response = await router.acompletion(
                    model=self.model,
                    messages=messages,
                    **final_kwargs
                )
                return AgentOutput(
                    response=final_response.choices[0].message.content or "",
                    tool_calls_made=tool_call_count
                )