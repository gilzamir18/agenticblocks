from typing import Any, Dict
from agenticblocks.core.block import Block

def block_to_tool_schema(block: Block) -> Dict[str, Any]:
    """Generates an OpenAI-compatible function schema from a Block transparently."""
    
    # Se for um MCP Tool direto com Schema pronto do servidor:
    if getattr(block, "is_mcp_proxy", False):
        schema = block.raw_mcp_schema
    else:
        # Obtain dynamic Pydantic Schema representation para Python Tools nativas
        schema = block.input_schema().model_json_schema()
    
    return {
        "type": "function",
        "function": {
            "name": block.name,
            "description": block.description or f"Executa a tarefa do bloco: {block.name}",
            "parameters": schema
        }
    }
