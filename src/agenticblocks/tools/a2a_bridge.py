import inspect
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
        
    class_doc = inspect.getdoc(block.__class__)
    if class_doc and class_doc.startswith("Usage docs:"):
        class_doc = ""
        
    run_doc = inspect.getdoc(block.run)
    
    doc_parts = []
    if getattr(block, "description", None):
        doc_parts.append(block.description)
    if class_doc:
        doc_parts.append(f"Detalhes: {class_doc}")
    if run_doc:
        doc_parts.append(f"Instruções: {run_doc}")
        
    final_description = "\n\n".join(doc_parts).strip()
    if not final_description:
        final_description = f"Executa a tarefa do bloco: {block.name}"
    
    return {
        "type": "function",
        "function": {
            "name": block.name,
            "description": final_description,
            "parameters": schema
        }
    }
