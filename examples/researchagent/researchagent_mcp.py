import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks.tools.mcp_client import MCPClientBridge

async def main():
    python_exe = sys.executable
    server_path = os.path.join(os.path.dirname(__file__), "mcp_server_research.py")
    
    mcp_bridge = MCPClientBridge(
        command=python_exe,
        args=[server_path]
    )
    
    print("==============================================================")
    print(" 1. Iniciando MCP Cient e conectando ao Servidor de Pesquisa...")
    print("==============================================================\n")
    
    try:
        mcp_tools = await mcp_bridge.connect()
        print(f"✅ Conectado com Sucesso via Stdio! \nFerramentas de Pesquisa Externas Injetadas: {[t.name for t in mcp_tools]}\n")
    except Exception as e:
        print(f"❌ Falha de Conexão MCP: {e}")
        return

    researcher_agent = LLMAgentBlock(
        name="researcher_mcp_super_agent",
        model="gemini/gemini-3-flash-preview", 
        system_prompt=(
            "Você é um Pesquisador Autônomo Metódico (Research Agent). "
            "Seu objetivo é responder a pergunta do usuário da forma mais acurada possível. "
            "Você NÃO tem ferramentas nativas. Usa os recursos providos pelo Servidor MCP. "
            "Se for fatos históricos: use 'web_search'. "
            "Se for regras operacionais: use 'read_local_file'. "
            "Se for sobre funcionários: use 'query_structured_db'. "
            "RACIOCINE sobre erros das ferramentas caso retornem arquivo ou banco de dados inexistente."
        ),
        tools=mcp_tools 
    )
    
    graph = WorkflowGraph()
    graph.add_block(researcher_agent)
    executor = WorkflowExecutor(graph)
    
    if not os.getenv("GEMINI_API_KEY"):
         print("⚠️ Lembrete: GEMINI_API_KEY necessária para o raciocínio final correr.\n")
         
    try:
        pergunta_usuario = "Você pode ler as regras operacionais da nossa empresa pra mim? Qual horário máximo? E também me falar quem foi o imperador do Brasil?"
        print(f"👤 Usuário: {pergunta_usuario}\n")
        
        ctx = await executor.run(initial_input={"prompt": pergunta_usuario})
        
        output = ctx.get_output("researcher_mcp_super_agent")
        print("\n[🎯 Resposta Final do Agente]:")
        print(output.response)
        print(f"\n-> [Estatística] Ferramentas MCP invocadas via Client/Server: {output.tool_calls_made}")
        
    except Exception as e:
        print(f"\n[🛑 Falha no provedor LLM]: {e}")
        
    finally:
        await mcp_bridge.disconnect()
        print("\n👋 Conexão MCP Encerrada de Forma Limpa com o Servidor.")

if __name__ == "__main__":
    asyncio.run(main())
