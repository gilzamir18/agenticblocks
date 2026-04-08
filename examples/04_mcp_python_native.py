import sys
import os
import asyncio

from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks.tools.mcp_client import MCPClientBridge

async def main():
    # Detectamos o caminho absoluto correto para invocar o servidor Python como subprocesso
    python_exe = sys.executable
    server_path = os.path.join(os.path.dirname(__file__), "mcp_server_estoque.py")
    
    # 1. Configurar a Ponte do Cliente
    # Em vez de npx, passamos o próprio Python rodando nosso server local
    mcp_bridge = MCPClientBridge(
        command=python_exe,
        args=[server_path]
    )
    
    print("Iniciando MCP Client e trocando handshake com Servidor Python de Estoque (Stdio)...")
    try:
        # A magia: nosso client.py extrai as tools direto do mcp_server_estoque.py
        mcp_tools = await mcp_bridge.connect()
        print(f"✅ Conectado! Ferramentas Descobertas no Servidor: {[t.name for t in mcp_tools]}\n")
    except Exception as e:
        print(f"❌ Falha de Conexão MCP: {e}")
        return

    # 2. Injetamos isso de forma transparente no nosso LLM!
    agente_comprador = LLMAgentBlock(
        name="agente_comprador",
        model="ollama/granite4:1b",
        system_prompt="Você é o assistente de compras da nossa matriz. Use a ferramenta do servidor para verificar O PREÇO e a DISPONIBILIDADE exata do que o usuário pediru.",
        tools=mcp_tools 
    )
    
    graph = WorkflowGraph()
    graph.add_block(agente_comprador)
    executor = WorkflowExecutor(graph)
    if agente_comprador.model.startswith("openai"):
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️ Lembrete: Defina OPENAI_API_KEY para a resposta final fluir do LiteLLM.")
         
    try:
        # 3. Disparamos o Loop de Raciocínio
        print("👤 Usuário: Estou montando um setup. Tem teclado e mouse disponível no nosso galpão?")
        ctx = await executor.run(initial_input={"prompt": "Estou montando um setup. Tem teclado e monitor disponível no nosso galpão?"})
        
        output = ctx.get_output("agente_comprador")
        print("\n[🎯 Resposta Final do Agente]:")
        print(output.response)
        print(f"-> [Estatística] Ferramentas MCP invocadas pela rede iterativamente: {output.tool_calls_made}")
        
    except Exception as e:
        print(f"\n[🛑 Falha no provedor LLM]: {e}")
        
    finally:
        # 4. Encerra processo stdio base de forma elegante
        await mcp_bridge.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
