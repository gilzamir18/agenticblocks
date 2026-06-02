from agenticblocks.blocks.llm.memgpt_agent import MemGPTAgentBlock
import asyncio
import os
import sys

# Adiciona src ao path para rodar localmente sem instalar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))

from agenticblocks.blocks.memory import ChromaArchivalMemory, SQLiteRecallMemory
from agenticblocks.core.function_block import as_tool
from agenticblocks.blocks.llm.agent import AgentInput

async def main():
    print("Iniciando memórias do agente...")
    
    # 1. Instancia as memórias (usaremos persistência local neste diretório)
    base_dir = os.path.dirname(__file__)
    archival = ChromaArchivalMemory(
        collection_name="conhecimento", 
        persist_directory=os.path.join(base_dir, "mem_archival")
    )
    recall = SQLiteRecallMemory(
        db_path=os.path.join(base_dir, "mem_recall.db")
    )
    
    # 2. Popula a Archival Memory na primeira execução (Opcional)
    try:
        if len(archival.collection.get()["ids"]) == 0:
            print("Inicializando Archival Memory limpa para a Companhia Virtual...")
    except Exception as e:
        print(f"Aviso ao verificar/popular memória: {e}")

    # 3. Define as ferramentas para o Agente
    @as_tool(name="search_archival", description="Busca manuais e fatos da empresa por significado semântico. Use quando o usuário perguntar informações da empresa ou dos produtos.")
    def search_archival(query: str, page: int = 1) -> str:
        print(f"\n[DEBUG] Agente buscando na Archival Memory por: '{query}'...")
        resultados = archival.search(query, page=page, page_size=3)
        if not resultados:
            return "Nenhuma informação encontrada na Archival Memory."
        
        text = f"--- Resultados Archival (Página {page}) ---\n"
        for r in resultados:
            text += f"- {r['content']} (Meta: {r['metadata']})\n"
        return text

    @as_tool(name="search_recall", description="Busca mensagens de conversas passadas com o usuário por palavra-chave. Use quando o usuário mencionar que já falou algo antes ou quando você não souber algo que já deveria saber.")
    def search_recall(keyword: str) -> str:
        print(f"\n[DEBUG] Agente buscando na Recall Memory por: '{keyword}'...")
        resultados = recall.search_keyword(keyword, limit=5)
        if not resultados:
            return "Nenhuma lembrança encontrada na Recall Memory."
        
        text = "--- Resultados Recall (Histórico) ---\n"
        for r in resultados:
            text += f"[{r['timestamp']}] {r['role'].upper()}: {r['content']}\n"
        return text

    @as_tool(name="save_archival", description="Salva de forma persistente fatos importantes sobre o usuário (ex: nome, gostos) ou conhecimentos relevantes na Archival Memory para uso futuro.")
    def save_archival(content: str, type_meta: str = "fato") -> str:
        print(f"\n[DEBUG] Agente salvando na Archival Memory: '{content}'...")
        archival.insert(content, metadata={"tipo": type_meta})
        return "Informação salva com sucesso na Archival Memory."

    prompt_path = os.path.join(base_dir, "MEMGPT.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        memgpt_system_prompt = f.read()

    agent = MemGPTAgentBlock(
        name="memgpt_chatbot",
        model=os.getenv("AGENTICBLOCKS_MODEL", "ollama/mistral-nemo:latest"),
        model_kargs={"fallbacks":["ollama/gemma4:latest"], "num_ctx":8128},
        max_heartbeats=5,
        tool_call_limits={"send_message":1},
        debug=True, # <--- ATIVA RELATÓRIO DE EXECUÇÃO
        system_prompt=memgpt_system_prompt,
        tools=[search_archival, search_recall, save_archival]
    )

    print("\n" + "="*60)
    print("Chatbot MemGPT Iniciado! (Digite 'sair' para encerrar)")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("Você: ")
            if user_input.lower() in ['sair', 'quit', 'exit']:
                break
                
            if not user_input.strip():
                continue
                
            # Salva a entrada do usuário na recall memory
            recall.append_message(role="user", content=user_input)
            
            # Executa o agente
            result = await agent.run(AgentInput(prompt=user_input))
            response = result.response
            
            # Salva a resposta do agente na recall
            recall.append_message(role="assistant", content=response)
            print(f"\nAgente: {response}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nErro durante execução: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())
