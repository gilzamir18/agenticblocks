import sys
import os
import asyncio
import json
import urllib.request
import urllib.parse

# Fallback path if run isolated
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pydantic import BaseModel
from agenticblocks.core.block import Block
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput

# ---------------------------------------------------------
# Ferramenta 1: Busca na Web Pública (Wikipedia Livre)
# ---------------------------------------------------------
class WebSearchInput(BaseModel):
    query: str

class WebSearchOutput(BaseModel):
    result: str

class WikipediaSearchBlock(Block[WebSearchInput, WebSearchOutput]):
    name: str = "web_search"
    description: str = "Usa a Wikipedia para buscar fatos públicos ou históricos da internet. Retorna um resumo da busca."

    async def run(self, input: WebSearchInput) -> WebSearchOutput:
        print(f"  [Ferramenta] 🌐 Buscando '{input.query}' na Wikipedia...")
        try:
            url = f"https://pt.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(input.query)}&utf8=&format=json"
            req = urllib.request.urlopen(url)
            res = json.loads(req.read())
            if len(res['query']['search']) > 0:
                summary = res['query']['search'][0]['snippet']
                summary = summary.replace('<span class="searchmatch">', '').replace('</span>', '')
                return WebSearchOutput(result=summary)
            else:
                return WebSearchOutput(result="Nada encontrado na Wikipedia para este termo.")
        except Exception as e:
             return WebSearchOutput(result=f"Erro ao consultar web: {e}")

# ---------------------------------------------------------
# Ferramenta 2: Leitura de Arquivo Local
# ---------------------------------------------------------
class LocalFileInput(BaseModel):
    filename: str

class LocalFileOutput(BaseModel):
    lines: list[str]
    error: str = ""

class LocalFileBlock(Block[LocalFileInput, LocalFileOutput]):
    name: str = "read_local_file"
    description: str = "Tenta ler um arquivo txt local na maquina. Útil para extrair manuais ou regras de negócio."

    async def run(self, input: LocalFileInput) -> LocalFileOutput:
        print(f"  [Ferramenta] 📁 Tentando ler o arquivo '{input.filename}'...")
        # Simula erro de digitação de caminhos para o LLM aprender e tentar de novo:
        if input.filename != "regras_internas.txt":
            return LocalFileOutput(lines=[], error="FileNotFoundError: O arquivo solicitado não existe. O único arquivo existente é 'regras_internas.txt'.")
        
        return LocalFileOutput(lines=[
            "Regra 1: O horário comercial é das 9h às 18h.",
            "Regra 2: Reuniões devem ser marcadas com 24h de antecedência."
        ])

# ---------------------------------------------------------
# Ferramenta 3: Base de Conhecimento Estruturada (Mock SQL/API)
# ---------------------------------------------------------
class KnowledgeBaseInput(BaseModel):
    employee_name: str

class KnowledgeBaseOutput(BaseModel):
    info: str

class KnowledgeBaseBlock(Block[KnowledgeBaseInput, KnowledgeBaseOutput]):
    name: str = "query_structured_db"
    description: str = "Consulta a base de dados SQL fechada e estruturada da empresa buscando por dados de funcionários."

    async def run(self, input: KnowledgeBaseInput) -> KnowledgeBaseOutput:
        print(f"  [Ferramenta] 🗄️ Consultando BD por dados do funcionário '{input.employee_name}'...")
        db = {
            "joao carlos": "João Carlos: Analista de Sistemas - Nível 3 - Férias Marcadas.",
            "maria lima": "Maria Lima: Diretora de Vendas - SLA de Resposta 2 horas."
        }
        
        normalized = input.employee_name.lower().strip()
        if normalized in db:
            return KnowledgeBaseOutput(info=db[normalized])
        else:
            return KnowledgeBaseOutput(info="Funcionario não encontrado no BD. Pode ter sido demitido ou não é um funcionário.")


# ---------------------------------------------------------
# Configurando o Workflow do Raciocínio
# ---------------------------------------------------------
async def main():
    graph = WorkflowGraph()

    # 1. Instanciamos as ferramentas
    web_block = WikipediaSearchBlock()
    file_block = LocalFileBlock()
    db_block = KnowledgeBaseBlock()

    # 2. Roteador/Orquestrador do Raciocínio do Agent
    # O LLM é instruído a usar as ferramentas para sanar lacunas
    researcher_agent = LLMAgentBlock(
        name="researcher_super_agent",
        model="gemini/gemini-3-flash-preview", # Usando o que você usou com sucesso antes!
        system_prompt=(
            "Você é um Pesquisador Autônomo Metódico (Research Agent). "
            "Seu objetivo é responder a pergunta do usuário da forma mais acurada possível. "
            "Para isso, você tem acesso a 3 ferramentas. VOCÊ DEVE RACIOCINAR e escolher qual usar:\n"
            "- Se pediu uma informação histórica/pública: use 'web_search'.\n"
            "- Se pediu sobre diretrizes ou regras internas da empresa: use 'read_local_file'.\n"
            "- Se perguntou sobre o perfil ou dados de um funcionário: use 'query_structured_db'.\n\n"
            "Se a sua primeira ferramenta falhar (ou voltar arquivo não encontrado/registro não achado), "
            "RACIOCINE e tente corrigir os parâmetros ou chame uma fonte pública se tiver dúvidas."
        ),
        tools=[web_block, file_block, db_block] 
    )

    graph.add_block(researcher_agent)
    executor = WorkflowExecutor(graph)

    print("====================================")
    print(" 🤖 INICIANDO O AGENTE PESQUISADOR")
    print("====================================\n")

    pergunta_usuario = "Você pode ler as regras operacionais da empresa pra mim do arquivo regras.txt? E também me falar quem foi o imperador do Brasil?"
    print(f"👤 Usuário: {pergunta_usuario}\n")

    try:
        ctx = await executor.run(initial_input={"prompt": pergunta_usuario})
        output = ctx.get_output("researcher_super_agent")
        
        print("\n[🎯 Resposta Final do Agente]:")
        print(output.response)
        print(f"\n-> [Estatística] O Agente raciocinou e rodou {output.tool_calls_made} ferramenta(s) antes de entregar sua resposta.")
        
    except Exception as e:
         print(f"\n[🛑 Falha no provedor LLM]: {e}")

if __name__ == "__main__":
    asyncio.run(main())
