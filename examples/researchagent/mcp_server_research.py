import asyncio
import json
import urllib.request
import urllib.parse
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("banco-pesquisa-mcp")

@app.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="web_search",
            description="Usa a Wikipedia para buscar fatos públicos ou históricos da internet. Retorna um resumo da busca.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "O termo ou frase para pesquisar na web."
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="read_local_file",
            description="Tenta ler um arquivo txt local na maquina. Útil para extrair manuais ou regras de negócio.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string", 
                        "description": "O nome absoluto ou relativo do arquivo a ser lido."
                    }
                },
                "required": ["filename"]
            }
        ),
        Tool(
            name="query_structured_db",
            description="Consulta a base de dados SQL fechada e estruturada da empresa buscando por dados de funcionários.",
            inputSchema={
                "type": "object",
                "properties": {
                    "employee_name": {
                        "type": "string", 
                        "description": "Nome do funcionário para buscar no Banco de Dados."
                    }
                },
                "required": ["employee_name"]
            }
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    if arguments is None:
        arguments = {}

    if name == "web_search":
        query = arguments.get("query", "")
        print(f"[Research Server] 🌐 Wikipedia: buscando '{query}'...", flush=True)
        try:
            url = f"https://pt.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
            req = urllib.request.urlopen(url)
            res = json.loads(req.read())
            if len(res['query']['search']) > 0:
                summary = res['query']['search'][0]['snippet']
                summary = summary.replace('<span class="searchmatch">', '').replace('</span>', '')
                resposta = summary
            else:
                resposta = "Nada encontrado na Wikipedia para este termo."
        except Exception as e:
            resposta = f"Erro ao consultar web: {e}"
            
        return [TextContent(type="text", text=resposta)]

    elif name == "read_local_file":
        filename = arguments.get("filename", "")
        print(f"[Research Server] 📁 Arquivo: tentando ler '{filename}'...", flush=True)
        # Erro intencional didático para forçar raciocinio
        if filename != "regras_internas.txt":
            return [TextContent(type="text", text="FileNotFoundError: O arquivo solicitado não existe. Tente listar e usar o 'regras_internas.txt'.")]
        
        return [TextContent(type="text", text="Regras: 1- Todo equipamento é restrito. 2- Horário oficial até as 18h.")]

    elif name == "query_structured_db":
        employee = arguments.get("employee_name", "")
        print(f"[Research Server] 🗄️ SQL DB: consultando '{employee}'...", flush=True)
        db = {
            "joao carlos": "João Carlos: Analista de Sistemas - Nível 3 - Férias Marcadas.",
            "maria lima": "Maria Lima: Diretora de Vendas - SLA de Resposta 2 horas."
        }
        
        normalized = employee.lower().strip()
        if normalized in db:
            resposta = db[normalized]
        else:
            resposta = "Funcionario não encontrado no BD. Pode ter sido demitido ou digitado com erro."
        return [TextContent(type="text", text=resposta)]

    else:
        raise ValueError(f"Ferramenta desconhecida no servidor de pesquisa MCP: {name}")


async def main():
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
