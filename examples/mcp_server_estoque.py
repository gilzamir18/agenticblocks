import asyncio
from mcp.server import Server
from mcp.types import Tool, TextContent

# 1. Definimos o nome do nosso servidor local
app = Server("banco-estoque-mcp")

# 2. Informamos ao protocolo MCP (e portanto ao mundo) quais 'Tools' nós oferecemos
@app.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="consultar_estoque",
            description="Consulta o banco de dados da empresa para checar quantidade em estoque e preço de um produto.",
            inputSchema={
                "type": "object",
                "properties": {
                    "produto": {
                        "type": "string", 
                        "description": "Nome do produto (ex: 'Monitor', 'Teclado', 'Mouse')"
                    }
                },
                "required": ["produto"]
            }
        )
    ]

# 3. Mapeamos a lógica real que será executada quando o Cliente pedir
@app.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    if name != "consultar_estoque":
        raise ValueError(f"Ferramenta desconhecida: {name}")
        
    produto = arguments.get("produto", "").lower()
    
    # Simulação de um banco de dados local (SQL, Pandas, etc)
    banco_de_dados = {
        "monitor": {"qtd": 45, "preco": "R$ 800,00"},
        "teclado": {"qtd": 12, "preco": "R$ 150,00"},
        "mouse": {"qtd": 0, "preco": "R$ 50,00"} # Sem estoque
    }
    
    dados = banco_de_dados.get(produto)
    if dados:
        resposta = f"Produto '{produto}' -> Em Estoque: {dados['qtd']} unidades | Preço BD: {dados['preco']}."
    else:
        resposta = f"Produto '{produto}' não cadastrado no sistema da empresa."
        
    return [TextContent(type="text", text=resposta)]

# 4. Inicializa o servidor atrelado as variáveis de entrada/saída (Stdio) padrão
async def main():
    from mcp.server.stdio import stdio_server
    
    # O stdio_server captura o stdin/stdout, que é por onde o ClientSession fala
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
