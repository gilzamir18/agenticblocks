import asyncio
from typing import Any, List, Dict
from contextlib import AsyncExitStack
from pydantic import BaseModel, ConfigDict
from agenticblocks.core.block import Block

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    pass

class MCPProxyInput(BaseModel):
    model_config = ConfigDict(extra='allow')
    # Captura quaisquer argumentos criados pelo LLM

class MCPProxyOutput(BaseModel):
    result: Any

class MCPProxyBlock(Block[MCPProxyInput, MCPProxyOutput]):
    is_mcp_proxy: bool = True
    raw_mcp_schema: dict = {}
    session: Any = None
    
    async def run(self, input: MCPProxyInput) -> MCPProxyOutput:
        # Usa dict() normal ou model_dump(exclude_unset=True) para extrair os extras
        # Nota: model_dump pode não espalhar os "extras" na raiz em algumas versoes do Pydantic, 
        # porém __dict__ com `extra='allow'` resolve.
        
        args = {k: getattr(input, k) for k in input.model_fields_set | set(input.model_extra or {})}
        
        # Processa Remote Procedure Call pro Servidor MCP real!
        result = await self.session.call_tool(self.name, args)
        
        # Em MCP o response contém .content (Array de TextContent/ImageContent etc)
        # Convertendo o conteúdo remotamente respondido pelo Servidor.
        text_responses = []
        if result and hasattr(result, "content"):
            for c in result.content:
                if getattr(c, "type", "") == "text":
                    text_responses.append(c.text)
        
        return MCPProxyOutput(result=text_responses)

class MCPClientBridge:
    """
    Abstração transparente para buscar ferramentas de servidores MCP compatíveis.
    Transforma ferramentas de rede em "ProxyBlocks" inseríveis em Agentes.
    """
    def __init__(self, command: str, args: List[str], env: Dict[str, str] = None):
        self.command = command
        self.args = args
        self.env = env
        self.exit_stack = AsyncExitStack()
        self.session = None

    async def connect(self) -> List[MCPProxyBlock]:
        """Estabelece link iterativo stdio com o Servidor MCP e extrai os Tools"""
        server_params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        except Exception as e:
            raise RuntimeError(f"Falha ao rodar sub-processo MCP. Verifique se tem o '{self.command}' instalado. Erro: {e}")
            
        self.read, self.write = stdio_transport
        
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.read, self.write))
        await self.session.initialize()
        
        # Obter tools do servidor remotamente
        response = await self.session.list_tools()
        
        proxy_blocks = []
        for tool in response.tools:
            # Para cada tool nativa do servidor MCP, instanciamos um proxy
            block = MCPProxyBlock(
                name=tool.name,
                description=tool.description or ""
            )
            block.raw_mcp_schema = tool.inputSchema
            block.session = self.session
            proxy_blocks.append(block)
            
        return proxy_blocks
        
    async def disconnect(self):
        """Fecha de forma segura."""
        await self.exit_stack.aclose()
