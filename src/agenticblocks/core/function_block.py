"""
function_block.py — Adaptador que permite usar funções simples como Blocos.

Uso:
    @as_tool
    async def buscar_clima(cidade: str) -> str:
        '''Retorna o clima atual de uma cidade.'''
        return f"Ensolarado em {cidade}"

    @as_tool(name="clima", description="Consulta o clima atual.")
    def buscar_clima_sync(cidade: str) -> str:
        return f"Ensolarado em {cidade}"

    agente = LLMAgentBlock(name="agente", tools=[buscar_clima, buscar_clima_sync])
"""

import asyncio
import inspect
from typing import Any, Callable, Optional, Type, get_type_hints

from pydantic import BaseModel, PrivateAttr, create_model

from agenticblocks.core.block import Block


class FunctionOutput(BaseModel):
    """Output padrão para funções que retornam valores primitivos ou dicts."""

    model_config = {"arbitrary_types_allowed": True}
    result: Any


def _build_input_model(func: Callable) -> Type[BaseModel]:
    """Constrói um modelo Pydantic dinamicamente a partir dos parâmetros da função."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        annotation = hints.get(param_name, Any)
        if param.default is inspect.Parameter.empty:
            fields[param_name] = (annotation, ...)
        else:
            fields[param_name] = (annotation, param.default)

    model_name = f"{''.join(w.title() for w in func.__name__.split('_'))}Input"
    return create_model(model_name, **fields)  # type: ignore[call-overload]


class FunctionBlock(Block):
    """
    Adapta uma função Python (sync ou async) para a interface Block.

    Suporta:
    - async def: chamada direta com await.
    - def: execução em thread pool via asyncio.to_thread (não bloqueia o event loop).

    O retorno da função é normalizado:
    - BaseModel → retornado diretamente.
    - dict     → serializado como FunctionOutput(result=<dict>).
    - qualquer outro → envolvido em FunctionOutput(result=<valor>).
    """

    model_config = {"arbitrary_types_allowed": True}

    # Atributos privados — não expostos ao schema Pydantic nem serializados.
    _func: Callable = PrivateAttr()
    _input_model: Type[BaseModel] = PrivateAttr()

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **data: Any,
    ) -> None:
        func_name = name or func.__name__
        func_desc = description or inspect.getdoc(func) or f"Executes {func_name}"
        super().__init__(name=func_name, description=func_desc, **data)
        self._func = func
        self._input_model = _build_input_model(func)

    # Sobrescrita de instância: retorna o modelo gerado especificamente para esta função.
    # Como input_schema é classmethod em Block, chamamos via instância — Python resolve
    # o método de instância primeiro quando disponível no __dict__ da classe concreta.
    def input_schema(self) -> Type[BaseModel]:  # type: ignore[override]
        return self._input_model

    async def run(self, input: BaseModel) -> BaseModel:  # type: ignore[override]
        kwargs = input.model_dump()

        if asyncio.iscoroutinefunction(self._func):
            raw = await self._func(**kwargs)
        else:
            # Funções síncronas rodam em thread pool para não bloquear o event loop.
            raw = await asyncio.to_thread(self._func, **kwargs)

        # Normalização do retorno
        if isinstance(raw, BaseModel):
            return raw
        return FunctionOutput(result=raw)


# ---------------------------------------------------------------------------
# Decorator @as_tool
# ---------------------------------------------------------------------------

def as_tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Any:
    """
    Converte uma função (sync ou async) em um FunctionBlock pronto para ser
    usado como ferramenta em LLMAgentBlock.tools — sem alterar a interface Block.

    Pode ser usado de duas formas:

        @as_tool
        async def minha_ferramenta(param: str) -> str: ...

        @as_tool(name="nome", description="Descrição.")
        def minha_ferramenta(param: str) -> str: ...
    """
    def _wrap(f: Callable) -> FunctionBlock:
        return FunctionBlock(func=f, name=name, description=description)

    if func is not None:
        # Chamado sem parênteses: @as_tool
        return _wrap(func)

    # Chamado com parênteses: @as_tool(...)
    return _wrap
