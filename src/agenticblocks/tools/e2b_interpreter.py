import os
import base64
from typing import List, Any, Optional
from pydantic import BaseModel, Field
from agenticblocks.core.block import Block

# Tenta importar o SDK do E2B, mas não crasha o sistema se estiver faltando
try:
    from e2b_code_interpreter import Sandbox
    E2B_AVAILABLE = True
except ImportError:
    E2B_AVAILABLE = False

class PythonInterpreterInput(BaseModel):
    code: str = Field(..., description="O código Python a ser executado no sandbox.")

class PythonInterpreterOutput(BaseModel):
    stdout: List[str] = Field(default_factory=list)
    stderr: List[str] = Field(default_factory=list)
    results: List[Any] = Field(default_factory=list)
    images_base64: List[str] = Field(default_factory=list, description="Lista de imagens geradas codificadas em base64.")
    error: Optional[str] = None

class E2BInterpreterBlock(Block[PythonInterpreterInput, PythonInterpreterOutput]):
    """
    Bloco que executa código Python em um ambiente sandbox seguro fornecido pelo E2B.
    Ideal para análise de dados, geração de gráficos e cálculos complexos.
    """
    name: str = "python_interpreter"
    description: str = "Executa código Python em um sandbox seguro. Útil para processamento de dados e gráficos."
    api_key: Optional[str] = Field(default=None, exclude=True)

    async def run(self, input: PythonInterpreterInput) -> PythonInterpreterOutput:
        if not E2B_AVAILABLE:
            return PythonInterpreterOutput(error="SDK 'e2b-code-interpreter' não instalado. Execute 'pip install e2b-code-interpreter'.")

        api_key = self.api_key or os.getenv("E2B_API_KEY")
        if not api_key:
            return PythonInterpreterOutput(error="E2B_API_KEY não encontrada nas variáveis de ambiente ou no bloco.")

        try:
            # Cria um sandbox efêmero para a execução
            # Nota: Em versões mais complexas, poderíamos manter a sessão aberta
            async with Sandbox(api_key=api_key) as sandbox:
                execution = await sandbox.run_code(input.code)
                
                results = []
                images = []
                
                # Processa os resultados da execução
                for res in execution.results:
                    # Se houver formatos visuais (ex: png, svg), armazenamos como base64
                    if hasattr(res, "png") and res.png:
                        images.append(res.png)
                    elif hasattr(res, "svg") and res.svg:
                        # Opcional: converter SVG ou manter como string? E2B costuma dar o raw.
                        pass
                    
                    # Tenta converter o resultado para algo serializável ou string
                    results.append(str(res))

                return PythonInterpreterOutput(
                    stdout=execution.logs.stdout,
                    stderr=execution.logs.stderr,
                    results=results,
                    images_base64=images,
                    error=execution.error.value if execution.error else None
                )

        except Exception as e:
            return PythonInterpreterOutput(error=f"Falha na execução do E2B: {str(e)}")
