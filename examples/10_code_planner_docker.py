import asyncio
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks.blocks.patterns.code_plan_executor import CodePlanExecutorBlock, CodePlanExecutorInput

async def main():
    agent = LLMAgentBlock(
        name="code_generator",
        model="ollama/mistral-nemo:latest",
        system_prompt=(
            "Você é um excelente programador Python.\n"
            "Escreva APENAS o código Python necessário para resolver o problema, nada mais.\n"
            "Use o bloco markdown ```python ... ``` para colocar o código.\n"
            "O resultado final DEVE ser impresso na tela usando print()."
        )
    )

    # Nota: É necessário ter o Docker rodando na máquina para este exemplo funcionar.
    code_planner = CodePlanExecutorBlock(
        executor_agent=agent,
        execution_mode="docker",
        docker_image="python:3.10-slim",
        max_retries=2
    )

    print("Iniciando o Code Planner em modo DOCKER...")
    print("O script gerado rodará dentro de um contêiner (python:3.10-slim).")
    
    result = await code_planner.run(CodePlanExecutorInput(
        task="Escreva um script que leia a versão do Python em execução e informações sobre o sistema operacional (uname ou sys.platform) e imprima o resultado."
    ))

    print("-" * 50)
    print(f"Sucesso: {result.success}")
    print(f"\nCódigo Gerado pelo LLM:\n{result.code_generated}")
    print(f"\nStdout da Execução (via Docker):\n{result.execution_stdout.strip()}")
    if result.execution_stderr:
        print(f"\nStderr da Execução:\n{result.execution_stderr.strip()}")
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
