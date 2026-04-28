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

    code_planner = CodePlanExecutorBlock(
        executor_agent=agent,
        execution_mode="local",
        max_retries=2
    )

    print("Iniciando o Code Planner em modo LOCAL...")
    result = await code_planner.run(CodePlanExecutorInput(
        task="Escreva um script para calcular os primeiros 10 números da sequência de Fibonacci e imprimí-los como uma lista."
    ))

    print("-" * 50)
    print(f"Sucesso: {result.success}")
    print(f"\nCódigo Gerado pelo LLM:\n{result.code_generated}")
    print(f"\nStdout da Execução:\n{result.execution_stdout.strip()}")
    if result.execution_stderr:
        print(f"\nStderr da Execução:\n{result.execution_stderr.strip()}")
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
