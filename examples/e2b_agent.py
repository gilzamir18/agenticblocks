import asyncio
import os
from dotenv import load_dotenv
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks.tools.e2b_interpreter import E2BInterpreterBlock

# Carrega variáveis de ambiente (.env)
load_dotenv()

async def main():
    # 1. Instanciamos o Bloco de Interpretação (E2B)
    # Certifique-se de que a variável de ambiente E2B_API_KEY está definida.
    interpreter = E2BInterpreterBlock()

    # 2. Criamos o Agente e injetamos o Sandbox como ferramenta
    agent = LLMAgentBlock(
        name="DataAnalystAgent",
        description="Um agente capaz de executar código para resolver problemas matemáticos e de dados.",
        system_prompt="Você é um cientista de dados. Use o 'python_interpreter' para qualquer cálculo complexo ou visualização.",
        tools=[interpreter],
        model="gpt-4o-mini",
        max_tool_calls=3
    )

    # 3. Fazemos uma pergunta que exige execução de código
    pergunta = "Qual é o 15º número da sequência de Fibonacci? Calcule usando Python para ter certeza."
    
    print(f"--- Iniciando Agente ---\nPergunta: {pergunta}\n")
    
    result = await agent.run(prompt=pergunta)
    
    print("\n--- Resposta Final ---")
    print(result.response)
    print(f"\nChamadas de ferramentas feitas: {result.tool_calls_made}")

if __name__ == "__main__":
    if not os.getenv("E2B_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        print("Aviso: Defina E2B_API_KEY e OPENAI_API_KEY no seu ambiente ou arquivo .env")
    
    asyncio.run(main())
