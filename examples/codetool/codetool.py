import asyncio
import os
from dotenv import load_dotenv
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks.tools.e2b_interpreter import E2BInterpreterBlock

# 1. Carregamos as chaves de API do arquivo .env
# É necessário ter OPENAI_API_KEY e E2B_API_KEY configuradas.
load_dotenv()

async def run_didactic_example():
    """
    Este exemplo demonstra como dar ao seu Agente a capacidade de 'pensar e agir' 
    através da execução de código Python em um ambiente sandbox seguro (E2B).
    """

    print("--- Agente de Análise com Code Interpreter ---")

    # 2. Instanciamos o Bloco E2B
    # Este bloco servirá como a 'mão' do agente para executar código.
    code_interpreter = E2BInterpreterBlock()

    # 3. Criamos o Agente e injetamos o Sandbox como uma 'Tool'
    # O AgentBlock irá converter automaticamente o E2BInterpreterBlock 
    # em uma definição de ferramenta que o LLM entende.
    analyst = LLMAgentBlock(
        name="DataChartAgent",
        description="Um agente que usa Python para criar visualizações e cálculos.",
        system_prompt=(
            "Você é um assistente especializado em visualização de dados. "
            "Sempre utilize o 'python_interpreter' para gerar gráficos ou realizar cálculos. "
            "Se gerar um gráfico, avise o usuário que o arquivo de imagem foi processado."
        ),
        tools=[code_interpreter],
        model="gpt-4o-mini"
    )

    # 4. Definimos uma tarefa que exige execução de código e geração de imagem
    task = "Crie um gráfico de barras comparando a população de Brasil, China e EUA."

    print(f"Tarefa: {task}")
    print("Processando... (O agente irá gerar e executar o código Python no E2B)")

    # 5. Executamos o agente
    result = await analyst.run(prompt=task)

    # 6. Exibimos o resultado final
    print("\n--- Resposta da Inteligência Artificial ---")
    print(result.response)
    
    print(f"\nNúmero de chamadas de ferramentas: {result.tool_calls_made}")
    
    # Nota Didática: 
    # O E2BInterpreterBlock retorna logs e imagens base64. 
    # Em uma aplicação real, você poderia salvar o 'result.images_base64' em arquivos .png.
    # Por agora, o agente apenas nos confirma que a tarefa foi feita.

if __name__ == "__main__":
    if not os.getenv("E2B_API_KEY"):
        print("ERRO: E2B_API_KEY não encontrada. Instale o SDK e obtenha uma chave em e2b.dev")
    else:
        asyncio.run(run_didactic_example())
