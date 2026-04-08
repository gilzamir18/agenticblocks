import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import json
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks.blocks.llm.agent import AgentInput
import asyncio
from pydantic import BaseModel
from agenticblocks.core.block import Block
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor

class EmailCheckerInput(BaseModel):
    from_content: str
    to_content: str
    subject_content: str
    email_content: str
    
class EmailCheckerOutput(BaseModel):
    content_type: str
    actions: str

class EmailCheckerAgenticBlock(Block[EmailCheckerInput, EmailCheckerOutput]):
    name: str = "email_checker"
    llmagent: LLMAgentBlock

    async def run(self, input: EmailCheckerInput) -> EmailCheckerOutput:
        prompt = f"""
        você é o Orquestrador Principal do fluxo de verificação de email.
        Aqui estão os detalhes do email:
            - De: {input.from_content}
            - Para: {input.to_content}
            - Assunto: {input.subject_content}
            - Corpo Oculto: {input.email_content}
        Classifique o tipo de conteúdo ("spam", "suporte", "vendas") e forneça
        quais ações devem ser efetuadas.

        Sua resposta FINAL deve ser extritamente um JSON
        no seguinte formato:
        {{
            "content_type": "",
            "actions": ""
        }}
        """

        agent_output = await self.llmagent.run(input=AgentInput(prompt=prompt))

        try:
            parsed_result = json.loads(agent_output.response.strip("```json\n").strip("```"))
        except json.JSONDecodeError:
            parsed_result = {"content_type": "desconhecido", "actions": agent_output.response}
            
        # Modelos menores (como Granite) podem retornar listas de ações de forma inesperada.
        actions_raw = parsed_result.get("actions", "")
        if isinstance(actions_raw, list):
            actions_str = " | ".join(str(x) for x in actions_raw)
        else:
            actions_str = str(actions_raw)

        return EmailCheckerOutput(
            content_type = str(parsed_result.get("content_type", "desconhecido")),
            actions = actions_str
        )
    

async def main():

    graph = WorkflowGraph()

    llmmodelagent = LLMAgentBlock(
        name="LLM Model Agent",
        model="ollama/granite4:1b",
        system_prompt="Você é um assistente de email, classifique o tipo de conteúdo e as ações necessárias."
    )

    block_email = EmailCheckerAgenticBlock(llmagent=llmmodelagent)
    graph.add_block(block_email)

    executor = WorkflowExecutor(graph)

    print("Iniciando Workflow...\n")
    
    if llmmodelagent.model.startswith("gemini") and not os.getenv("GEMINI_API_KEY"):
         print("⚠️ Lembrete: Defina GEMINI_API_KEY para a resposta final fluir do LiteLLM.")
    elif llmmodelagent.model.startswith("openai") and not os.getenv("OPENAI_API_KEY"):
         print("⚠️ Lembrete: Defina OPENAI_API_KEY para a resposta final fluir do LiteLLM.")
    try:
        # 3. Disparamos o Loop de Raciocínio
        print("👤 Simulando o recebimento de um email...")
        email_data = {
            "from_content": "cliente@email.com",
            "to_content": "suporte@empresa.com",
            "subject_content": "Problema no acesso à conta",
            "email_content": "Não consigo entrar no sistema há horas. Me ajude rápido!"
        }
        
        ctx = await executor.run(initial_input=email_data)
        
        output = ctx.get_output("email_checker")
        print("\n[🎯 Resposta Final do Agente]:")
        print(f"Tipo: {output.content_type}\nAções: {output.actions}")


    except Exception as e:
        print(f"\n[🛑 Falha no provedor LLM]: {e}")

if __name__ == "__main__":
    asyncio.run(main())