from typing import Any, Dict, List, Optional, Callable, Tuple
from pydantic import BaseModel
from agenticblocks.core.block import Block
from agenticblocks.blocks.patterns.code_executor import PythonCodeExecutorBlock, PythonCodeExecutorInput

class CodePlanExecutorInput(BaseModel):
    task: str
    history: str = ""

class CodePlanExecutorOutput(BaseModel):
    response: str
    code_generated: str
    execution_stdout: str
    execution_stderr: str
    success: bool

class CodePlanExecutorBlock(Block[CodePlanExecutorInput, CodePlanExecutorOutput]):
    name: str = "code_plan_executor"
    description: str = "Generates a Python script to solve a task and executes it (optionally in Docker)."
    executor_agent: Block
    execution_mode: str = "local" # 'local' or 'docker'
    docker_image: str = "python:3.10-slim"
    max_retries: int = 2
    prompt_template: str = (
        "OBJETIVO DO PLANEJADOR:\n{task}\n\n"
        "HISTÓRICO RECENTE:\n{history}\n\n"
        "Escreva um script Python completo que resolva a tarefa acima.\n"
        "O código deve imprimir (usando print) o resultado final ou dados necessários.\n"
        "{extra_instruction}\n"
    )

    model_config = {"arbitrary_types_allowed": True}

    async def run(self, input: CodePlanExecutorInput) -> CodePlanExecutorOutput:
        extra_instruction = ""
        code_executor = PythonCodeExecutorBlock(
            execution_mode=self.execution_mode,
            docker_image=self.docker_image
        )

        for attempt in range(self.max_retries + 1):
            prompt = self.prompt_template.format(
                task=input.task,
                history=input.history,
                extra_instruction=extra_instruction
            )
            
            # 1. Ask LLM to generate code
            agent_input_model = self.executor_agent.input_schema()
            try:
                agent_input = agent_input_model(prompt=prompt)
            except Exception:
                agent_input = agent_input_model(**{"prompt": prompt})
                
            result = await self.executor_agent.run(agent_input)
            agent_reply = getattr(result, "response", str(result))
            
            # 2. Execute code
            exec_input = PythonCodeExecutorInput(code=agent_reply)
            exec_output = await code_executor.run(exec_input)
            
            if exec_output.is_valid:
                # Success
                return CodePlanExecutorOutput(
                    response="Execução finalizada com sucesso.",
                    code_generated=agent_reply,
                    execution_stdout=exec_output.stdout,
                    execution_stderr=exec_output.stderr,
                    success=True
                )
            else:
                # Failure, retry with error feedback
                extra_instruction = (
                    f"AVISO: O código anterior falhou na execução.\n"
                    f"Código de saída: {exec_output.exit_code}\n"
                    f"Erro/Stderr:\n{exec_output.stderr}\n"
                    f"Stdout:\n{exec_output.stdout}\n\n"
                    "Por favor, corrija o código Python e retorne o novo script completo."
                )

        return CodePlanExecutorOutput(
            response="Falha ao executar o plano após as tentativas.",
            code_generated=agent_reply,
            execution_stdout=exec_output.stdout,
            execution_stderr=exec_output.stderr,
            success=False
        )
