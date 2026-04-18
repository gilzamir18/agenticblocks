import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_model

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
        You are the Main Orchestrator of the email verification flow.
        Here are the email details:
            - From: {input.from_content}
            - To: {input.to_content}
            - Subject: {input.subject_content}
            - Body: {input.email_content}
        Classify the content type ("spam", "support", "sales") and specify
        which actions should be taken.

        Your FINAL response must be strictly a JSON object
        in the following format:
        {{
            "content_type": "",
            "actions": ""
        }}
        """

        agent_output = await self.llmagent.run(input=AgentInput(prompt=prompt))

        try:
            parsed_result = json.loads(agent_output.response.strip("```json\n").strip("```"))
        except json.JSONDecodeError:
            parsed_result = {"content_type": "unknown", "actions": agent_output.response}

        # Smaller models (e.g. Granite) may return actions as a list unexpectedly.
        actions_raw = parsed_result.get("actions", "")
        if isinstance(actions_raw, list):
            actions_str = " | ".join(str(x) for x in actions_raw)
        else:
            actions_str = str(actions_raw)

        return EmailCheckerOutput(
            content_type=str(parsed_result.get("content_type", "unknown")),
            actions=actions_str,
        )
    

async def main():

    graph = WorkflowGraph()

    llm_agent = LLMAgentBlock(
        name="LLM Model Agent",
        model=get_model(),
        system_prompt="You are an email assistant. Classify the content type and specify the necessary actions."
    )

    block_email = EmailCheckerAgenticBlock(llmagent=llm_agent)
    graph.add_block(block_email)

    executor = WorkflowExecutor(graph)

    print("Starting workflow...\n")

    if llm_agent.model.startswith("gemini") and not os.getenv("GEMINI_API_KEY"):
        print("⚠️  Reminder: set GEMINI_API_KEY so LiteLLM can reach the Gemini API.")
    elif llm_agent.model.startswith("openai") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Reminder: set OPENAI_API_KEY so LiteLLM can reach the OpenAI API.")
    try:
        # Trigger the reasoning loop
        print("👤 Simulating the reception of an email...")
        email_data = {
            "from_content": "client@email.com",
            "to_content": "support@company.com",
            "subject_content": "Problem accessing my account",
            "email_content": "I haven't been able to log into the system for hours. Please help me quickly!"
        }

        ctx = await executor.run(initial_input=email_data)

        output = ctx.get_output("email_checker")
        print("\n[🎯 Agent Final Response]:")
        print(f"Type: {output.content_type}\nActions: {output.actions}")

    except Exception as e:
        print(f"\n[🛑 LLM provider failure]: {e}")

if __name__ == "__main__":
    asyncio.run(main())