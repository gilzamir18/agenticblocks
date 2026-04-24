import asyncio
import os
from agenticblocks.blocks.llm.rlm_agent import RLMAgentBlock
from agenticblocks.blocks.llm.agent import AgentInput

async def main():
    # Make sure you have your OpenAI API key set if you use the openai backend
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set. RLM may fail if using openai backend.")
    
    print("Initializing RLMAgentBlock...")
    # Note: Ensure rlms is installed: pip install rlms
    rlm_agent = RLMAgentBlock(
        name="recursive_math_agent",
        backend="litellm",
        backend_kwargs={"model_name": "ollama/mistral-nemo:latest"},
        environment="local",
        verbose=True
    )
    
    prompt = "Print me the first 5 powers of two, each on a newline."
    print(f"Running agent with prompt: {prompt}")
    
    input_data = AgentInput(prompt=prompt)
    output = await rlm_agent.run(input=input_data)
    
    print("\n--- Final Output ---")
    print(f"Tool calls made: {output.tool_calls_made}")
    print(output.response)

if __name__ == "__main__":
    asyncio.run(main())
