import asyncio
from typing import Optional

from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.core.function_block import as_tool
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock

# Simulate getting user input
@as_tool(name="get_user_input")
def get_user_input(prompt: str) -> dict:
    print(f"\n[USER INPUT PROMPT]: {prompt}")
    # In a real app, this would wait for user text.
    # We will simulate that the user wants to buy a "blue shirt".
    return {"text": "I want to buy a blue shirt."}

# Simulate intention parsing
@as_tool(name="intention_agent")
def intention_agent(text: str) -> dict:
    print(f"\n[INTENTION AGENT]: Parsing '{text}'")
    # Simulate extraction
    if "blue shirt" in text.lower():
        extracted = "blue_shirt"
    else:
        extracted = "unknown"
    return {"intention": extracted}

# Simulate intention validator
@as_tool(name="check_intention")
def check_intention(intention: str) -> dict:
    print(f"\n[CHECK INTENTION]: Checking '{intention}'")
    if intention != "unknown":
        return {"is_valid": True, "feedback": "", "intention": intention}
    else:
        return {"is_valid": False, "feedback": "Could not extract a valid intention."}

# Simulate sales agent
@as_tool(name="sales_agent")
def sales_agent(intention: str) -> dict:
    print(f"\n[SALES AGENT]: Processing sale for '{intention}'")
    # Simulate a successful sale but with missing address
    return {"status": "needs_address"}

# Simulate sales validator
@as_tool(name="check_done")
def check_done(status: str) -> dict:
    print(f"\n[CHECK DONE]: Checking status '{status}'")
    if status == "success":
        return {"is_valid": True, "feedback": ""}
    else:
        return {"is_valid": False, "feedback": "Please provide your address."}

async def main():
    graph = WorkflowGraph()

    # Add blocks
    graph.add_block(get_user_input)
    graph.add_block(intention_agent)
    graph.add_block(check_intention)
    graph.add_block(sales_agent)
    graph.add_block(check_done)

    # Inner cycle: keep asking until we get a valid intention
    graph.add_cycle(
        name="intention_loop",
        sequence=["get_user_input", "intention_agent", "check_intention"],
        condition_block="check_intention",
        max_iterations=3,
    )

    # Outer cycle: execute the intention loop, then process sale, repeat if sale fails
    graph.add_cycle(
        name="refine_loop",
        sequence=["intention_loop", "sales_agent", "check_done"],
        condition_block="check_done",
        max_iterations=2,
    )

    # Run the graph starting at the outer loop
    executor = WorkflowExecutor(graph, verbose=True)
    
    print("=== STARTING WORKFLOW ===")
    ctx = await executor.run(initial_input={"prompt": "Hello, how can I help you today?"})
    
    print("\n=== FINAL RESULTS ===")
    print("Intention Loop Result:", ctx.get_output("intention_loop"))
    print("Refine Loop Result:", ctx.get_output("refine_loop"))

if __name__ == "__main__":
    asyncio.run(main())
