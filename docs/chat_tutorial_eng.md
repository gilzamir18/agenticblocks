# Tutorial: Creating a Conversational Agent with a Loop in AgenticBlocks

This tutorial demonstrates how to build a chatbot operating in a continuous terminal loop. Using the graph structure of `agenticblocks`, we'll connect a language model (LLM) to input and output tools, creating an execution cycle that is only interrupted by a specific user command.

## Prerequisites

Make sure to import the essential modules for handling asynchronous operations, graph management, and tool definition:

```python
import asyncio
import os
import sys
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput, AgentOutput
from agenticblocks import as_tool
```

---

## Step 1: Observability and Prompt Handling

To better understand how the agent is "thinking", it's useful to extend the base agent class to print the exact prompt being sent to the model on each iteration.

Additionally, we need a function to dynamically build and update the prompt context on every chat cycle.

```python
# Updates the prompt on each iteration, focusing on the user's latest message
def build_chat_prompt(orig, iteration, producer, feedback):
    lines = feedback.splitlines()
    last_user = next((l for l in reversed(lines) if l.startswith("User:")), "")
    return (
        f"{feedback}\n\n"
        f"[Focus on answering: {last_user}]"
    )

# Custom agent that prints the prompt to the terminal before executing
class ObservableLLMAgent(LLMAgentBlock):
    async def run(self, input: AgentInput) -> AgentOutput:
        print("-" * 94)
        print(f"[{self.name}] Prompt received: {input.prompt}")
        print("-" * 94)
        return await super().run(input)
```

---

## Step 2: Defining the Tools

`agenticblocks` allows you to turn regular Python functions into tools that the graph or agent can use, via the `@as_tool` decorator. We'll define three main tools: state control, data input, and data output.

```python
chat_history = []

# 1. Tool to check whether the chat should be terminated
@as_tool(name="check_done")
def check_done(last_message: str) -> dict:
    # If the user types /bye, the loop is considered complete (is_valid=True)
    if last_message and "/bye" in last_message:
        return {"is_valid": True, "feedback": ""}
    
    # Otherwise, continue the loop by returning the updated history
    hist = "\n".join(chat_history)
    return {"is_valid": False, "feedback": f"history: {hist}"}

# 2. Tool to capture user input from the terminal
@as_tool(name="get_user_input")
def get_user_input() -> dict:
    print("You: ", end="")
    user_input = input()
    chat_history.append(f"User: {user_input}")
    return {"prompt": user_input}

# 3. Mandatory tool for the agent to print and save its response
@as_tool(name="print_researcher_response", description="Prints the researcher's response and adds it to the chat history")
def print_researcher_response(response: str):
    print(f"Researcher: {response}")
    chat_history.append(f"Researcher: {response}")
```

---

## Step 3: Building the Graph and Configuring the Agent

Now, inside our main function, we initialize the graph and configure the central intelligence node: the `ObservableLLMAgent`.

In this setup, we explicitly instruct the model (in this case, `ollama/mistral-nemo:latest`) to take on the persona of a research assistant and force the use of the print tool using `litellm_kwargs` parameters.

```python
async def main():
    # Initialize the workflow graph
    graph = WorkflowGraph()

    # Instantiate the agent block
    agent_block = ObservableLLMAgent(
        name="research_agent",
        model="ollama/mistral-nemo:latest",
        description="Research agent for answering user questions",
        system_prompt="""You are a research assistant specialized in helping researchers
        on a wide variety of topics. Use your intrinsic knowledge to answer the 
        user's questions. You will speak first — greet them and ask:
        What would my divinity like to know?""",
        tools=[print_researcher_response],
        max_iterations=1,
        # Forces the LLM to always use the tool to respond
        litellm_kwargs={
            "temperature": 0.7, 
            "tool_choice": {"type": "function", "function": {"name": "print_researcher_response"}}
        }
    )

    # Add all constructed blocks to the graph
    graph.add_block(agent_block)
    graph.add_block(get_user_input)
    graph.add_block(check_done)
```

---

## Step 4: Creating the Execution Cycle

The magic of a continuous conversation happens by connecting the blocks in a cycle. The defined sequence dictates the execution order, and the stop condition controls when the cycle should be broken.

```python
    # Define the iterative conversation cycle
    graph.add_cycle(
        name="chat_loop",
        sequence=["research_agent", "get_user_input", "check_done"],
        condition_block="check_done", # The block that determines whether the loop ends
        max_iterations=1000,          # Safety limit
        augment_fn=build_chat_prompt  # Function that formats the context on each turn
    )
```
*The execution sequence will be:*
1. The agent (`research_agent`) processes the prompt and responds using the print tool.
2. The system waits for user input (`get_user_input`).
3. The system checks whether the stop condition has been met (`check_done`). If not, `augment_fn` updates the prompt and the cycle restarts.

---

## Step 5: Running the Workflow

Finally, we create the executor and trigger the workflow with an initial input.

```python
    # Create the executor with logging enabled
    executor = WorkflowExecutor(graph, verbose=True)

    # Trigger execution by providing the initial state
    ctx = await executor.run(
        initial_input={"prompt": "You are in a continuous conversation. Respond to the most recent message in the history."}
    )
    
    # Extract and display the final result of the chat cycle
    cr = ctx.cycle_results.get("chat_loop")
    if cr:
        print("Chat conversation ", cr.output)

# Standard asyncio entry point
if __name__ == "__main__":
    asyncio.run(main())
```

### Summary of Expected Behavior:
When running the script, the terminal will display logs indicating the start of the workflow. The agent will analyze the *system prompt* and deliver its opening greeting via the `print_researcher_response` function. The flow will then pause waiting for your input via `input()`. The loop will remain active, feeding the intelligence with the accumulated history until the `/bye` command is detected, at which point the graph will terminate execution successfully.