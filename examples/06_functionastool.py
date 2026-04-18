import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import get_model

from agenticblocks import as_tool
from agenticblocks.blocks.llm.agent import LLMAgentBlock

# async def — called directly with await
@as_tool
async def get_weather(city: str) -> str:
    """Returns the current weather for a city."""
    return f"Sunny in {city}"

# synchronous def — runs in a thread pool via asyncio.to_thread
@as_tool(name="current_time", description="Returns the current system time.")
def current_time() -> str:
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")

# Usage is identical to any other Block
agent = LLMAgentBlock(
    name="assistant",
    model=get_model(),
    tools=[get_weather, current_time],
)
