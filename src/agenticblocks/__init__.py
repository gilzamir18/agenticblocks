"""agenticblocks: A composable building block library for AI workflows."""
__version__ = "0.1.0"

from agenticblocks.core.function_block import FunctionBlock, as_tool
from agenticblocks.blocks.flow.prompt_builder import PromptBuilderBlock

__all__ = ["FunctionBlock", "as_tool", "PromptBuilderBlock"]
