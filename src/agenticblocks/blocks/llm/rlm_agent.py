import time
from pydantic import Field
from typing import Dict, Any, Optional

from agenticblocks.core.agent import AgentBlock
from agenticblocks.blocks.llm.agent import AgentInput, AgentOutput

try:
    from rlm import RLM
except ImportError:
    RLM = None

class RLMAgentBlock(AgentBlock[AgentInput, AgentOutput]):
    description: str = "Autonomous LLM-based Agent using Recursive Language Models (RLM)."
    
    backend: str = "openai"
    backend_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"model_name": "gpt-4o-mini"})
    
    environment: str = "local"
    environment_kwargs: Dict[str, Any] = Field(default_factory=dict)
    
    verbose: bool = False
    
    model_config = {"arbitrary_types_allowed": True}
    
    async def run(self, input: AgentInput) -> AgentOutput:
        if RLM is None:
            raise ImportError(
                "The 'rlms' package is required to use RLMAgentBlock. "
                "Please install it using 'pip install rlms' or by updating your dependencies."
            )
            
        start_time = time.monotonic()
        
        # Initialize the RLM instance
        rlm_instance = RLM(
            backend=self.backend,
            backend_kwargs=self.backend_kwargs,
            environment=self.environment,
            environment_kwargs=self.environment_kwargs,
            verbose=self.verbose
        )
        
        # Call completion
        # Note: rlm.completion is synchronous in the basic usage based on the README
        result = rlm_instance.completion(input.prompt)
        
        # Calculate iterations or tool calls if available in metadata
        # The exact metadata structure depends on rlm's logger or internal state.
        # We'll set a default of 0 for tool_calls_made if not easily extractable.
        tool_calls_made = 0
        if hasattr(result, "metadata") and result.metadata:
            # You could inspect result.metadata to count sub-calls if needed
            pass
            
        return AgentOutput(
            response=result.response,
            tool_calls_made=tool_calls_made
        )
