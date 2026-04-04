from pydantic import BaseModel
from typing import TypeVar, Generic, get_args

Input = TypeVar("Input", bound=BaseModel)
Output = TypeVar("Output", bound=BaseModel)

class Block(BaseModel, Generic[Input, Output]):
    name: str

class FetchInput(BaseModel): pass
class FetchOutput(BaseModel): pass

class FetchBlock(Block[FetchInput, FetchOutput]):
    pass

print("orig_bases:", getattr(FetchBlock, "__orig_bases__", None))
try:
    print("pydantic_generic:", getattr(FetchBlock, "__pydantic_generic_metadata__", None))
except:
    pass
