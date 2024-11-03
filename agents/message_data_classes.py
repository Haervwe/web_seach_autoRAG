from typing import List
from dataclasses import dataclass

@dataclass
class ToolResultsMessage:
    content: List
    source: str

@dataclass
class Message:
    content: str
    source: str
