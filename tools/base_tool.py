from abc import ABC, abstractmethod
from harness_core.types import ToolResult


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    required_permission: str = "READ"

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass

    def to_description(self) -> dict:
        return {
            "name"       : self.name,
            "description": self.description,
            "permission" : self.required_permission
        }
