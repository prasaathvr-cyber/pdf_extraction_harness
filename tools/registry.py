from typing import Dict
from tools.base_tool import BaseTool
from harness_core.types import ToolResult
from permissions.permission_checker import PermissionChecker


class ToolRegistry:
    """Component #3 - Tools & Skills Registry"""

    def __init__(self, permission_checker: PermissionChecker = None):
        self._tools: Dict[str, BaseTool] = {}
        self._permissions = permission_checker or PermissionChecker()

    def register(self, tool: BaseTool):
        # Handle both old tools (without metadata) and new tools (with metadata)
        if hasattr(tool, 'metadata') and callable(getattr(tool, 'metadata')):
            try:
                metadata = tool.metadata()
                name = metadata['name']
            except (AttributeError, KeyError, TypeError):
                name = tool.__class__.__name__
        else:
            name = getattr(tool, 'name', tool.__class__.__name__)
        
        self._tools[name] = tool
        print(f"  [Registry] Registered tool: '{name}'")

    def get(self, tool_name: str) -> BaseTool:
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' not found in registry. "
                           f"Available: {list(self._tools.keys())}")
        return self._tools[tool_name]

    def execute(self, tool_name: str, params: dict) -> ToolResult:
        # Component #9: Check permissions before dispatch
        self._permissions.check_and_raise(tool_name)
        tool = self.get(tool_name)
        return tool.execute(**params)

    def list_descriptions(self) -> list:
        return [t.to_description() for t in self._tools.values()]

    def list_names(self) -> list:
        return list(self._tools.keys())