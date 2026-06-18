from config.settings import PERMISSION_LEVELS, TOOL_PERMISSIONS, DANGEROUS_BASH_COMMANDS


class PermissionChecker:
    """Component #9 - Permissions & Safety Layer"""

    def __init__(self, user_level: str = 'WORKSPACE'):
        self.user_level = user_level

    def can_execute(self, tool_name: str) -> bool:
        required = TOOL_PERMISSIONS.get(tool_name, 'READ')
        return PERMISSION_LEVELS[self.user_level] >= PERMISSION_LEVELS[required]

    def classify_bash(self, command: str) -> str:
        cmd = command.strip().lower()
        for dangerous in DANGEROUS_BASH_COMMANDS:
            if cmd.startswith(dangerous):
                return 'FULL'
        return 'READ'

    def check_and_raise(self, tool_name: str):
        if not self.can_execute(tool_name):
            required = TOOL_PERMISSIONS.get(tool_name, 'READ')
            raise PermissionError(
                f"Tool '{tool_name}' requires '{required}' permission. "
                f"Current level: '{self.user_level}'"
            )
