from typing import Callable, Dict, List


class HookRegistry:
    """Component #8 - Lifecycle Hooks (pre/post tool execution)"""

    def __init__(self):
        self._pre_hooks:  Dict[str, List[Callable]] = {}
        self._post_hooks: Dict[str, List[Callable]] = {}

    def register_pre_hook(self, tool_name: str, func: Callable):
        self._pre_hooks.setdefault(tool_name, []).append(func)

    def register_post_hook(self, tool_name: str, func: Callable):
        self._post_hooks.setdefault(tool_name, []).append(func)

    def trigger_pre(self, tool_name: str, params: dict) -> dict:
        """Runs before tool execution. Hooks can modify params."""
        for hook in self._pre_hooks.get(tool_name, []):
            result = hook(tool_name, params)
            if result is not None:
                params = result
        return params

    def trigger_post(self, tool_name: str, result: dict):
        """Runs after tool execution. For logging/auditing."""
        for hook in self._post_hooks.get(tool_name, []):
            hook(tool_name, result)


# ── Built-in hooks (examples) ─────────────────────────────────

def log_pre_hook(tool_name: str, params: dict) -> dict:
    """Logs every tool call before execution"""
    print(f"  [PRE-HOOK]  Tool='{tool_name}' | Page={params.get('page', '?')}")
    return params


def log_post_hook(tool_name: str, result: dict):
    """Logs every tool result after execution"""
    status = result.get('status', 'unknown')
    page = result.get('page', '?')
    tables = len(result.get('tables', []))
    figures = len(result.get('figures', []))
    print(f"  [POST-HOOK] Tool='{tool_name}' | Page {page} | Status='{status}' | Tables={tables} | Figures={figures}")


def zero_table_detection_hook(tool_name: str, result: dict):
    """
    Detects when a page was extracted but zero tables were found.
    Flags this for QA review (stage1's approach).
    """
    if tool_name == 'vision_extract_page':
        tables = result.get('tables', [])
        figures = result.get('figures', [])
        page = result.get('page', '?')
        
        # Flag pages that have neither tables nor figures for manual review
        if not tables and not figures:
            print(f"  [ZERO-CONTENT] Page {page} has no tables or figures — may need manual review")


def table_header_validation_hook(tool_name: str, result: dict):
    """
    Post-extraction validation: checks if any tables have bad headers.
    This is mostly handled by VisionPageTool's header repair, but this hook
    can flag edge cases for logging.
    """
    if tool_name == 'vision_extract_page':
        for table in result.get('tables', []):
            headers = table.get('headers', [])
            if not headers or all(not h or str(h).lower() in ('nan', 'none', '') for h in headers):
                page = result.get('page', '?')
                print(f"  [HEADER-ISSUE] Page {page}: table has empty/invalid headers — vision tool should have repaired this")


def build_default_hooks() -> HookRegistry:
    """Returns a HookRegistry with default logging and validation hooks attached"""
    registry = HookRegistry()
    
    # Register pre/post hooks for vision extraction tool
    registry.register_pre_hook('vision_extract_page', log_pre_hook)
    registry.register_post_hook('vision_extract_page', log_post_hook)
    
    # Register detection hooks (post-execution only)
    registry.register_post_hook('vision_extract_page', zero_table_detection_hook)
    registry.register_post_hook('vision_extract_page', table_header_validation_hook)
    
    # Validate_json tool hooks
    registry.register_pre_hook('validate_json', log_pre_hook)
    registry.register_post_hook('validate_json', log_post_hook)
    
    return registry