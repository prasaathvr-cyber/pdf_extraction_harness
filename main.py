import sys
from pathlib import Path

from config.settings import print_config
from harness_core.harness import PDFExtractionHarness
from context.context_manager import ContextManager
from tools.registry import ToolRegistry
from tools.pdf_tools import VisionPageTool, ValidateJsonTool
from tools.page_analysis_tool import AnalyzePagesOverviewTool
from tools.priority_tool import GetExtractionPriorityTool
# ← NEW: Add Phase 4 imports
from agents.validation_agents import SubAgentOrchestrator
from tools.validation_layer import ExtractionValidator
from memory.session_logger import SessionLogger
from hooks.hook_registry import build_default_hooks
from permissions.permission_checker import PermissionChecker
from models.bedrock_client import BedrockClient
from tools.repair_table_headers_tool import RepairTableHeadersTool
from tools.detect_encoding_issues_tool import DetectEncodingIssuesTool
from tools.validate_extraction_quality_tool import ValidateExtractionQualityTool
from tools.pymupdf_fallback_tool import PyMuPDFFallbackTool

def build_harness(pdf_path: str, goal: str = "extract_all_data") -> PDFExtractionHarness:
    """Wire all 9 components together and return a ready harness"""

    print("\n  Initializing harness components...")

    # Component #9: Permissions & Safety
    permissions = PermissionChecker(user_level='WORKSPACE')

    # Component #3: Tools Registry
    registry = ToolRegistry(permission_checker=permissions)
    
    # Register tools
    registry.register(AnalyzePagesOverviewTool())
    registry.register(RepairTableHeadersTool())
    registry.register(DetectEncodingIssuesTool())
    registry.register(ValidateExtractionQualityTool())
    registry.register(PyMuPDFFallbackTool())
    registry.register(GetExtractionPriorityTool())
    registry.register(VisionPageTool())
    registry.register(ValidateJsonTool())

    # Component #2: Context Management
    context_mgr = ContextManager()

    # Component #6: Session Persistence
    session_log = SessionLogger(pdf_name=Path(pdf_path).name)

    # Component #8: Lifecycle Hooks
    hooks = build_default_hooks()

    # LLM Client (Bedrock)
    bedrock = BedrockClient()

    # Component #1: While Loop (Harness Core)
    # ← PASS GOAL HERE
    harness = PDFExtractionHarness(
        pdf_path    = pdf_path,
        goal        = goal,  # ← User-informed goal
        registry    = registry,
        context_mgr = context_mgr,
        session_log = session_log,
        hook_registry = hooks,
        bedrock     = bedrock
    )

    return harness


def main():
    # Get PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = str(Path(__file__).parent.parent.parent.parent /
                       "input_HA517" / "Faraone2004.pdf")

    # Get goal from user (optional)
    goal = "extract_all_data"  # Default
    if len(sys.argv) > 2:
        goal = sys.argv[2]  # Allow override: python main.py pdf.pdf extract_key_findings

    # Verify PDF exists
    if not Path(pdf_path).exists():
        print(f"\n  ERROR: PDF not found at '{pdf_path}'")
        print("  Usage: python main.py <path_to_pdf> [goal]")
        print("  Goals: extract_all_data, extract_key_findings, extract_methodology, extract_results")
        sys.exit(1)

    # Print configuration
    print_config()

    # Build and run harness
    harness = build_harness(pdf_path, goal)
    result  = harness.run()

    print(f"\n  Done! Result keys: {list(result.keys())}")
    print(f"  Quality Score: {result.get('quality_score', 'N/A')}/100")
    return result


if __name__ == '__main__':
    main()

#Test