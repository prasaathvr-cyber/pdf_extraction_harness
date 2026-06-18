from tools.base_tool import BaseTool
from harness_core.types import ToolResult

class ValidateExtractionQualityTool(BaseTool):
    """Validate extraction quality before storing"""
    
    def metadata(self):
        return {
            "name": "validate_extraction_quality",
            "description": "Validate extraction quality and scoring",
            "parameters": {
                "extraction": "dict - Extracted data",
                "page_num": "int - Page number"
            }
        }
    
    def execute(self, **kwargs) -> ToolResult:
        try:
            extraction = kwargs.get("extraction", {})
            page_num = kwargs.get("page_num")
            
            score = 100
            issues = []
            
            # Check text quality
            text = extraction.get("text", "")
            if not text or len(text) < 50:
                score -= 30
                issues.append("insufficient_text")
            elif "/C" in text or "x9" in text or "\u0004" in text:
                score -= 20
                issues.append("encoding_issues_present")
            
            # Check tables quality
            tables = extraction.get("tables", [])
            if tables:
                for t in tables:
                    headers = t.get("headers", [])
                    if not headers:
                        score -= 10
                        issues.append("missing_table_headers")
            
            # Check figures quality
            figures = extraction.get("figures", [])
            
            quality_level = "EXCELLENT" if score >= 90 else "GOOD" if score >= 75 else "ACCEPTABLE" if score >= 60 else "POOR"
            
            return ToolResult(
                status="success",
                data={
                    "quality_score": score,
                    "quality_level": quality_level,
                    "issues": issues,
                    "recommendation": "STORE" if score >= 70 else "FALLBACK"
                }
            )
        
        except Exception as e:
            return ToolResult(status="error", error_message=str(e))