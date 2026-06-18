from tools.base_tool import BaseTool
from harness_core.types import ToolResult
import re

class DetectEncodingIssuesTool(BaseTool):
    """Detects and fixes text encoding issues"""
    
    def metadata(self):
        return {
            "name": "detect_encoding_issues",
            "description": "Detect and fix text encoding corruption",
            "parameters": {
                "text": "str - Text to check for encoding issues"
            }
        }
    
    def execute(self, **kwargs) -> ToolResult:
        try:
            text = kwargs.get("text", "")
            
            if not text:
                return ToolResult(status="success", data={
                    "issues": [],
                    "cleaned_text": text,
                    "quality": "CLEAN"
                })
            
            issues = []
            cleaned = text
            
            # Fix chi-square
            if "x9\n2" in cleaned or "x92" in cleaned:
                cleaned = re.sub(r'x9\s*2\s*=', 'χ² =', cleaned)
                cleaned = re.sub(r'x92\s*=', 'χ² =', cleaned)
                issues.append("chi_square_corruption_fixed")
            
            # Fix minus signs
            if "\u0004" in cleaned:
                cleaned = re.sub(r'\u00040+(\d+)', r'-\1', cleaned)
                issues.append("minus_sign_corruption_fixed")
            
            # Remove control characters
            before = len(cleaned)
            cleaned = ''.join(c for c in cleaned if ord(c) >= 32 or c in '\n\t\r')
            if len(cleaned) < before:
                issues.append("control_characters_removed")
            
            # Remove journal control artifacts
            if "/C" in cleaned:
                cleaned = re.sub(r'/C\d{2}', '', cleaned)
                issues.append("journal_control_chars_removed")
            
            quality = "CLEAN" if not issues else "FIXED"
            
            return ToolResult(
                status="success",
                data={
                    "issues": issues,
                    "cleaned_text": cleaned,
                    "quality": quality,
                    "chars_removed": before - len(cleaned) if before > len(cleaned) else 0
                }
            )
        
        except Exception as e:
            return ToolResult(status="error", error_message=str(e))