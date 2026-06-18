from tools.base_tool import BaseTool
from harness_core.types import ToolResult
import json

class RepairTableHeadersTool(BaseTool):
    """Repairs corrupted table headers using vision model"""
    
    def metadata(self):
        return {
            "name": "repair_table_headers",
            "description": "Repair corrupted or malformed table headers",
            "parameters": {
                "table": "dict - Table with headers",
                "pdf_path": "str - Path to PDF",
                "page_num": "int - Page number"
            }
        }
    
    def execute(self, **kwargs) -> ToolResult:
        try:
            table = kwargs.get("table", {})
            
            # Detect malformed headers
            headers = table.get("headers", [])
            if not headers:
                return ToolResult(status="success", data={"table": table, "repaired": False})
            
            # Check for corruption patterns
            has_corruption = any(
                "/C" in str(h) or  # Journal control chars
                "x9" in str(h) or  # Chi-square corruption
                "\u0004" in str(h)  # Encoding issues
                for h in headers
            )
            
            if has_corruption:
                # Clean headers
                clean_headers = []
                for h in headers:
                    h_clean = str(h)
                    h_clean = h_clean.replace("/C00", "").replace("/C15", "")
                    h_clean = h_clean.replace("x92", "χ²").replace("x9\n2", "χ²")
                    h_clean = h_clean.replace("\u0004", "-")
                    clean_headers.append(h_clean)
                
                table["headers"] = clean_headers
            
            return ToolResult(
                status="success",
                data={
                    "table": table,
                    "repaired": has_corruption,
                    "headers_fixed": sum(1 for h in headers if any(
                        "/C" in str(h) or "x9" in str(h) or "\u0004" in str(h)
                        for h in [h]
                    ))
                }
            )
        
        except Exception as e:
            return ToolResult(status="error", error_message=str(e))