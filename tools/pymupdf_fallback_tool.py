from tools.base_tool import BaseTool
from harness_core.types import ToolResult
import fitz

class PyMuPDFFallbackTool(BaseTool):
    """Fallback to PyMuPDF for text extraction when vision fails"""
    
    def metadata(self):
        return {
            "name": "pymupdf_fallback",
            "description": "Extract text using PyMuPDF when vision fails",
            "parameters": {
                "pdf_path": "str - Path to PDF",
                "page_num": "int - Page number"
            }
        }
    
    def execute(self, **kwargs) -> ToolResult:
        try:
            pdf_path = kwargs.get("pdf_path")
            page_num = kwargs.get("page_num")
            
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]
            text = page.get_text()
            doc.close()
            
            return ToolResult(
                status="success",
                data={
                    "page": page_num,
                    "text": text,
                    "tables": [],
                    "figures": [],
                    "method": "pymupdf_fallback"
                }
            )
        
        except Exception as e:
            return ToolResult(status="error", error_message=str(e))