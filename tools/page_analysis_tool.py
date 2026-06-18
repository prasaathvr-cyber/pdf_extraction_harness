import fitz
import json
from typing import List, Dict
from tools.base_tool import BaseTool
from harness_core.types import ToolResult


class AnalyzePagesOverviewTool(BaseTool):
    """
    Component #5 - Built-in Skill: Quick page-type classification
    
    Fast skim of ALL pages using PyMuPDF (no vision model).
    Classifies each page as: title, abstract, methods, results, discussion, references, other
    
    This is Phase 1 of adaptive extraction:
    - Fast (no vision calls)
    - Gives model page structure info
    - Enables intelligent prioritization in Phase 2
    """
    
    name = "analyze_pages_overview"
    description = "Quickly analyze all pages to classify their type (title, abstract, methods, results, discussion, references, other). Returns page types for intelligent extraction planning."
    required_permission = "READ"

    def __init__(self):
        super().__init__()

    def _classify_page(self, text: str, page_num: int, total_pages: int) -> str:
        """
        Classify a page based on its text content and position.
        Uses simple heuristics: keywords, position in doc, structural markers.
        """
        text_lower = text.lower()
        
        # Position-based rules (very reliable)
        if page_num == 0:
            return "title"
        
        if page_num == total_pages - 1:
            return "references"
        
        # Keyword-based rules
        # Check for abstract (usually early, short section)
        if ("abstract" in text_lower or "summary" in text_lower) and page_num <= 3:
            return "abstract"
        
        # Introduction comes early
        if "introduction" in text_lower and page_num <= 5:
            return "introduction"
        
        # Methods/Design section (substantial text, tables)
        if ("method" in text_lower or "design" in text_lower or 
            "participant" in text_lower or "population" in text_lower or
            "procedure" in text_lower or "protocol" in text_lower):
            return "methods"
        
        # Results section (tables, numbers, statistics)
        if ("result" in text_lower or "finding" in text_lower or 
            "outcome" in text_lower or "efficacy" in text_lower or
            "p < " in text or "p=" in text):
            return "results"
        
        # Discussion/Conclusion (interpretation, implications)
        if ("discussion" in text_lower or "conclusion" in text_lower or 
            "implication" in text_lower or "limitation" in text_lower):
            return "discussion"
        
        # References (lots of citations)
        if "reference" in text_lower or text_lower.count("[") > 20:
            return "references"
        
        # Appendix/Supplementary
        if "appendix" in text_lower or "supplementary" in text_lower:
            return "appendix"
        
        # Default
        return "other"

    def _extract_page_text(self, pdf_path: str, page_num: int, sample_size: int = 1500) -> str:
        """
        Extract text from a single page using PyMuPDF.
        Limits to sample_size chars to keep it quick.
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            text = page.get_text()
            doc.close()
            
            # Return first N chars (usually enough to classify)
            return text[:sample_size].strip()
        except Exception as e:
            return f"[Error reading page {page_num}: {str(e)}]"

    def execute(self, pdf_path: str, **kwargs) -> ToolResult:
        """
        Main execution: analyze all pages and return classification.
        
        Returns:
            ToolResult with data containing:
            {
                "total_pages": int,
                "pages": [
                    {
                        "page": int (1-indexed),
                        "page_type": str,
                        "confidence": float (0.0-1.0),
                        "preview": str (first 150 chars)
                    },
                    ...
                ]
            }
        """
        try:
            # Open PDF and get page count
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
            
            if total_pages == 0:
                return ToolResult(
                    status="error",
                    error_message="PDF has 0 pages",
                    data={"total_pages": 0, "pages": []}
                )
            
            print(f"  Analyzing {total_pages} pages...")
            
            pages_analysis = []
            
            # Analyze each page
            for page_idx in range(total_pages):
                # Extract text sample
                text_sample = self._extract_page_text(pdf_path, page_idx)
                
                # Classify page type
                page_type = self._classify_page(text_sample, page_idx, total_pages)
                
                # Confidence is simple: full confidence for position-based (first/last),
                # lower confidence for keyword-based
                if page_idx == 0 or page_idx == total_pages - 1:
                    confidence = 0.95
                elif page_type in ("abstract", "introduction"):
                    confidence = 0.85
                elif page_type in ("methods", "results", "discussion"):
                    confidence = 0.80
                else:
                    confidence = 0.60
                
                # Store analysis for this page
                analysis = {
                    "page": page_idx + 1,  # 1-indexed for readability
                    "page_type": page_type,
                    "confidence": confidence,
                    "preview": text_sample[:150]  # First 150 chars as preview
                }
                
                pages_analysis.append(analysis)
                print(f"    Page {page_idx + 1:3d}: {page_type:15s} (conf: {confidence:.2f})")
            
            print(f"  ✓ Page analysis complete\n")
            
            return ToolResult(
                status="success",
                data={
                    "total_pages": total_pages,
                    "pages": pages_analysis
                }
            )
        
        except Exception as e:
            return ToolResult(
                status="error",
                error_message=f"Failed to analyze pages: {str(e)}",
                data={"total_pages": 0, "pages": []}
            )