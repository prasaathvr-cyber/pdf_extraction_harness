import json
import re
from typing import List, Dict, Any
from tools.base_tool import BaseTool
from harness_core.types import ToolResult
from models.bedrock_client import BedrockClient


class GetExtractionPriorityTool(BaseTool):
    """
    Component #5 - Built-in Skill: Intelligent page prioritization
    
    UPDATED: All pages will be extracted. No SKIP category.
    Model decides extraction ORDER (HIGH → MEDIUM → LOW), not deletion.
    
    Takes page analysis from AnalyzePagesOverviewTool and asks the model
    to prioritize which pages to extract FIRST (HIGH), SECOND (MEDIUM), 
    THIRD (LOW).
    
    This is Phase 2 of adaptive extraction:
    - Model analyzes page types
    - Model decides extraction ORDER (priority)
    - Returns prioritized extraction plan
    - Agent MUST extract all pages, no exceptions
    """
    
    name = "get_extraction_priority"
    description = "Analyze page types and create an intelligent extraction priority plan. Model prioritizes extraction order: HIGH (extract first), MEDIUM (extract second), LOW (extract last). ALL pages will be extracted."
    required_permission = "READ"

    def __init__(self):
        super().__init__()
        self.bedrock = BedrockClient()

    def _build_prioritization_prompt(self, pages_analysis: List[Dict], pdf_name: str, total_pages: int) -> str:
        """
        Build a focused prompt asking the model to prioritize page extraction order.
        
        IMPORTANT: No SKIP category. All pages will be extracted.
        Model decides ORDER only, not deletion.
        """
        pages_json = json.dumps(pages_analysis, indent=2)
        
        prompt = f"""You are a document analysis expert specializing in clinical research PDFs.

DOCUMENT: {pdf_name}
TOTAL PAGES: {total_pages}

HERE IS THE PAGE STRUCTURE:
{pages_json}

YOUR TASK:
Analyze this page structure and create an extraction priority plan.
ALL PAGES WILL BE EXTRACTED. You are NOT deciding what to skip.
You are deciding the ORDER in which pages should be extracted.

Assign each page a priority level:
- HIGH priority: Extract first. Contains key data (methods, results, tables, figures, critical data)
- MEDIUM priority: Extract second. Supporting content (introduction, discussion, interpretation)
- LOW priority: Extract last. Metadata and references (title, abstract, references, appendices, boilerplate)

RULES:
1. Methods and Results pages are HIGH (core data)
2. Introduction and Discussion are MEDIUM (context)
3. Title, Abstract, References, Appendices are LOW (metadata)
4. EVERY page gets one of these three priorities
5. NO page is skipped or excluded
6. You are deciding EXTRACTION ORDER, not what to keep/discard

RETURN ONLY RAW JSON (no markdown, no explanation):
{{
  "strategy": "brief one-line summary of your extraction order strategy",
  "rationale": "2-3 sentence explanation of why you prioritized this way",
  "extraction_plan": [
    {{"page": <number>, "page_type": "<from input>", "priority": "HIGH|MEDIUM|LOW", "reason": "why this priority"}},
    ...
  ],
  "total_pages_to_extract": <total number of pages (should equal {total_pages})>,
  "high_priority_pages": [<list of HIGH page numbers>],
  "medium_priority_pages": [<list of MEDIUM page numbers>],
  "low_priority_pages": [<list of LOW page numbers>]
}}

Be concise. Return ONLY the JSON object, nothing else."""
        
        return prompt

    def _parse_extraction_plan(self, raw_response: str) -> Dict[str, Any]:
        """
        Parse the model's response into a structured extraction plan.
        Handles both clean JSON and JSON with markdown fences.
        """
        # Try to extract JSON from response
        try:
            # Remove markdown fences if present
            if "```json" in raw_response:
                json_str = raw_response.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_response:
                json_str = raw_response.split("```")[1].split("```")[0].strip()
            else:
                json_str = raw_response
            
            # Find JSON object (in case there's text before/after)
            json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            
            parsed = json.loads(json_str)
            return parsed
        
        except (json.JSONDecodeError, IndexError) as e:
            print(f"  [WARNING] Failed to parse JSON response: {e}")
            print(f"  Raw response: {raw_response[:500]}")
            return None

    def execute(self, pages_analysis: List[Dict], pdf_path: str = None, pdf_name: str = None, **kwargs) -> ToolResult:
        """
        Main execution: call model to prioritize extraction order.
        
        IMPORTANT: All pages will be extracted. No pages are skipped.
        
        Inputs:
            pages_analysis: Output from AnalyzePagesOverviewTool
                Expected: list of {"page": int, "page_type": str, "confidence": float, "preview": str}
            pdf_path: Path to PDF (for context)
            pdf_name: Name of PDF (for display)
        
        Returns:
            ToolResult with data containing:
            {
                "strategy": str,
                "rationale": str,
                "extraction_plan": [
                    {
                        "page": int,
                        "page_type": str,
                        "priority": "HIGH|MEDIUM|LOW",
                        "reason": str
                    },
                    ...
                ],
                "total_pages_to_extract": int,
                "high_priority_pages": [int, ...],
                "medium_priority_pages": [int, ...],
                "low_priority_pages": [int, ...],
                "all_pages": [int, ...],  # NEW: All pages that will be extracted
                "total_pages": int
            }
        """
        
        if not pages_analysis:
            return ToolResult(
                status="error",
                error_message="pages_analysis is empty",
                data={}
            )
        
        try:
            total_pages = len(pages_analysis)
            pdf_name = pdf_name or "Unknown PDF"
            
            print(f"  Creating extraction priority plan (all pages will be extracted)...")
            
            # Build the prioritization prompt
            prompt = self._build_prioritization_prompt(pages_analysis, pdf_name, total_pages)
            
            # Call Bedrock to get model's prioritization decision
            system_prompt = "You are an expert at analyzing document structure and creating extraction strategies. Be concise and strategic."
            
            raw_response = self.bedrock.invoke(system_prompt, [
                {"role": "user", "content": prompt}
            ])
            
            # Parse the response
            priority_plan = self._parse_extraction_plan(raw_response)
            
            if not priority_plan or "extraction_plan" not in priority_plan:
                return ToolResult(
                    status="error",
                    error_message="Model did not return valid extraction plan",
                    data={"raw_response": raw_response[:500]}
                )
            
            # Validate and enhance the plan
            extraction_plan = priority_plan.get("extraction_plan", [])
            
            # Collect pages by priority
            high_priority = [item["page"] for item in extraction_plan if item.get("priority") == "HIGH"]
            medium_priority = [item["page"] for item in extraction_plan if item.get("priority") == "MEDIUM"]
            low_priority = [item["page"] for item in extraction_plan if item.get("priority") == "LOW"]
            
            # NEW: All pages combined (in order: HIGH, MEDIUM, LOW)
            all_pages = high_priority + medium_priority + low_priority
            
            # Validation: Ensure all pages are assigned (no skips)
            if len(all_pages) != total_pages:
                print(f"  [WARNING] Not all pages were prioritized!")
                print(f"  Expected {total_pages} pages, got {len(all_pages)}")
                print(f"  Missing pages will be treated as LOW priority")
                
                # Add any missing pages as LOW priority
                assigned_pages = set(all_pages)
                for page_num in range(1, total_pages + 1):
                    if page_num not in assigned_pages:
                        low_priority.append(page_num)
                        all_pages.append(page_num)
            
            print(f"  ✓ Extraction priority plan created:")
            print(f"    - HIGH priority ({len(high_priority)} pages): {high_priority[:10]}{'...' if len(high_priority) > 10 else ''}")
            print(f"    - MEDIUM priority ({len(medium_priority)} pages): {medium_priority[:10]}{'...' if len(medium_priority) > 10 else ''}")
            print(f"    - LOW priority ({len(low_priority)} pages): {low_priority[:10]}{'...' if len(low_priority) > 10 else ''}")
            print(f"    - TOTAL PAGES TO EXTRACT: {len(all_pages)} (100% of PDF)")
            print(f"    Strategy: {priority_plan.get('strategy', 'N/A')}\n")
            
            # Return the full plan with metadata
            result_data = {
                "strategy": priority_plan.get("strategy", "Extract all pages in priority order"),
                "rationale": priority_plan.get("rationale", ""),
                "extraction_plan": extraction_plan,
                "total_pages_to_extract": len(all_pages),
                "high_priority_pages": high_priority,
                "medium_priority_pages": medium_priority,
                "low_priority_pages": low_priority,
                "all_pages": sorted(all_pages),  # NEW: All pages in numeric order
                "total_pages": total_pages
            }
            
            return ToolResult(
                status="success",
                data=result_data
            )
        
        except Exception as e:
            import traceback
            print(f"  [ERROR] {str(e)}")
            traceback.print_exc()
            return ToolResult(
                status="error",
                error_message=f"Failed to create extraction priority: {str(e)}",
                data={}
            )