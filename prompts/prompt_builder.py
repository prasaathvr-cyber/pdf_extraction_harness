from pathlib import Path
from config.settings import PROMPT_DIR


# ── Static base prompt (kept first for prompt caching) ─────────
BASE_SYSTEM_PROMPT = """You are a clinical PDF extraction specialist orchestrating vision-based extraction.

Your job is to coordinate the extraction of all content from a clinical research PDF.
The vision_extract_page tool will handle all extraction (text, tables, figures) per page.

EXTRACTION STRATEGY:
1. Decide which page to extract next (sequentially from page 1 onward)
2. Call vision_extract_page tool with the page number
3. Review the extracted content (text, tables, figures)
4. Move to the next page
5. When all pages are processed, respond with action: finish

You have access to these tools:
{tool_descriptions}

CRITICAL RULES:
- Extract pages sequentially (page 1, then 2, then 3, etc.)
- NEVER skip pages
- The vision_extract_page tool extracts text, tables, AND figures from each page in one call
- Do NOT call extract_text, extract_table, extract_figure separately — use vision_extract_page only
- When all {total_pages} pages have been extracted, call action: finish
- Always respond with valid JSON only

RESPONSE FORMAT (always respond with valid JSON only):
{{
  "reasoning": "brief explanation of what you are doing",
  "action": "vision_extract_page" | "validate_json" | "finish",
  "params": {{
    "for vision_extract_page": {{"page": <page_number>}},
    "for validate_json"     : {{}},
    "for finish"            : {{}}
  }}
}}
"""

# ── Agent-specific additions (dynamic, loaded after base) ───────
AGENT_PROMPTS = {
    "vision_extraction": """
EXTRACTION MODE: Vision-Based Per-Page Extraction
You are orchestrating the vision extraction process. Each page is extracted in full (text + tables + figures).
Your job is to:
1. Decide which page to extract next
2. Call vision_extract_page with that page number
3. Wait for the result
4. Move to the next page
5. When done, finish

Focus on: extracting every page, never skipping, capturing all tables and figures.
""",
    "text_agent": """
SPECIALIST ROLE: Text Extraction Expert
(Legacy — replaced by vision_extract_page)
""",
    "table_agent": """
SPECIALIST ROLE: Table Extraction Expert
(Legacy — replaced by vision_extract_page)
""",
    "figure_agent": """
SPECIALIST ROLE: Figure Extraction Expert
(Legacy — replaced by vision_extract_page)
"""
}

# ── NEW: Phase 3 Quality Control Instructions ──────────────────
PHASE_3_QUALITY_INSTRUCTIONS = """
After extracting each page with vision_extract_page:

1. Quality Control Protocol:
   - If tables found: Call repair_table_headers_tool
   - If encoding issues detected: Call detect_encoding_issues_tool
   - After repairs: Call validate_extraction_quality_tool
   - If quality score < 70: Use pymupdf_fallback_tool

2. Only store extraction when:
   - Quality score >= 70, OR
   - Fallback provided valid text

3. Tool usage priority:
   - repair_table_headers: Always after table extraction
   - detect_encoding: Always before storing text
   - validate_extraction_quality: Always before storing
   - pymupdf_fallback: Only if quality check fails
"""


class PromptBuilder:
    """Component #7 - System Prompt Assembly (updated for vision-based extraction)"""

    def __init__(self, tool_descriptions: list, total_pages: int = 0):
        self.tool_descriptions = tool_descriptions
        self.total_pages = total_pages

    def _format_tools(self) -> str:
        lines = []
        for t in self.tool_descriptions:
            lines.append(f"  - {t['name']}: {t['description']}")
        return "\n".join(lines)

    def build(self, agent_type: str = "vision_extraction", extra_context: str = "") -> str:
        """
        Assembles the system prompt.
        Static base first (for prompt caching), dynamic parts after.
        Updated for vision-based extraction with Phase 3 quality control.
        """
        # Static base (cached by Bedrock)
        prompt = BASE_SYSTEM_PROMPT.format(
            tool_descriptions=self._format_tools(),
            total_pages=self.total_pages
        )

        # Dynamic agent-specific addition
        agent_addition = AGENT_PROMPTS.get(agent_type, "")
        prompt += agent_addition

        # NEW: Add Phase 3 quality instructions if in Phase 3 context
        if extra_context and "Phase 3" in extra_context:
            prompt += "\n\n" + PHASE_3_QUALITY_INSTRUCTIONS

        # Dynamic extra context (e.g. progress so far)
        if extra_context:
            prompt += f"\n\nCURRENT PROGRESS:\n{extra_context}"

        # Load any .md instructions from prompts/instructions/ folder
        prompt += self._load_instruction_files()

        return prompt

    def _load_instruction_files(self) -> str:
        extra = ""
        if PROMPT_DIR.exists():
            for md_file in sorted(PROMPT_DIR.glob("*.md")):
                content = md_file.read_text(encoding='utf-8').strip()
                if content:
                    extra += f"\n\n[INSTRUCTIONS FROM {md_file.name}]\n{content}"
        return extra