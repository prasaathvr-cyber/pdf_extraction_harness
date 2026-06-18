import json
import base64
import re
import time
import fitz
from tools.base_tool import BaseTool
from harness_core.types import ToolResult
from models.bedrock_client import BedrockClient


# ── Vision-based extraction prompts ────────────────────────────────────────

VISION_PROMPT_FULL = """You are a medical document extraction specialist.

This is page {page_num} of a clinical research PDF titled: {study_name}

Extract ALL content from this page and return ONLY a raw JSON object.
No markdown, no code fences, no explanation — raw JSON only.

Required JSON structure:
{{
  "page": {page_num},
  "text": "<all body text in reading order, preserving paragraph breaks with newlines>",
  "tables": [
    {{
      "table_num": <number>,
      "caption": "<table title or caption if visible, else empty string>",
      "headers": ["<col1>", "<col2>"],
      "rows": [["<val1>", "<val2>"], ["<val1>", "<val2>"]]
    }}
  ],
  "figures": [
    {{
      "figure_num": <number>,
      "caption": "<figure caption if visible, else empty string>",
      "description": "<describe ALL data, numbers, flows, labels visible in the figure>",
      "structured_data": "<see rules below>"
    }}
  ]
}}

Critical rules:
- Preserve ALL numerical values exactly: p-values, confidence intervals, odds ratios, percentages, sample sizes, effect sizes
- Maintain correct reading order for multi-column layouts — left column first, then right column
- Include ALL text: headings, body paragraphs, footnotes, references
- If no tables on this page: "tables": []
- If no figures on this page: "figures": []

Figure extraction rules — apply based on figure type:
- Flow diagrams / CONSORT diagrams: transcribe EVERY number inside EVERY box and arrow label. structured_data must be a JSON string listing each box label and its number, e.g. '{{"Randomised": 148, "Allocated MPH": 75}}'
- Forest plots: extract EVERY row as a structured table with columns: study_label, n_treatment, n_control, effect_size, ci_lower, ci_upper, weight_percent. structured_data must be a JSON string of this table as an array of objects. Also include the pooled/diamond row.
- Bar charts / line charts: read ALL axis tick values and ALL data point values for every series. structured_data must be a JSON string listing series name and all data points with their x-axis labels.
- Scatter plots: extract ALL data points (x, y), axis labels, any regression line. structured_data = JSON string with axis names and point array.
- Other figures: structured_data = ""

Return raw JSON only — absolutely no other text before or after"""

VISION_PROMPT_TEXT_ONLY = """You are a medical document extraction specialist.

This is page {page_num} of a clinical research PDF titled: {study_name}

Extract only the body text from this page. Return ONLY a raw JSON object, no markdown, no code fences.

Required JSON structure:
{{
  "page": {page_num},
  "text": "<all body text in reading order, preserving paragraph breaks with newlines>",
  "tables": [],
  "figures": []
}}

Return raw JSON only — absolutely no other text before or after."""

VISION_PROMPT_TABLES_ONLY = """You are a medical document extraction specialist.

This is page {page_num} of a clinical research PDF titled: {study_name}

Extract only the tables and figures from this page. Return ONLY a raw JSON object, no markdown, no code fences.

Required JSON structure:
{{
  "page": {page_num},
  "tables": [
    {{
      "table_num": <number>,
      "caption": "<table title or caption if visible, else empty string>",
      "headers": ["<col1>", "<col2>"],
      "rows": [["<val1>", "<val2>"], ["<val1>", "<val2>"]]
    }}
  ],
  "figures": [
    {{
      "figure_num": <number>,
      "caption": "<figure caption if visible, else empty string>",
      "description": "<describe ALL data, numbers, flows, labels visible in the figure>",
      "structured_data": "<see rules below>"
    }}
  ]
}}

Critical rules:
- Preserve ALL numerical values exactly: p-values, confidence intervals, odds ratios, percentages, sample sizes, effect sizes
- If no tables on this page: "tables": []
- If no figures on this page: "figures": []

Figure extraction rules — apply based on figure type:
- Flow diagrams / CONSORT diagrams: transcribe EVERY number inside EVERY box and arrow label. structured_data must be a JSON string listing each box label and its number.
- Forest plots: extract EVERY row as a structured table with columns: study_label, n_treatment, n_control, effect_size, ci_lower, ci_upper, weight_percent. structured_data must be a JSON string of this table as an array of objects. Also include the pooled/diamond row.
- Bar charts / line charts: read ALL axis tick values and ALL data point values for every series. structured_data must be a JSON string listing series name and all data points with their x-axis labels.
- Scatter plots: extract ALL data points (x, y), axis labels, any regression line. structured_data = JSON string with axis names and point array.
- Other figures: structured_data = ""

Return raw JSON only — absolutely no other text before or after."""

VISION_PROMPT_HEADER_REPAIR = """You are a medical document extraction specialist.

This is page {page_num} of a clinical research PDF titled: {study_name}

A table was extracted from this page but its column headers are missing or empty.
Look carefully at the table area on this page and return ONLY a raw JSON object with the correct headers.
No markdown, no code fences, no explanation — raw JSON only.

Required JSON structure:
{{
  "headers": ["<col1>", "<col2>", "<col3>"]
}}

Rules:
- Return exactly as many headers as there are columns in the table
- If a header spans multiple sub-columns, write it as one string e.g. "Treatment (N=102)"
- If a column has no visible header text, use a descriptive label like "Parameter" or "Category"
- Return raw JSON only — absolutely no other text before or after"""


class VisionPageTool(BaseTool):
    """
    Component #5 - Built-in Skill: Vision-based extraction of text, tables, and figures
    
    Replaces the old 3 separate tools with a unified vision approach:
    1. Renders PDF page to high-res JPEG (216 DPI)
    2. Sends image to Claude Vision
    3. Handles 3-tier fallback: full → split → PyMuPDF text
    4. Repairs bad table headers automatically
    """
    name = "vision_extract_page"
    description = "Extract text, tables, and figures from a PDF page using vision. Params: pdf_path (str), page (int), study_name (str)"
    required_permission = "READ"

    def __init__(self):
        super().__init__()
        self.bedrock = BedrockClient()
        self.page_resolution = 3  # 216 DPI
        self._render_cache = {}  # Cache rendered pages to avoid re-rendering

    def _render_page_to_base64(self, pdf_path: str, page_num: int) -> str:
        """Render PDF page to JPEG and encode as base64"""
        cache_key = (pdf_path, page_num)
        if cache_key in self._render_cache:
            return self._render_cache[cache_key]

        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(self.page_resolution, self.page_resolution))

            # Ensure RGB (no alpha channel issues)
            if pix.n - pix.alpha < 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            img_bytes = pix.tobytes("jpeg", jpg_quality=92)
            doc.close()

            b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
            self._render_cache[cache_key] = b64
            return b64
        except Exception as e:
            raise RuntimeError(f"Failed to render page {page_num}: {e}")

    def _call_vision(self, b64_image: str, prompt: str, max_tokens: int = 8192) -> str:
        """Call Claude Vision via Bedrock and parse response"""
        try:
            import json as json_lib
            request_body = json_lib.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            })

            response = self.bedrock.client.invoke_model(
                modelId=self.bedrock.model_id,
                body=request_body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json_lib.loads(response["body"].read())
            raw_text = response_body["content"][0]["text"].strip()

            # Strip markdown code fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[-1]
                raw_text = raw_text.rsplit("```", 1)[0].strip()

            return raw_text
        except Exception as e:
            raise RuntimeError(f"Vision API call failed: {e}")

    def _pdftotext_fallback(self, pdf_path: str, page_num: int) -> str:
        """Fallback text extraction using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            text = page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            return f"[PyMuPDF fallback failed: {e}]"

    def _headers_need_repair(self, headers: list) -> bool:
        """Check if table headers are bad (empty, NaN, generic)"""
        if not headers:
            return True
        bad_count = 0
        for h in headers:
            s = str(h).strip()
            if s in ("", "nan", "None") or s.startswith("col_"):
                bad_count += 1
        return bad_count > len(headers) // 2

    def _repair_headers(self, b64_image: str, table: dict, page_num: int, study_name: str) -> dict:
        """Detect and repair bad table headers via re-prompt"""
        if not self._headers_need_repair(table.get("headers", [])):
            return table

        try:
            raw = self._call_vision(
                b64_image,
                VISION_PROMPT_HEADER_REPAIR.format(page_num=page_num, study_name=study_name),
                max_tokens=512
            )
            repaired = json.loads(raw)
            new_headers = repaired.get("headers", [])
            if new_headers and len(new_headers) == len(table.get("headers", new_headers)):
                table["headers"] = new_headers
                return table
        except Exception:
            pass

        return table

    def execute(self, pdf_path: str, page: int, study_name: str = "Unknown", **kwargs) -> ToolResult:
        """
        Main execution: render page → attempt full vision prompt → fallback to split → fallback to text
        """
        page_idx = page - 1  # Convert to 0-based index

        # ── Attempt 1: Render page to image ────────────────────────────────────
        try:
            b64_image = self._render_page_to_base64(pdf_path, page_idx)
        except Exception as e:
            return ToolResult(
                status="error",
                error_message=f"Failed to render page {page}: {e}",
                data={"page": page, "status": "render_failed"}
            )

        # ── Attempt 2: Full vision prompt ──────────────────────────────────────
        try:
            raw = self._call_vision(
                b64_image,
                VISION_PROMPT_FULL.format(page_num=page, study_name=study_name)
            )
            page_data = json.loads(raw)

            # Repair any bad headers
            repaired_tables = []
            for t in page_data.get("tables", []):
                repaired_tables.append(
                    self._repair_headers(b64_image, t, page, study_name)
                )
            page_data["tables"] = repaired_tables
            page_data["status"] = "success_full"

            return ToolResult(status="success", data=page_data)

        except (json.JSONDecodeError, KeyError, Exception) as e:
            pass  # Fall through to split attempt

        # ── Attempt 3: Split into text-only + tables-only ──────────────────────
        text_result = ""
        tables_result = []
        figures_result = []
        text_succeeded = False
        tables_succeeded = False

        # 3a: Text only
        try:
            raw_text = self._call_vision(
                b64_image,
                VISION_PROMPT_TEXT_ONLY.format(page_num=page, study_name=study_name),
                max_tokens=4096
            )
            parsed = json.loads(raw_text)
            text_result = parsed.get("text", "")
            text_succeeded = True
        except Exception:
            pass

        if text_succeeded and not text_result.strip():
            text_result = self._pdftotext_fallback(pdf_path, page_idx)

        if not text_succeeded:
            text_result = self._pdftotext_fallback(pdf_path, page_idx)

        # 3b: Tables + figures
        try:
            raw_tables = self._call_vision(
                b64_image,
                VISION_PROMPT_TABLES_ONLY.format(page_num=page, study_name=study_name),
                max_tokens=4096
            )
            parsed = json.loads(raw_tables)
            tables_result = parsed.get("tables", [])
            figures_result = parsed.get("figures", [])

            # Repair bad headers
            repaired_tables = []
            for t in tables_result:
                repaired_tables.append(
                    self._repair_headers(b64_image, t, page, study_name)
                )
            tables_result = repaired_tables
            tables_succeeded = True
        except Exception:
            pass

        if text_succeeded or tables_succeeded:
            return ToolResult(
                status="success",
                data={
                    "page": page,
                    "status": "success_split",
                    "text": text_result,
                    "tables": tables_result,
                    "figures": figures_result
                }
            )

        # ── Attempt 4: All vision failed → PyMuPDF text only ───────────────────
        return ToolResult(
            status="success",
            data={
                "page": page,
                "status": "partial_fallback",
                "text": text_result,
                "tables": [],
                "figures": []
            }
        )


class ValidateJsonTool(BaseTool):
    """Component #5 - Built-in Skill: Validate extraction before finishing"""
    name = "validate_json"
    description = "Validate the current extraction output. Params: state (dict)"
    required_permission = "READ"

    def execute(self, state: dict = None, **kwargs) -> ToolResult:
        issues = []

        if not state:
            return ToolResult(status="error", error_message="No state to validate")

        if not state.get('text_chunks'):
            issues.append("No text extracted")
        if len(state.get('text_chunks', [])) < 1:
            issues.append("Text list is empty")

        if issues:
            return ToolResult(
                status="success",
                data={"valid": False, "issues": issues}
            )

        return ToolResult(
            status="success",
            data={
                "valid"         : True,
                "text_sections" : len(state.get('text_chunks', [])),
                "tables_found"  : len(state.get('tables', [])),
                "figures_found" : len(state.get('figures', []))
            }
        )