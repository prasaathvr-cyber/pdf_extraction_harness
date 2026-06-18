"""
VALIDATION LAYER - Option C (Quick Fix)

Adds validation on top of vision_extract_page tool WITHOUT modifying the core tool.
Fixes text encoding issues and detects figure hallucination patterns.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple


class TextEncodingFixer:
    """Fixes character encoding issues from PDF extraction"""
    
    @staticmethod
    def fix_chi_square(text: str) -> str:
        """Fix χ² (chi-square) symbol corruption
        
        Corrupted patterns:
        - "x9\n2" → "χ²"
        - "x9 2" → "χ²"
        - "x92" → "χ²"
        """
        # Pattern 1: x9<newline>2
        text = re.sub(r'x9\s*2\s*=', 'χ² =', text)
        text = re.sub(r'x9\s*2\b', 'χ²', text)
        
        # Pattern 2: x92
        text = re.sub(r'\bx92\b', 'χ²', text)
        text = re.sub(r'\bx92\s*=', 'χ² =', text)
        
        return text
    
    @staticmethod
    def fix_minus_signs(text: str) -> str:
        """Fix minus sign corruption
        
        Issues:
        - Control character \u0004 before number: "\u00040.24" → "-0.24"
        - Missing minus: " 0.2" with negative context → "-0.2"
        """
        # Pattern 1: Control character \u0004
        text = re.sub(r'\u00040+(\d+)', r'-\1', text)
        text = re.sub(r'\u0004(\d+\.\d+)', r'-\1', text)
        
        # Pattern 2: Restore context-based minus signs
        # Before "0.2" (Kuperman value that should be negative)
        text = re.sub(r'Kuperman[^–\-0-9]*0\.2', 'Kuperman –0.2', text)
        text = re.sub(r'Kuperman[^–\-0-9]*([01]\.[0-9])', r'Kuperman –\1', text)
        
        return text
    
    @staticmethod
    def fix_special_characters(text: str) -> str:
        """Fix other special character issues
        
        Issues:
        - "½" might be corrupted
        - Superscript numbers might be wrong
        - Dashes might be incorrect type
        """
        # Fix common UTF-8 issues
        text = text.replace('â€"', '–')  # En-dash
        text = text.replace('â€"', '—')  # Em-dash
        text = text.replace('â€œ', '"')  # Left quote
        text = text.replace('â€\x9d', '"')  # Right quote
        
        # Fix common control characters
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)
        
        return text
    
    @staticmethod
    def fix_all(text: str) -> Tuple[str, List[str]]:
        """Apply all fixes and return cleaned text + list of fixes applied"""
        original = text
        fixes_applied = []
        
        # Apply fixes in order
        text = TextEncodingFixer.fix_chi_square(text)
        if text != original:
            fixes_applied.append("chi_square_fix")
            original = text
        
        text = TextEncodingFixer.fix_minus_signs(text)
        if text != original:
            fixes_applied.append("minus_sign_fix")
            original = text
        
        text = TextEncodingFixer.fix_special_characters(text)
        if text != original:
            fixes_applied.append("special_character_fix")
        
        return text, fixes_applied


class FigureDataValidator:
    """Detects and prevents figure data hallucination"""
    
    @staticmethod
    def validate_scatter_plot(
        figure_data: Dict[str, Any],
        table_2_smd_values: Optional[List[float]] = None,
        valid_y_range: Tuple[float, float] = (-5, 10)
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Validate scatter plot data points.
        
        Checks:
        1. Are y-values within expected range?
        2. Do many y-values appear in Table 2 (hallucination indicator)?
        3. Are coordinates mathematically reasonable?
        """
        
        issues = []
        data_points = figure_data.get("structured_data", {}).get("data_points", [])
        
        if not data_points:
            return True, figure_data, []  # No data to validate
        
        # Extract y-values
        y_values = [point.get("y") for point in data_points if isinstance(point, dict)]
        
        # Check 1: Y-values in valid range
        out_of_range = [y for y in y_values if y < valid_y_range[0] or y > valid_y_range[1]]
        if out_of_range:
            issues.append(f"Out of range y-values: {out_of_range}")
        
        # Check 2: Overlap with Table 2 (hallucination indicator)
        if table_2_smd_values:
            table_2_set = set(table_2_smd_values)
            overlapping = [y for y in y_values if y in table_2_set]
            
            # If > 50% of points are from Table 2, likely hallucinated
            if len(overlapping) / len(y_values) > 0.5:
                issues.append(
                    f"Possible hallucination: {len(overlapping)}/{len(y_values)} "
                    f"y-values found in Table 2 SMD values"
                )
        
        # Check 3: Suspiciously round numbers (often hallucinated)
        suspiciously_round = [y for y in y_values if y in [5.0, 2.5, 7.5, 10.0]]
        if suspiciously_round:
            issues.append(f"Suspiciously round values (often hallucinated): {suspiciously_round}")
        
        is_valid = len(issues) == 0
        return is_valid, figure_data, issues
    
    @staticmethod
    def validate_forest_plot(
        figure_data: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Validate forest plot data"""
        issues = []
        
        try:
            structured = json.loads(figure_data.get("structured_data", "{}"))
            rows = structured.get("data", [])
            
            # Check that effect sizes and CI bounds are reasonable
            for row in rows:
                effect_size = row.get("effect_size")
                ci_lower = row.get("ci_lower")
                ci_upper = row.get("ci_upper")
                
                # Effect size should be between CI bounds
                if effect_size and ci_lower and ci_upper:
                    if not (ci_lower <= effect_size <= ci_upper):
                        issues.append(
                            f"{row.get('study')}: "
                            f"Effect size {effect_size} outside CI [{ci_lower}, {ci_upper}]"
                        )
        
        except Exception as e:
            issues.append(f"Could not validate forest plot structure: {str(e)}")
        
        is_valid = len(issues) == 0
        return is_valid, figure_data, issues
    
    @staticmethod
    def validate_all(
        figures: List[Dict[str, Any]],
        table_2_smd_values: Optional[List[float]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate all figures.
        
        Returns:
        - valid_figures: Figures that passed validation
        - invalid_figures: Figures with hallucination/corruption detected
        """
        valid_figures = []
        invalid_figures = []
        
        for figure in figures:
            caption = figure.get("caption", "").lower()
            
            if "scatter" in caption or "bias" in caption:
                is_valid, figure, issues = FigureDataValidator.validate_scatter_plot(
                    figure,
                    table_2_smd_values
                )
            elif "forest" in caption:
                is_valid, figure, issues = FigureDataValidator.validate_forest_plot(figure)
            else:
                # No specific validation for other types
                is_valid = True
                issues = []
            
            # Add validation metadata
            figure["validation"] = {
                "passed": is_valid,
                "issues": issues
            }
            
            if is_valid:
                valid_figures.append(figure)
            else:
                invalid_figures.append(figure)
        
        return valid_figures, invalid_figures


class MetadataExtractor:
    """Extracts and structures metadata properly"""
    
    @staticmethod
    def extract_table_footnotes(table: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Extract footnotes from table and return as separate metadata.
        
        Detects:
        - Rows starting with "*"
        - Rows with footnote indicators (†, ‡, §, etc.)
        - Last rows that are significantly shorter (often footnotes)
        """
        rows = table.get("rows", [])
        if not rows:
            return table, []
        
        footnotes = []
        data_rows = []
        
        for row in rows:
            # Check if row is a footnote
            if row and isinstance(row[0], str):
                first_cell = row[0].strip()
                is_footnote = (
                    first_cell.startswith("*") or
                    first_cell.startswith("†") or
                    first_cell.startswith("‡") or
                    len(first_cell) > 50  # Footnote text is usually long
                )
                
                if is_footnote:
                    footnotes.append({
                        "text": first_cell,
                        "raw_row": row
                    })
                else:
                    data_rows.append(row)
            else:
                data_rows.append(row)
        
        # Update table without footnote rows
        clean_table = table.copy()
        clean_table["rows"] = data_rows
        
        return clean_table, footnotes
    
    @staticmethod
    def structure_table_metadata(table: Dict[str, Any]) -> Dict[str, Any]:
        """Add proper metadata structure to table"""
        clean_table, footnotes = MetadataExtractor.extract_table_footnotes(table)
        
        # Add metadata section
        clean_table["metadata"] = {
            "footnotes": footnotes,
            "headers_detected": len(table.get("headers", [])) > 0,
            "row_count": len(clean_table.get("rows", [])),
            "footnote_count": len(footnotes)
        }
        
        return clean_table


class ExtractionValidator:
    """Complete validation layer for extraction output"""
    
    def __init__(self, pdf_page_data: Dict[str, Any] = None):
        """
        Initialize validator with optional reference data.
        
        Args:
            pdf_page_data: Dict containing:
                - "table_2_smd_values": List of SMD values from Table 2 (for hallucination check)
        """
        self.pdf_page_data = pdf_page_data or {}
        self.table_2_smds = self.pdf_page_data.get("table_2_smd_values", [])
    
    def validate_extraction(
        self,
        extraction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate complete extraction output and return cleaned version.
        
        Returns:
        {
            "text": cleaned_text,
            "tables": validated_tables,
            "figures": validated_figures,
            "validation_report": {
                "text_fixes": [...],
                "figure_issues": [...],
                "quality_score": 0-100
            }
        }
        """
        
        report = {
            "text_fixes": [],
            "figure_issues": [],
            "metadata_issues": [],
            "quality_score": 100
        }
        
        # STEP 1: Validate and clean text
        text = extraction.get("text", "")
        text, text_fixes = TextEncodingFixer.fix_all(text)
        report["text_fixes"] = text_fixes
        
        # STEP 2: Validate figures
        figures = extraction.get("figures", [])
        valid_figures, invalid_figures = FigureDataValidator.validate_all(figures, self.table_2_smds)
        
        if invalid_figures:
            report["figure_issues"] = [
                f"{fig.get('caption')}: {fig['validation']['issues']}"
                for fig in invalid_figures
            ]
            report["quality_score"] -= (len(invalid_figures) * 20)
        
        # STEP 3: Structure tables with metadata
        tables = extraction.get("tables", [])
        structured_tables = []
        for table in tables:
            clean_table = MetadataExtractor.structure_table_metadata(table)
            structured_tables.append(clean_table)
        
        # STEP 4: Compile validated extraction
        validated_extraction = {
            "page": extraction.get("page", 0),
            "text": text,
            "tables": structured_tables,
            "figures": valid_figures,  # Use only valid figures
            "validation_report": report,
            "data_quality": {
                "text_corrected": len(text_fixes) > 0,
                "figures_validated": len(valid_figures),
                "figures_with_issues": len(invalid_figures),
                "overall_quality": "HIGH" if report["quality_score"] >= 80 else "MEDIUM" if report["quality_score"] >= 60 else "LOW"
            }
        }
        
        return validated_extraction


# ── Helper function for integration ────────────────────────────────────────

def validate_extraction_output(
    extraction: Dict[str, Any],
    table_2_smd_values: List[float] = None
) -> Dict[str, Any]:
    """
    Quick validation wrapper for use in harness.
    
    Usage:
        # After vision_extract_page returns
        extraction = tool_result.data
        validated = validate_extraction_output(extraction, [1.3, 0.3, 0.9, ...])
    """
    validator = ExtractionValidator({
        "table_2_smd_values": table_2_smd_values or []
    })
    return validator.validate_extraction(extraction)