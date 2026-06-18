"""
COMPONENT #4: SUB-AGENTS IMPLEMENTATION

Three specialized sub-agents that validate extraction in parallel.
Each agent has a focused task, restricted tool set, and own session.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


# ── Base Sub-Agent ─────────────────────────────────────────────────────────

@dataclass
class SubAgentState:
    """State for a sub-agent session"""
    agent_name: str
    task: str
    iteration: int = 0
    messages: List[Dict[str, str]] = None
    results: Dict[str, Any] = None
    issues_found: List[str] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.results is None:
            self.results = {}
        if self.issues_found is None:
            self.issues_found = []


class SubAgent(ABC):
    """Base class for sub-agents"""
    
    def __init__(self, name: str, restricted_tools: List[str] = None):
        self.name = name
        self.restricted_tools = restricted_tools or []
        self.state = SubAgentState(agent_name=name, task=self.get_task())
    
    @abstractmethod
    def get_task(self) -> str:
        """Get this agent's task description"""
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Validate data and return (is_valid, processed_data, issues_found)
        """
        pass
    
    def get_system_prompt(self) -> str:
        """Get specialized system prompt for this agent"""
        return f"""You are a specialized validation agent: {self.name}

Your task: {self.get_task()}

You have access to these tools only: {', '.join(self.restricted_tools)}

You do NOT have permission to:
- Modify the harness architecture
- Extract new data
- Make autonomous decisions beyond your task

Focus strictly on your validation task. Return structured JSON responses."""


# ── Figure Validation Sub-Agent ────────────────────────────────────────────

class FigureValidationAgent(SubAgent):
    """Validates figure data and detects hallucination"""
    
    def __init__(self):
        super().__init__(
            name="Figure Validation Agent",
            restricted_tools=["validate_coordinates", "cross_check_tables"]
        )
        self.table_2_smds = {}
    
    def get_task(self) -> str:
        return """Validate extracted figure data:
1. Check if figure coordinates are real or hallucinated
2. Cross-reference with Table 2 SMD values
3. Detect suspicious/invented numbers
4. Flag figures with issues
5. Return only validated figures"""
    
    def set_reference_data(self, table_2_smds: List[float]):
        """Set Table 2 SMD values for hallucination detection"""
        self.table_2_smds = set(table_2_smds)
    
    def validate(self, figures: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Validate all figures"""
        self.state.iteration += 1
        
        validated_figures = []
        issues = []
        
        for figure in figures:
            fig_name = figure.get("caption", "Unknown")
            
            # Validate figure data
            fig_is_valid, fig_data, fig_issues = self._validate_figure(figure)
            
            if fig_issues:
                issues.extend([f"{fig_name}: {issue}" for issue in fig_issues])
            
            if fig_is_valid:
                validated_figures.append(fig_data)
            else:
                # Flag but don't discard - let harness decide
                fig_data["validation_status"] = "FLAGGED_WITH_ISSUES"
                fig_data["validation_issues"] = fig_issues
                validated_figures.append(fig_data)
        
        # Compile result
        result = {
            "figures_validated": len(validated_figures),
            "issues_found": len(issues),
            "validated_figures": validated_figures
        }
        
        self.state.results = result
        self.state.issues_found = issues
        
        # Overall validity: True if no high-severity issues
        is_valid = len([i for i in issues if "hallucination" in i.lower()]) == 0
        
        return is_valid, result, issues
    
    def _validate_figure(self, figure: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Validate individual figure"""
        issues = []
        caption = figure.get("caption", "").lower()
        
        # Get structured data
        try:
            structured = json.loads(figure.get("structured_data", "{}"))
        except:
            structured = {}
        
        # Validate based on figure type
        if "scatter" in caption or "bias" in caption:
            issues.extend(self._validate_scatter_plot(figure, structured))
        elif "forest" in caption:
            issues.extend(self._validate_forest_plot(figure, structured))
        
        is_valid = len(issues) == 0
        return is_valid, figure, issues
    
    def _validate_scatter_plot(self, figure: Dict[str, Any], structured: Dict) -> List[str]:
        """Validate scatter plot specifically"""
        issues = []
        data_points = structured.get("data_points", [])
        
        if not data_points:
            return issues
        
        y_values = [p.get("y") for p in data_points if isinstance(p, dict)]
        
        # Check 1: Do > 50% of y-values come from Table 2?
        if self.table_2_smds:
            table_overlap = sum(1 for y in y_values if y in self.table_2_smds)
            overlap_ratio = table_overlap / len(y_values) if y_values else 0
            
            if overlap_ratio > 0.5:
                issues.append(
                    f"HALLUCINATION WARNING: {table_overlap}/{len(y_values)} "
                    f"y-values appear in Table 2 SMD list (likely extracted from table, not figure)"
                )
        
        # Check 2: Suspiciously round numbers
        suspicious = [y for y in y_values if y in [5.0, 2.5, 7.5, 10.0, 1.0]]
        if suspicious:
            issues.append(f"Suspiciously round y-values (often hallucinated): {suspicious}")
        
        # Check 3: Out of expected range
        out_of_range = [y for y in y_values if y < -10 or y > 15]
        if out_of_range:
            issues.append(f"Y-values far outside expected range [-5, 10]: {out_of_range}")
        
        return issues
    
    def _validate_forest_plot(self, figure: Dict[str, Any], structured: Dict) -> List[str]:
        """Validate forest plot specifically"""
        issues = []
        rows = structured.get("data", [])
        
        for row in rows:
            effect_size = row.get("effect_size")
            ci_lower = row.get("ci_lower")
            ci_upper = row.get("ci_upper")
            
            # Check CI bounds logic
            if all([effect_size, ci_lower, ci_upper]):
                if not (ci_lower <= effect_size <= ci_upper):
                    issues.append(
                        f"Invalid CI bounds for {row.get('study')}: "
                        f"ES={effect_size} outside [{ci_lower}, {ci_upper}]"
                    )
        
        return issues


# ── Text Cleaning Sub-Agent ───────────────────────────────────────────────

class TextCleaningAgent(SubAgent):
    """Cleans text encoding issues"""
    
    def __init__(self):
        super().__init__(
            name="Text Cleaning Agent",
            restricted_tools=["fix_encoding", "detect_corruption"]
        )
    
    def get_task(self) -> str:
        return """Clean extracted text:
1. Fix χ² (chi-square) symbol corruption
2. Fix minus sign encoding issues
3. Detect and fix special character problems
4. Validate text readability
5. Return cleaned, readable text"""
    
    def validate(self, text: str) -> Tuple[bool, str, List[str]]:
        """Clean text"""
        self.state.iteration += 1
        
        issues_found = []
        cleaned_text = text
        
        # Fix chi-square
        if re.search(r'x9[\s\n]?2', cleaned_text):
            cleaned_text = re.sub(r'x9\s*2\s*=', 'χ² =', cleaned_text)
            cleaned_text = re.sub(r'x9[\s\n]?2\b', 'χ²', cleaned_text)
            issues_found.append("chi_square_corruption_fixed")
        
        # Fix minus signs
        if '\u0004' in cleaned_text:
            cleaned_text = re.sub(r'\u00040+(\d+)', r'-\1', cleaned_text)
            cleaned_text = re.sub(r'\u0004(\d+\.\d+)', r'-\1', cleaned_text)
            issues_found.append("minus_sign_corruption_fixed")
        
        # Fix other control characters
        before_len = len(cleaned_text)
        cleaned_text = ''.join(c for c in cleaned_text if ord(c) >= 32 or c in '\n\t\r')
        if len(cleaned_text) < before_len:
            issues_found.append("control_characters_removed")
        
        # Validate text quality
        encoding_issues = self._detect_remaining_issues(cleaned_text)
        if encoding_issues:
            issues_found.extend(encoding_issues)
        
        is_valid = len([i for i in issues_found if "corruption" in i]) == 0
        
        self.state.results = {
            "original_length": len(text),
            "cleaned_length": len(cleaned_text),
            "fixes_applied": issues_found,
            "text_quality": "HIGH" if is_valid else "NEEDS_REVIEW"
        }
        self.state.issues_found = issues_found
        
        return is_valid, cleaned_text, issues_found
    
    def _detect_remaining_issues(self, text: str) -> List[str]:
        """Detect any remaining encoding issues"""
        issues = []
        
        # Check for high UTF-8 characters that shouldn't be there
        suspicious_chars = sum(1 for c in text if ord(c) > 127 and c not in 'χ†‡§¶†‡–—""''„»«')
        if suspicious_chars > 5:
            issues.append(f"possible_utf8_issues: {suspicious_chars} suspicious characters")
        
        # Check for repetitive corruption patterns
        if 'x9' in text or '\u0004' in text:
            issues.append("unresolved_encoding_patterns")
        
        return issues


# ── Metadata Structure Sub-Agent ──────────────────────────────────────────

class MetadataStructureAgent(SubAgent):
    """Structures and validates table metadata"""
    
    def __init__(self):
        super().__init__(
            name="Metadata Structure Agent",
            restricted_tools=["extract_metadata", "validate_structure"]
        )
    
    def get_task(self) -> str:
        return """Structure table metadata:
1. Extract footnotes from tables
2. Separate metadata from data
3. Validate table structure
4. Ensure clean data/metadata separation
5. Return properly structured tables with metadata"""
    
    def validate(self, tables: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Structure and validate tables"""
        self.state.iteration += 1
        
        structured_tables = []
        issues = []
        
        for table in tables:
            table_name = table.get("caption", "Unknown")
            
            # Extract footnotes
            clean_table, footnotes = self._extract_footnotes(table)
            
            # Add metadata
            clean_table["metadata"] = {
                "original_row_count": len(table.get("rows", [])),
                "data_row_count": len(clean_table.get("rows", [])),
                "footnotes": footnotes,
                "footnote_count": len(footnotes),
                "headers": clean_table.get("headers", [])
            }
            
            structured_tables.append(clean_table)
            
            # Check for issues
            if len(footnotes) > 0:
                issues.append(f"{table_name}: extracted {len(footnotes)} footnotes")
        
        result = {
            "tables_structured": len(structured_tables),
            "total_footnotes_extracted": sum(len(t["metadata"]["footnotes"]) for t in structured_tables),
            "structured_tables": structured_tables
        }
        
        self.state.results = result
        self.state.issues_found = issues
        
        is_valid = True  # Metadata extraction is always valid
        
        return is_valid, result, issues
    
    def _extract_footnotes(self, table: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Extract footnote rows from table"""
        rows = table.get("rows", [])
        footnotes = []
        data_rows = []
        
        for row in rows:
            if not row:
                continue
            
            first_cell = str(row[0]).strip()
            is_footnote = (
                first_cell.startswith("*") or
                first_cell.startswith("†") or
                first_cell.startswith("‡") or
                len(first_cell) > 80  # Long text often indicates footnote
            )
            
            if is_footnote:
                footnotes.append(first_cell)
            else:
                data_rows.append(row)
        
        clean_table = table.copy()
        clean_table["rows"] = data_rows
        
        return clean_table, footnotes


# ── Sub-Agent Orchestrator ────────────────────────────────────────────────

class SubAgentOrchestrator:
    """Manages all sub-agents and coordinates their work"""
    
    def __init__(self):
        self.figure_agent = FigureValidationAgent()
        self.text_agent = TextCleaningAgent()
        self.metadata_agent = MetadataStructureAgent()
        self.agents = [self.figure_agent, self.text_agent, self.metadata_agent]
    
    def validate_extraction(
        self,
        extraction: Dict[str, Any],
        table_2_smds: List[float] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate all sub-agents to validate and clean extraction.
        
        Returns validated extraction with quality report.
        """
        
        # Set reference data for figure agent
        if table_2_smds:
            self.figure_agent.set_reference_data(table_2_smds)
        
        # Run agents
        
        # 1. Text cleaning
        text_is_valid, cleaned_text, text_issues = self.text_agent.validate(
            extraction.get("text", "")
        )
        
        # 2. Figure validation
        fig_is_valid, fig_result, fig_issues = self.figure_agent.validate(
            extraction.get("figures", [])
        )
        
        # 3. Metadata structuring
        meta_is_valid, meta_result, meta_issues = self.metadata_agent.validate(
            extraction.get("tables", [])
        )
        
        # Compile results
        validated_extraction = {
            "page": extraction.get("page", 0),
            "text": cleaned_text,
            "tables": meta_result.get("structured_tables", []),
            "figures": fig_result.get("validated_figures", []),
            "sub_agent_validation": {
                "text_agent": {
                    "valid": text_is_valid,
                    "issues": text_issues,
                    "results": self.text_agent.state.results
                },
                "figure_agent": {
                    "valid": fig_is_valid,
                    "issues": fig_issues,
                    "results": self.figure_agent.state.results
                },
                "metadata_agent": {
                    "valid": meta_is_valid,
                    "issues": meta_issues,
                    "results": self.metadata_agent.state.results
                }
            },
            "overall_quality": self._compute_quality_score(
                text_is_valid, fig_is_valid, meta_is_valid,
                text_issues, fig_issues, meta_issues
            )
        }
        
        return validated_extraction
    
    def _compute_quality_score(
        self,
        text_valid: bool,
        fig_valid: bool,
        meta_valid: bool,
        text_issues: List[str],
        fig_issues: List[str],
        meta_issues: List[str]
    ) -> Dict[str, Any]:
        """Compute overall quality score"""
        score = 100
        
        # Deduct for issues
        if not text_valid:
            score -= 20
        if not fig_valid:
            score -= 30  # More important
        if not meta_valid:
            score -= 10
        
        # Deduct for each issue category
        score -= len(text_issues) * 2
        score -= len(fig_issues) * 3
        score -= len(meta_issues) * 1
        
        score = max(0, min(100, score))  # Clamp 0-100
        
        return {
            "score": score,
            "quality": "EXCELLENT" if score >= 90 else "GOOD" if score >= 75 else "ACCEPTABLE" if score >= 60 else "NEEDS_REVIEW",
            "text_quality": "CLEAN" if text_valid else "CORRUPTED",
            "figure_quality": "VALIDATED" if fig_valid else "FLAGGED",
            "metadata_quality": "STRUCTURED" if meta_valid else "UNSTRUCTURED"
        }


# ── Helper function ──────────────────────────────────────────────────────

import re  # Import needed for TextCleaningAgent


def validate_with_sub_agents(
    extraction: Dict[str, Any],
    table_2_smds: List[float] = None
) -> Dict[str, Any]:
    """
    Quick sub-agent validation wrapper for use in harness.
    
    Usage:
        # After Phase 3 extraction
        validated = validate_with_sub_agents(extraction, table_2_smds)
    """
    orchestrator = SubAgentOrchestrator()
    return orchestrator.validate_extraction(extraction, table_2_smds)