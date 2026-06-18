from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set


@dataclass
class TableData:
    name: str
    page: int
    rows: int
    cols: int
    data: List[List[str]]


@dataclass
class FigureData:
    name: str
    page: int
    description: str
    figure_type: str = "unknown"


@dataclass
class HarnessState:
    pdf_path: str
    pdf_name: str
    iteration: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)
    text_chunks: List[str] = field(default_factory=list)
    tables: List[Dict] = field(default_factory=list)
    figures: List[Dict] = field(default_factory=list)
    is_done: bool = False
    current_page: int = 1
    total_pages: int = 0
    error_count: int = 0
    extraction_plan: List[Dict[str, Any]] = field(default_factory=list)
    extracted_pages: Set[int] = field(default_factory=set)
    goal: str = "extract_all_data"  # ← ADD THIS: User-informed goal
    
    # ← NEW: Phase 4 validation fields
    validation_results: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0

    def to_output_dict(self) -> Dict[str, Any]:
        from datetime import datetime
        return {
            "pdf"            : self.pdf_name,
            "extraction_date": datetime.now().isoformat(),
            "total_pages"    : self.total_pages,
            "text"           : self.text_chunks,
            "tables"         : self.tables,
            "figures"        : self.figures,
            "total_iterations": self.iteration,
            "goal"           : self.goal,
            "status"         : "complete" if self.is_done else "incomplete"
        }


@dataclass
class ToolResult:
    status: str
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""


@dataclass
class AgentDecision:
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""