import json
from datetime import datetime
from pathlib import Path
from config.settings import SESSION_DIR


class SessionLogger:
    """Component #6 - Session Persistence (append-only JSONL with metadata tracking)"""

    def __init__(self, pdf_name: str):
        safe_name = pdf_name.replace('.pdf', '').replace(' ', '_').lower()
        self.session_file = SESSION_DIR / f"{safe_name}_session.jsonl"
        
        # Metadata tracking for QA
        self.metadata = {
            "pdf_name": pdf_name,
            "session_started": datetime.now().isoformat(),
            "pages_extracted": [],
            "zero_table_pages": [],
            "extraction_status": {}
        }

    def log(self, pdf: str, iteration: int, action: str,
            params: dict = None, result: dict = None):
        """Log an event to the session file"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "pdf"      : pdf,
            "iteration": iteration,
            "action"   : action,
            "params"   : params or {},
            "result"   : result or {}
        }
        
        # Track metadata for QA
        self._update_metadata(action, event)
        
        with open(self.session_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event) + '\n')

    def _update_metadata(self, action: str, event: dict):
        """Update metadata based on extraction results"""
        result = event.get('result', {})
        
        # Track extracted pages
        if action == 'vision_extract_page':
            page = result.get('page')
            status = result.get('status', 'unknown')
            
            if page:
                self.metadata['pages_extracted'].append({
                    'page': page,
                    'status': status,
                    'tables_found': len(result.get('tables', [])),
                    'figures_found': len(result.get('figures', []))
                })
                
                # Flag zero-table pages for QA
                if not result.get('tables') and not result.get('figures'):
                    self.metadata['zero_table_pages'].append(page)
            
            # Track status
            self.metadata['extraction_status'][f'page_{page}'] = status
        
        # Track finish status
        elif action == 'finish':
            self.metadata['session_completed'] = datetime.now().isoformat()
            self.metadata['final_status'] = result.get('status', 'unknown')

    def get_metadata(self) -> dict:
        """Return accumulated metadata for QA/reporting"""
        return {
            **self.metadata,
            "total_pages_extracted": len(self.metadata['pages_extracted']),
            "zero_table_page_count": len(self.metadata['zero_table_pages']),
            "session_file": str(self.session_file)
        }

    def replay(self) -> list:
        """Replay all logged events from this session"""
        if not self.session_file.exists():
            return []
        events = []
        with open(self.session_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def get_path(self) -> str:
        """Return path to session file"""
        return str(self.session_file)

    def save_metadata_summary(self) -> str:
        """Save metadata summary as a separate JSON file for easy QA review"""
        summary_file = self.session_file.parent / f"{self.session_file.stem}_metadata.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_metadata(), f, indent=2)
        return str(summary_file)