from typing import List, Dict
from config.settings import HARNESS_CONTEXT_LIMIT, HARNESS_COMPACTION_THRESHOLD


class ContextManager:
    """Component #2 - Context Management (token budgeting & compaction)"""

    def __init__(self):
        self.limit     = HARNESS_CONTEXT_LIMIT
        self.threshold = HARNESS_COMPACTION_THRESHOLD

    def _count_tokens(self, messages: List[Dict]) -> int:
        """Rough token count: 1 token ≈ 4 chars"""
        total_chars = sum(len(str(m.get('content', ''))) for m in messages)
        return total_chars // 4

    def should_compact(self, messages: List[Dict]) -> bool:
        return self._count_tokens(messages) > self.threshold

    def compact(self, messages: List[Dict]) -> List[Dict]:
        """Keep last 10 messages verbatim, summarize the rest into one message"""
        if len(messages) <= 10:
            return messages

        old_messages  = messages[:-10]
        keep_messages = messages[-10:]

        # Build a summary of old messages
        summary_parts = []
        for m in old_messages:
            role    = m.get('role', 'unknown')
            content = str(m.get('content', ''))[:200]  # first 200 chars each
            summary_parts.append(f"{role}: {content}")

        summary = (
            "[CONTEXT COMPACTED - Summary of earlier conversation]\n"
            + "\n".join(summary_parts)
        )

        compacted_message = {"role": "user", "content": summary}
        print(f"  [Context] Compacted {len(old_messages)} messages → 1 summary message")
        return [compacted_message] + keep_messages

    def manage(self, messages: List[Dict]) -> List[Dict]:
        """Call this every iteration to keep context under control"""
        if self.should_compact(messages):
            messages = self.compact(messages)
        return messages
