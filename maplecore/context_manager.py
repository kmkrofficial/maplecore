"""
Context Management for MAPLE Orchestration
"""

from typing import List, Dict, Set

class ContextSynthesizer:
    def __init__(self, max_tokens: int = 4000, chars_per_token: float = 4.0):
        """
        Synthesizes and manages context chunks across multiple reasoning hops.
        
        Args:
            max_tokens: Maximum allowed tokens in the context window.
            chars_per_token: Heuristic for character-to-token conversion.
        """
        self.max_chars = int(max_tokens * chars_per_token)
        # Store chunks as a list of dicts to maintain order and metadata
        # e.g., [{"text": "...", "id": 123, "hop": 0}, ...]
        self.context_chunks: List[Dict] = []
        self.seen_hashes: Set[int] = set()

    def add_chunks(self, chunks: List[str], current_hop: int = 0) -> int:
        """
        Adds new chunks to the context window, deduplicating and sliding the window.
        Returns the number of *new* chunks successfully added.
        """
        added = 0
        for chunk in chunks:
            chunk_hash = hash(chunk)
            if chunk_hash not in self.seen_hashes:
                self.seen_hashes.add(chunk_hash)
                self.context_chunks.append({
                    "text": chunk,
                    "hop": current_hop
                })
                added += 1
                
        self._enforce_sliding_window()
        return added

    def _enforce_sliding_window(self):
        """
        Prunes the oldest/least relevant chunks if the context exceeds max_chars.
        Retains the most recently added chunks (highest hop) and initial core chunks if possible,
        but prefers evicting older hops to make room.
        For simplicity, we'll evict starting from hop 0 (assuming newer hops are more targeted),
        or just FIFO if we hit limits.
        """
        current_chars = sum(len(c["text"]) for c in self.context_chunks)
        
        if current_chars <= self.max_chars:
            return
            
        # If we need to evict, we remove from the beginning (oldest chunks)
        # In a more advanced implementation, we might score relevance.
        while current_chars > self.max_chars and self.context_chunks:
            evicted = self.context_chunks.pop(0)
            self.seen_hashes.remove(hash(evicted["text"]))
            current_chars -= len(evicted["text"])

    def get_context_string(self) -> str:
        """
        Returns the synthesized context as a single formatted string.
        """
        if not self.context_chunks:
            return "No context available."
            
        return "\n---\n".join([c["text"] for c in self.context_chunks])

    def clear(self):
        self.context_chunks.clear()
        self.seen_hashes.clear()
