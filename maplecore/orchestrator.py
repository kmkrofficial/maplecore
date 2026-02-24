import re
import logging
from typing import Callable, Tuple

from maplecore.search import MapleScanner
from maplecore.indexer import MapleIndexer, Index
from maplecore.context_manager import ContextSynthesizer
from maplecore.prompt_templates import ORCHESTRATION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class RecursiveOrchestrator:
    def __init__(self, indexer: MapleIndexer, scanner: MapleScanner, max_hops: int = 3):
        """
        Orchestrates Search-Reason-Search (SRS) loops against a pre-built MAPLE Index.
        
        Args:
            indexer: Instance of MapleIndexer.
            scanner: Instance of MapleScanner.
            max_hops: Maximum number of recursive LLM searches.
        """
        self.indexer = indexer
        self.scanner = scanner
        self.max_hops = max_hops

    def generate_answer(
        self, 
        question: str, 
        index: Index, 
        llm_generator: Callable[[str], str],
        top_k: int = 5
    ) -> Tuple[str, int]:
        """
        Executes Search-Reason-Search loop.
        
        Args:
            question: Original evaluation question.
            index: MAPLE Index.
            llm_generator: Callable encapsulating the LLM API call.
            top_k: Chunks to fetch per hop.
            
        Returns:
            final_answer: The final textual answer.
            hops: Total number of recursive hops executed.
        """
        synthesizer = ContextSynthesizer(max_tokens=4000)
        
        # Initial search iteration
        res = self.scanner.search(self.indexer.encode_query(question), index, strategy="adaptive")
        k = min(top_k, len(res.block_ids))
        initial_chunks = [index.blocks[idx].text for idx in res.block_ids[:k]]
        synthesizer.add_chunks(initial_chunks, current_hop=0)
        
        hops = 0
        while hops <= self.max_hops:
            context_str = synthesizer.get_context_string()
            prompt = f"{ORCHESTRATION_SYSTEM_PROMPT}\n\nContext:\n{context_str}\n\nQuestion: {question}"
            
            # Request reasoning from LLM
            response_text = llm_generator(prompt)
            
            # Detect Orchestrator search commands: [SEARCH: "subquery"]
            match = re.search(r'\[SEARCH:\s*"([^"]+)"\]', response_text)
            if not match:
                # LLM believes it possesses the answer natively
                return response_text.strip(), hops
                
            sub_query = match.group(1)
            
            # Execute secondary MAPLE vector search against the existing context
            sub_res = self.scanner.search(self.indexer.encode_query(sub_query), index, strategy="adaptive")
            sk = min(top_k, len(sub_res.block_ids))
            new_chunks = [index.blocks[idx].text for idx in sub_res.block_ids[:sk]]
            
            # Attempt to splice sub-contexts uniquely into memory
            added = synthesizer.add_chunks(new_chunks, current_hop=hops + 1)
            
            if added == 0:
                # Fallback: Model requested a search, but the index had no new information
                # Prevent infinite loop by halting and forcing a fallback answer
                prompt_fallback = f"{ORCHESTRATION_SYSTEM_PROMPT}\n\nContext:\n{context_str}\n\nQuestion: {question}\n\nPREVIOUS SEARCH FAILED TO FIND NEW INFO. Provide the best partial answer or output 'Insufficient context'."
                final_response = llm_generator(prompt_fallback)
                return final_response.strip(), hops + 1
                
            hops += 1

        # Fallback if max hops limit is reached
        prompt_fallback = f"{ORCHESTRATION_SYSTEM_PROMPT}\n\nContext:\n{synthesizer.get_context_string()}\n\nQuestion: {question}\n\nMAX SEARCHES REACHED. Provide the best partial answer or output 'Insufficient context'."
        final_response = llm_generator(prompt_fallback)
        return final_response.strip(), hops
