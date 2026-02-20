#!/usr/bin/env python3
"""
Benchmark 06: E2E Generation Matrix
===================================
Tests MAPLE retrieved context against actual LLM generation accuracy.
Evaluates Gemini API models and local 4-bit Llama-3.

Tracks metrics on a Weights & Biases dashboard.
"""

import os
import gc
import json
import logging
import time
from typing import List, Dict

import torch
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from datasets import load_dataset
from maplecore import MapleIndexer, MapleNet, MapleScanner
from benchmarks.config import MODEL_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# LLMs to Test
GEMINI_MODELS = [
    "gemini-1.5-pro-latest",  # Will try to use 1.5 versions available
    "gemini-1.5-flash-latest"
]
LOCAL_LLM = "meta-llama/Meta-Llama-3-8B-Instruct"

def compute_rouge_score(prediction: str, references: List[str]) -> float:
    """Simple token-overlap Rogue-L estimation."""
    pred_tokens = set(prediction.lower().split())
    if not pred_tokens:
        return 0.0
        
    best_score = 0.0
    for ref in references:
        ref_tokens = set(ref.lower().split())
        if not ref_tokens:
            continue
        overlap = len(pred_tokens.intersection(ref_tokens))
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            best_score = max(best_score, f1)
            
    return best_score

def retrieve_contexts(num_samples: int = 50) -> List[Dict]:
    """Uses MAPLE to fetch Top-5 blocks for NarrativeQA questions."""
    logger.info(f"Loading {num_samples} NarrativeQA questions...")
    ds = load_dataset("deepmind/narrativeqa", split="test")
    
    # We will use the Oracle data to avoid re-indexing full books in this script.
    oracle_path = "data/oracle_data.json"
    if not os.path.exists(oracle_path):
        logger.error(f"Need {oracle_path} to provide block text corpuses.")
        return []
        
    with open(oracle_path, "r", encoding="utf-8") as f:
        oracle = json.load(f)
        
    samples = oracle["samples"][:num_samples]
    
    logger.info("Initializing MAPLE Pipeline...")
    indexer = MapleIndexer(device=DEVICE)
    if not MODEL_PATH.exists():
        logger.error(f"MAPLE Net not found at {MODEL_PATH}")
        return []
        
    maple_net = MapleNet.load(str(MODEL_PATH), device=DEVICE)
    scanner = MapleScanner(maple_net, device=DEVICE)
    
    eval_cache = []
    
    for i, s in enumerate(samples):
        question = s["question"]
        
        # Build synthetic index
        block_texts = list(s["all_block_texts"].values())
        if len(block_texts) < 5:
            continue
            
        index = indexer.create_index("\n\n".join(block_texts))
        
        # Search
        res = scanner.search(indexer.encode_query(question), index, strategy="adaptive")
        
        # Format Top 5
        context_blocks = "\n---\n".join([index.blocks[idx].text for idx in res.top_k[:5]])
        
        # Match question to Native NarrativeQA for answers
        gt_answers = []
        for item in ds:
            if item["question"]["text"] == question:
                gt_answers = [ans["text"] for ans in item["answers"]]
                break
                
        if not gt_answers:
            # Fallback if somehow not found (unlikely)
            gt_answers = ["Unknown"]
        
        eval_cache.append({
            "question": question,
            "context": context_blocks,
            "ground_truths": gt_answers
        })
        
        if (i+1) % 10 == 0:
            logger.info(f"  Retrieved {i+1}/{len(samples)}")
            
    # CRITICAL VRAM PURGE
    logger.info("Purging MAPLE from VRAM...")
    del scanner
    del maple_net
    del indexer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return eval_cache

def evaluate_gemini(eval_cache: List[Dict], run) -> None:
    if genai is None or not os.environ.get("GEMINI_API_KEY"):
        logger.warning("No GEMINI_API_KEY exported. Skipping Gemini Evaluations.")
        return
        
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    
    for model_name in GEMINI_MODELS:
        logger.info(f"Evaluating {model_name}...")
        try:
            model = genai.GenerativeModel(model_name)
        except Exception as e:
            logger.warning(f"Failed loading {model_name}: {e}")
            continue
            
        scores = []
        for i, item in enumerate(eval_cache):
            prompt = f"Context:\n{item['context']}\n\nQuestion: {item['question']}\nProvide a brief, direct answer based ONLY on the context:"
            
            try:
                response = model.generate_content(prompt)
                prediction = response.text
            except Exception as e:
                logger.warning(f"Gemini API rate limit/error: {e}")
                time.sleep(5)
                prediction = ""
                
            score = compute_rouge_score(prediction, item["ground_truths"])
            scores.append(score)
            
            time.sleep(2) # Limit hits
            
        avg_score = np.mean(scores) * 100
        logger.info(f"  {model_name} F1-Overlap: {avg_score:.1f}%")
        
        if run:
            wandb.log({f"e2e/{model_name}_f1": avg_score})

def evaluate_llama_local(eval_cache: List[Dict], run) -> None:
    logger.info(f"Evaluating {LOCAL_LLM} locally in 4-bit...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import transformers
    except ImportError:
        logger.warning("Missing transformers/bitsandbytes. Skipping local Llama.")
        return
        
    try:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLM,
            quantization_config=quant_config,
            device_map="auto"
        )
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=50
        )
    except Exception as e:
        logger.warning(f"Failed to load local Llama (requires access/hardware): {e}")
        return
        
    scores = []
    for i, item in enumerate(eval_cache):
        prompt = f"Context:\n{item['context']}\n\nQuestion: {item['question']}\nAnswer:"
        # Truncate prompt safely
        prompt = prompt[:3500] 
        
        try:
            out = pipeline(prompt, do_sample=False, return_full_text=False)
            prediction = out[0]["generated_text"].strip()
        except Exception as e:
            logger.warning(f"Llama inference failed: {e}")
            prediction = ""
            
        score = compute_rouge_score(prediction, item["ground_truths"])
        scores.append(score)
        
    avg_score = np.mean(scores) * 100
    logger.info(f"  {LOCAL_LLM} F1-Overlap: {avg_score:.1f}%")
    
    if run:
        wandb.log({f"e2e/llama3_8b_f1": avg_score})
        
    # CRITICAL VRAM PURGE
    logger.info("Purging Llama-3 from VRAM...")
    del pipeline
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    print("="*60)
    print("Benchmark 06: E2E Generation Metrics")
    print("="*60)
    
    run = None
    if wandb:
        run = wandb.init(project="maple-e2e-matrix", name="e2e_generation")
    
    eval_cache = retrieve_contexts(num_samples=50)
    if not eval_cache:
        logger.error("Failed to construct evaluation caching. Exiting.")
        if run: run.finish()
        return
        
    evaluate_gemini(eval_cache, run)
    evaluate_llama_local(eval_cache, run)
    
    if run:
        run.finish()
        
    print("\nBENCHMARK 06 COMPLETE")

if __name__ == "__main__":
    main()
