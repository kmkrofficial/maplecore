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
import string
import concurrent.futures
from typing import List, Dict

from dotenv import load_dotenv

load_dotenv()

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
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-flash-lite-latest"
]
LOCAL_LLM = "meta-llama/Meta-Llama-3-8B-Instruct"

API_CONFIG = {
    "gemini-3.1-pro-preview": {"max_workers": 1, "delay": 2.5}, # 25 RPM
    "gemini-3-flash-preview": {"max_workers": 10, "delay": 0},  # 1000 RPM
    "gemini-flash-lite-latest": {"max_workers": 10, "delay": 0} # 1000 RPM
}

def normalize_text(text: str) -> str:
    """Normalize text for evaluation: lowercase, remove punctuation and articles."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in {"a", "an", "the"}]
    return " ".join(tokens)

def compute_rouge_score(prediction: str, references: List[str]) -> Dict[str, float]:
    """Simple token-overlap Rogue-L estimation returning precision, recall, f1."""
    norm_pred = normalize_text(prediction)
    pred_tokens = set(norm_pred.split())
    
    if not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    
    for ref in references:
        norm_ref = normalize_text(ref)
        ref_tokens = set(norm_ref.split())
        if not ref_tokens:
            continue
            
        overlap = len(pred_tokens.intersection(ref_tokens))
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            
    return {"precision": best_precision, "recall": best_recall, "f1_score": best_f1}

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
        
        # Extract Top-5 blocks text
        context_blocks = "\n---\n".join([index.blocks[idx].text for idx in res.block_ids[:5]])
        
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

def _process_gemini_sample(item: Dict, model, delay: float) -> dict:
    prompt = f"Context:\n{item['context']}\n\nQuestion: {item['question']}\nYou are a strict data extraction system. Answer using ONLY the exact words from the context. Do NOT write full sentences. Do NOT say 'The answer is...'. If the answer is 'John', output 'John'."
    
    start_time = time.time()
    try:
        response = model.generate_content(prompt)
        prediction = response.text
    except Exception as e:
        logger.warning(f"Gemini API error: {e}")
        time.sleep(5)
        prediction = ""
        
    latency = time.time() - start_time
    
    if delay > 0:
        time.sleep(delay)
        
    scores = compute_rouge_score(prediction, item["ground_truths"])
    
    return {
        "question": item["question"],
        "prediction": prediction,
        "ground_truths": item["ground_truths"],
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1_score": scores["f1_score"],
        "latency_sec": latency
    }

def evaluate_gemini_model(eval_cache: List[Dict], model_name: str, run) -> dict:
    logger.info(f"Starting evaluations for {model_name}...")
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        logger.warning(f"Failed loading {model_name}: {e}")
        return {}

    config = API_CONFIG.get(model_name, {"max_workers": 1, "delay": 2.5})
    max_workers = config["max_workers"]
    delay = config["delay"]
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_gemini_sample, item, model, delay) for item in eval_cache]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            
    if not results:
        return {}
        
    avg_precision = np.mean([r["precision"] for r in results]) * 100
    avg_recall = np.mean([r["recall"] for r in results]) * 100
    avg_f1 = np.mean([r["f1_score"] for r in results]) * 100
    avg_latency = np.mean([r["latency_sec"] for r in results])
    
    logger.info(f"  {model_name} - F1: {avg_f1:.1f}%, Prec: {avg_precision:.1f}%, Rec: {avg_recall:.1f}%, Latency: {avg_latency:.2f}s")
    
    if run:
        wandb.log({
            f"{model_name}/f1_score": avg_f1,
            f"{model_name}/precision": avg_precision,
            f"{model_name}/recall": avg_recall,
            f"{model_name}/latency_sec": avg_latency
        })
        
    return {
        "model": model_name,
        "metrics": {
            "f1_score": avg_f1,
            "precision": avg_precision,
            "recall": avg_recall,
            "latency_sec": avg_latency
        },
        "samples": results
    }

def evaluate_all_gemini(eval_cache: List[Dict], run) -> List[dict]:
    if genai is None or not os.environ.get("GEMINI_API_KEY"):
        logger.warning("No GEMINI_API_KEY exported. Skipping Gemini Evaluations.")
        return []
        
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        
    all_reports = []
    # Launch all Gemini models in parallel using another ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GEMINI_MODELS)) as top_executor:
        futures = [top_executor.submit(evaluate_gemini_model, eval_cache, model_name, run) for model_name in GEMINI_MODELS]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                all_reports.append(res)
                
    return all_reports

def evaluate_llama_local(eval_cache: List[Dict], run) -> dict:
    logger.info(f"Evaluating {LOCAL_LLM} locally in 4-bit...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import transformers
    except ImportError:
        logger.warning("Missing transformers/bitsandbytes. Skipping local Llama.")
        return {}
        
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
        return {}
        
    results = []
    for i, item in enumerate(eval_cache):
        prompt = f"Context:\n{item['context']}\n\nQuestion: {item['question']}\nYou are a strict data extraction system. Answer using ONLY the exact words from the context. Do NOT write full sentences. Do NOT say 'The answer is...'. If the answer is 'John', output 'John'."
        # Truncate prompt safely
        prompt = prompt[:3500] 
        
        start_time = time.time()
        try:
            out = pipeline(prompt, do_sample=False, return_full_text=False)
            prediction = out[0]["generated_text"].strip()
        except Exception as e:
            logger.warning(f"Llama inference failed: {e}")
            prediction = ""
            
        latency = time.time() - start_time
        scores = compute_rouge_score(prediction, item["ground_truths"])
        
        results.append({
            "question": item["question"],
            "prediction": prediction,
            "ground_truths": item["ground_truths"],
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1_score": scores["f1_score"],
            "latency_sec": latency
        })
        
    if not results:
        return {}
        
    avg_precision = np.mean([r["precision"] for r in results]) * 100
    avg_recall = np.mean([r["recall"] for r in results]) * 100
    avg_f1 = np.mean([r["f1_score"] for r in results]) * 100
    avg_latency = np.mean([r["latency_sec"] for r in results])
    
    logger.info(f"  {LOCAL_LLM} - F1: {avg_f1:.1f}%, Prec: {avg_precision:.1f}%, Rec: {avg_recall:.1f}%, Latency: {avg_latency:.2f}s")
    
    if run:
        wandb.log({
            f"llama3_8b/f1_score": avg_f1,
            f"llama3_8b/precision": avg_precision,
            f"llama3_8b/recall": avg_recall,
            f"llama3_8b/latency_sec": avg_latency
        })
        
    # CRITICAL VRAM PURGE
    logger.info("Purging Llama-3 from VRAM...")
    del pipeline
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": LOCAL_LLM,
        "metrics": {
            "f1_score": avg_f1,
            "precision": avg_precision,
            "recall": avg_recall,
            "latency_sec": avg_latency
        },
        "samples": results
    }

def main():
    print("="*60)
    print("Benchmark 06: E2E Generation Metrics")
    print("="*60)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    run = None
    if wandb:
        run = wandb.init(
            project="maplecore", 
            name=f"e2e_{timestamp}", 
            group="e2e_evaluation",
            tags=["generation"]
        )
    
    eval_cache = retrieve_contexts(num_samples=50)
    if not eval_cache:
        logger.error("Failed to construct evaluation caching. Exiting.")
        if run: run.finish()
        return
        
    all_reports = []
    
    # Run Gemini concurrently
    gemini_reports = evaluate_all_gemini(eval_cache, run)
    all_reports.extend(gemini_reports)
    
    # Run Llama sequentially
    llama_report = evaluate_llama_local(eval_cache, run)
    if llama_report:
        all_reports.append(llama_report)
        
    if run:
        run.finish()
        
    # JSON Dump
    os.makedirs("results", exist_ok=True)
    report_path = f"results/e2e_metrics_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=4)
        
    logger.info(f"Saved local report to {report_path}")
    print("\nBENCHMARK 06 COMPLETE")

if __name__ == "__main__":
    main()
