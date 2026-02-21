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

def evaluate_with_llm_judge(judge_model, generated_answer: str, ground_truth_list: list) -> int:
    """Uses a dedicated LLM to semantically judge the correctness of the answer."""
    prompt = f"""You are an expert evaluator grading a question-answering system.
Determine if the 'Generated Answer' is factually correct and semantically matches ANY of the acceptable 'Ground Truths'.
Ignore minor differences in grammar, punctuation, or extra conversational filler.

Ground Truths: {ground_truth_list}
Generated Answer: {generated_answer}

Respond with EXACTLY and ONLY the word "YES" if the answer is correct, or "NO" if it is incorrect."""

    try:
        verdict = judge_model.generate_content(prompt)
        try:
            verdict_text = verdict.text.upper()
            return 1 if "YES" in verdict_text else 0
        except ValueError:
            return 0
    except Exception as e:
        logger.warning(f"Judge API error: {e}")
        time.sleep(2) # Backoff for the judge
        return 0



def retrieve_contexts(num_samples: int = 50) -> List[Dict]:
    """Uses MAPLE to fetch Top-5 blocks for NarrativeQA questions."""
    logger.info(f"Loading {num_samples} NarrativeQA questions...")
    ds = load_dataset("deepmind/narrativeqa", split="train")
    
    # We will use the Oracle data to avoid re-indexing full books in this script.
    oracle_path = "data/oracle_data.json"
    if not os.path.exists(oracle_path):
        logger.error(f"Need {oracle_path} to provide block text corpuses.")
        return []
        
    with open(oracle_path, "r", encoding="utf-8") as f:
        oracle = json.load(f)
        
    samples = oracle["samples"][:num_samples]
    
    eval_cache = []
    
    for i, s in enumerate(samples):
        question = s["question"]
        
        # Directly extract the 5 answer-bearing blocks from the oracle data
        oracle_ids = s.get("top_5_block_ids", [])
        if not oracle_ids:
            oracle_ids = list(s["all_block_texts"].keys())[:5]
            
        block_text_dict = s["all_block_texts"]
        context_blocks = "\n---\n".join([block_text_dict[str(idx)] for idx in oracle_ids if str(idx) in block_text_dict])
        
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
            
    return eval_cache

def _process_gemini_sample(item: Dict, model, delay: float) -> dict:
    prompt = f"Context:\n{item['context']}\n\nQuestion: {item['question']}\nUsing the provided context, answer the question accurately and concisely. If the context does not contain the answer, output exactly 'Not found'."
    
    start_time = time.time()
    try:
        response = model.generate_content(prompt)
        try:
            prediction = response.text
        except ValueError:
            # Safe fallback for empty parts where finish_reason is 1 (STOP)
            prediction = ""
    except Exception as e:
        logger.warning(f"Gemini API error: {e}")
        time.sleep(5)
        prediction = ""
        
    latency = time.time() - start_time
    
    if delay > 0:
        time.sleep(delay)
        
    baseline_judge = genai.GenerativeModel('gemini-flash-lite-latest')
    score = evaluate_with_llm_judge(baseline_judge, prediction, item["ground_truths"])
    
    return {
        "question": item["question"],
        "prediction": prediction,
        "ground_truths": item["ground_truths"],
        "accuracy": score,
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
        
    avg_accuracy = np.mean([r["accuracy"] for r in results]) * 100
    avg_latency = np.mean([r["latency_sec"] for r in results])
    
    logger.info(f"  {model_name} - Accuracy: {avg_accuracy:.1f}%, Latency: {avg_latency:.2f}s")
    
    if run:
        wandb.log({
            f"{model_name}/accuracy": avg_accuracy,
            f"{model_name}/latency_sec": avg_latency
        })
        
    return {
        "model": model_name,
        "metrics": {
            "accuracy": avg_accuracy,
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
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
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
        
        gen_config = GenerationConfig(
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            generation_config=gen_config,
        )
    except Exception as e:
        logger.warning(f"Failed to load local Llama (requires access/hardware): {e}")
        return {}
        
    results = []
    for i, item in enumerate(eval_cache):
        prompt = f"Context:\n{item['context']}\n\nQuestion: {item['question']}\nUsing the provided context, answer the question accurately and concisely. If the context does not contain the answer, output exactly 'Not found'."
        # Truncate prompt safely
        prompt = prompt[:3500] 
        
        start_time = time.time()
        try:
            out = pipeline(prompt, return_full_text=False)
            prediction = out[0]["generated_text"].strip()
        except Exception as e:
            logger.warning(f"Llama inference failed: {e}")
            prediction = ""
            
        latency = time.time() - start_time
        
        baseline_judge = genai.GenerativeModel('gemini-flash-lite-latest')
        score = evaluate_with_llm_judge(baseline_judge, prediction, item["ground_truths"])
        
        results.append({
            "question": item["question"],
            "prediction": prediction,
            "ground_truths": item["ground_truths"],
            "accuracy": score,
            "latency_sec": latency
        })
        
    if not results:
        return {}
        
    avg_accuracy = np.mean([r["accuracy"] for r in results]) * 100
    avg_latency = np.mean([r["latency_sec"] for r in results])
    
    logger.info(f"  {LOCAL_LLM} - Accuracy: {avg_accuracy:.1f}%, Latency: {avg_latency:.2f}s")
    
    if run:
        wandb.log({
            f"llama3_8b/accuracy": avg_accuracy,
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
            "accuracy": avg_accuracy,
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
