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



def load_and_prepare_dataset(dataset_name: str, num_samples: int = 50):
    if dataset_name == "squad":
        ds = load_dataset('squad', split='validation')
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples: break
            samples.append({
                "question": item["question"],
                "context": item["context"],
                "ground_truths": item["answers"]["text"]
            })
        return samples
    elif dataset_name == "hotpot_qa":
        ds = load_dataset('hotpot_qa', 'distractor', split='validation')
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples: break
            context_str = " ".join([" ".join(sents) for sents in item["context"]["sentences"]])
            samples.append({
                "question": item["question"],
                "context": context_str,
                "ground_truths": [item["answer"]]
            })
        return samples
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def retrieve_contexts(dataset_name: str, num_samples: int = 50) -> List[Dict]:
    """Uses MAPLE to fetch Top-5 blocks for the given dataset."""
    logger.info(f"Loading {num_samples} {dataset_name} questions...")
    raw_samples = load_and_prepare_dataset(dataset_name, num_samples)
    
    logger.info("Initializing MAPLE Pipeline...")
    indexer = MapleIndexer(device=DEVICE)
    
    # OVERRIDE: Prevent Crossed Latent Space by explicitly loading BGE weights
    bge_weights = "models/maple_bge_small_en_v1.5.pth"
    if not os.path.exists(bge_weights):
        logger.error(f"MAPLE Net not found at {bge_weights}")
        return []
        
    maple_net = MapleNet.load(bge_weights, device=DEVICE)
    scanner = MapleScanner(maple_net, device=DEVICE)
    
    eval_cache = []
    
    for i, s in enumerate(raw_samples):
        question = s["question"]
        raw_context = s["context"]
        gt_answers = s["ground_truths"]
        
        # Build index for this specific context
        index = indexer.create_index(raw_context)
        
        if index.num_blocks == 0:
            continue
            
        # Search
        res = scanner.search(indexer.encode_query(question), index, strategy="adaptive")
        
        # Extract Top-5 blocks text
        top_k = min(5, len(res.block_ids))
        context_blocks = "\n---\n".join([index.blocks[idx].text for idx in res.block_ids[:top_k]])
        
        eval_cache.append({
            "question": question,
            "context": context_blocks,
            "ground_truths": gt_answers
        })
        
        if (i+1) % 10 == 0:
            logger.info(f"  Retrieved {i+1}/{len(raw_samples)}")
            
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

def evaluate_gemini_model(eval_cache: List[Dict], model_name: str, dataset_name: str, run) -> dict:
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
            f"{dataset_name}/{model_name}/accuracy": avg_accuracy,
            f"{dataset_name}/{model_name}/latency_sec": avg_latency
        })
        
    return {
        "model": model_name,
        "metrics": {
            "accuracy": avg_accuracy,
            "latency_sec": avg_latency
        },
        "samples": results
    }

def evaluate_all_gemini(eval_cache: List[Dict], dataset_name: str, run) -> List[dict]:
    if genai is None or not os.environ.get("GEMINI_API_KEY"):
        logger.warning("No GEMINI_API_KEY exported. Skipping Gemini Evaluations.")
        return []
        
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        
    all_reports = []
    # Launch all Gemini models in parallel using another ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GEMINI_MODELS)) as top_executor:
        futures = [top_executor.submit(evaluate_gemini_model, eval_cache, model_name, dataset_name, run) for model_name in GEMINI_MODELS]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                all_reports.append(res)
                
    return all_reports

def evaluate_llama_local(eval_cache: List[Dict], dataset_name: str, run) -> dict:
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
            f"{dataset_name}/llama3_8b/accuracy": avg_accuracy,
            f"{dataset_name}/llama3_8b/latency_sec": avg_latency
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
    TARGET_DATASETS = ["squad", "hotpot_qa"]
    
    run = None
    if wandb:
        run = wandb.init(
            project="maplecore", 
            name=f"e2e_{timestamp}", 
            group="e2e_evaluation",
            tags=["generation"] + TARGET_DATASETS
        )
        
    all_reports = {}
    
    for dataset_name in TARGET_DATASETS:
        print(f"\n--- Evaluating Dataset: {dataset_name} ---")
        eval_cache = retrieve_contexts(dataset_name=dataset_name, num_samples=50)
        if not eval_cache:
            logger.error(f"Failed to construct evaluation caching for {dataset_name}. Skipping.")
            continue
            
        dataset_reports = []
        
        # Run Gemini concurrently
        gemini_reports = evaluate_all_gemini(eval_cache, dataset_name, run)
        dataset_reports.extend(gemini_reports)
        
        # Run Llama sequentially
        llama_report = evaluate_llama_local(eval_cache, dataset_name, run)
        if llama_report:
            dataset_reports.append(llama_report)
            
        all_reports[dataset_name] = dataset_reports
        
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
