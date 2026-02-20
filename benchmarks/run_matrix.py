#!/usr/bin/env python3
"""
MAPLE W&B Benchmark Matrix
==========================
Automates training and recall evaluation across:
 - BAAI/bge-small-en-v1.5
 - sentence-transformers/all-MiniLM-L6-v2
 - nomic-ai/nomic-embed-text-v1.5

Logs all metrics and output charts to Weights & Biases.
"""

import os
import sys
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Insert project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import wandb
except ImportError:
    print("wandb not found. Please install with `pip install maplecore[benchmarks]`")
    sys.exit(1)

import importlib

from scripts.train_maple import run as run_train
recall_module = importlib.import_module("benchmarks.01_recall_enhanced")
run_recall = recall_module.run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_TO_TEST = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "nomic-ai/nomic-embed-text-v1.5"
]

def format_shortname(name: str) -> str:
    return name.split("/")[-1].replace("-", "_").lower()

def run_matrix():
    print(f"Starting W&B E2E Model Matrix on {len(MODELS_TO_TEST)} Encoders...")
    
    for embed_model in MODELS_TO_TEST:
        shortname = format_shortname(embed_model)
        output_model = Path(f"models/maple_{shortname}.pth")
        
        # Initialize W&B Run
        run = wandb.init(
            project="maplecore",
            name=f"recall_{shortname}",
            config={
                "embedding_model": embed_model,
                "epochs": 20,
                "batch_size": 32,
            },
            reinit=True
        )
        
        logger.info(f"\n{'='*60}\nEvaluating: {embed_model}\n{'='*60}")
        
        try:
            # 1. Train MAPLE for this specific embedding model
            run_train(
                epochs=20,
                batch_size=32,
                embedding_model=embed_model,
                output_model_path=output_model
            )
            
            # 2. Run Recall Evaluation
            output_data = run_recall(
                max_samples=200,
                embedding_model=embed_model,
                model_path=output_model
            )
            
            if not output_data:
                logger.warning(f"No output data for {embed_model}, skipping metrics...")
                continue
                
            # 3. Log to W&B
            log_dict = {}
            for ds_name, ds_stats in output_data.items():
                for method, k_vals in ds_stats["metrics"]["recall@5"].items():
                    log_dict[f"{ds_name}/recall@5/{method}"] = k_vals
            
            wandb.log(log_dict)
            
            # Log chart artifact
            chart_path = "benchmarks/results/recall_grouped_bar.png"
            if os.path.exists(chart_path):
                metrics_artifact = wandb.Artifact(f"recall_chart_{shortname}", type="chart")
                metrics_artifact.add_file(chart_path)
                run.log_artifact(metrics_artifact)
                
            logger.info(f"Matrix run for {embed_model} complete!")
                
        except Exception as e:
            logger.error(f"Failed handling {embed_model}: {e}")
            
        finally:
            run.finish()

if __name__ == "__main__":
    run_matrix()
