#!/usr/bin/env python3
"""
run_bge_upgrade.py
==================
Scout-KV Phase 6: BGE Embeddings Upgrade

Replaces noisy Llama mean-pooling with BAAI/bge-small-en-v1.5 (384 dim)
for precise retrieval.

Pipeline:
1. Re-generate training data with BGE embeddings
2. Train ScoutBGE model (384 -> 128 -> 1)
3. Validate on Sherlock Holmes needle test
"""

import sys
import os
import json
import time
import gc
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import requests

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


# =============================================================================
# ScoutBGE Model (384 dim input)
# =============================================================================

class ScoutBGE(nn.Module):
    """Scout model for BGE embeddings (384 dim)."""
    def __init__(self, input_dim=768, hidden_dim=128, dropout=0.3):
        # Input: query (384) + block (384) = 768
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


class ScoutDataset(Dataset):
    """Dataset for BGE embeddings."""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        combined = torch.cat([item["query_emb"], item["block_emb"]], dim=0)
        label = torch.tensor(item["label"], dtype=torch.float32)
        return combined, label, item["question"], item["block_id"]


def collate_fn(batch):
    embeddings = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    questions = [b[2] for b in batch]
    block_ids = [b[3] for b in batch]
    return embeddings, labels, questions, block_ids


# =============================================================================
# Step 1: Re-Generate Training Data with BGE
# =============================================================================

def regenerate_training_data(bge_model):
    """Re-generate training data using BGE embeddings."""
    print("\n" + "="*70)
    print("STEP 1: Re-Generate Training Data with BGE")
    print("="*70)
    
    # Load oracle data
    with open("oracle_data.json", "r", encoding="utf-8") as f:
        oracle_data = json.load(f)
    
    samples = oracle_data["samples"]
    print(f"Loaded {len(samples)} oracle samples")
    
    # Load NarrativeQA for original context
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    print("Loading NarrativeQA for context...")
    dataset = load_dataset("deepmind/narrativeqa", split="test", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    # Create lookup by question
    oracle_lookup = {s["question"]: s for s in samples}
    
    BLOCK_SIZE = 128
    MAX_TOKENS = 2048
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    
    training_data = []
    matched = 0
    
    print("Processing samples...")
    for i, ds_sample in enumerate(dataset):
        if matched >= len(samples):
            break
            
        try:
            context = ds_sample["document"]["text"]
            question = ds_sample["question"]["text"]
        except:
            continue
            
        if question not in oracle_lookup:
            continue
            
        oracle = oracle_lookup[question]
        top5_ids = set(oracle["top_5_block_ids"])
        
        # Truncate context like Phase 1
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        if len(context_tokens) > MAX_TOKENS:
            context_tokens = context_tokens[:MAX_TOKENS]
            context = tokenizer.decode(context_tokens, skip_special_tokens=True)
        
        # Chunk context into blocks
        num_blocks = (len(context_tokens) + BLOCK_SIZE - 1) // BLOCK_SIZE
        blocks = []
        for j in range(num_blocks):
            start = j * BLOCK_SIZE
            end = min((j + 1) * BLOCK_SIZE, len(context_tokens))
            block_tokens = context_tokens[start:end]
            block_text = tokenizer.decode(block_tokens, skip_special_tokens=True)
            blocks.append(block_text)
        
        # Encode with BGE
        # Query: with instruction prefix
        query_text = QUERY_PREFIX + question
        query_emb = bge_model.encode(query_text, convert_to_tensor=True, device="cuda")
        query_emb = query_emb.cpu().float()
        
        # Blocks: no prefix
        block_embs = bge_model.encode(blocks, convert_to_tensor=True, device="cuda", batch_size=32)
        block_embs = block_embs.cpu().float()
        
        # Create training examples
        for bid in range(len(blocks)):
            is_top5 = 1 if bid in top5_ids else 0
            training_data.append({
                "query_emb": query_emb,
                "block_emb": block_embs[bid],
                "label": is_top5,
                "question": question,
                "block_id": bid
            })
        
        matched += 1
        if matched % 10 == 0:
            print(f"  Processed {matched}/{len(samples)} samples")
    
    print(f"\n✓ Generated {len(training_data)} training examples")
    
    # Stats
    positive = sum(1 for ex in training_data if ex["label"] == 1)
    print(f"  Class balance: {positive} positive, {len(training_data)-positive} negative")
    
    return training_data


# =============================================================================
# Step 2: Train ScoutBGE
# =============================================================================

def compute_recall_at_k(model, val_loader, device, k=5):
    """Compute Recall@K for validation."""
    model.eval()
    question_preds = defaultdict(list)
    
    with torch.no_grad():
        for embs, labels, questions, block_ids in val_loader:
            embs = embs.to(device)
            logits = model(embs)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            for q, bid, prob, lbl in zip(questions, block_ids, probs, labels):
                question_preds[q].append((bid, float(prob), int(lbl.item())))
    
    recalls = []
    for q, preds in question_preds.items():
        preds.sort(key=lambda x: x[1], reverse=True)
        pred_top_k = set(p[0] for p in preds[:k])
        true_top_k = set(p[0] for p in preds if p[2] == 1)
        
        if len(true_top_k) > 0:
            hits = len(pred_top_k & true_top_k)
            recall = hits / min(len(true_top_k), k)
            recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0


def train_scout_bge(training_data, epochs=20, batch_size=32, lr=1e-4):
    """Train the ScoutBGE model."""
    print("\n" + "="*70)
    print("STEP 2: Train ScoutBGE")
    print("="*70)
    
    device = torch.device("cuda")
    
    # Split by question (80/20)
    questions = list(set(ex["question"] for ex in training_data))
    np.random.shuffle(questions)
    split = int(len(questions) * 0.8)
    train_q = set(questions[:split])
    val_q = set(questions[split:])
    
    train_data = [ex for ex in training_data if ex["question"] in train_q]
    val_data = [ex for ex in training_data if ex["question"] in val_q]
    
    print(f"Train: {len(train_data)} examples from {len(train_q)} questions")
    print(f"Val:   {len(val_data)} examples from {len(val_q)} questions")
    
    train_loader = DataLoader(ScoutDataset(train_data), batch_size=batch_size, 
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ScoutDataset(val_data), batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)
    
    # Model: 384 + 384 = 768 input
    model = ScoutBGE(input_dim=768, hidden_dim=128, dropout=0.3).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    
    best_recall = 0.0
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for embs, labels, _, _ in train_loader:
            embs, labels = embs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(embs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        val_recall = compute_recall_at_k(model, val_loader, device, k=5)
        
        if val_recall > best_recall:
            best_recall = val_recall
            torch.save(model.state_dict(), "scout_bge.pth")
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Val Recall@5: {val_recall*100:.1f}%")
    
    print(f"\n✓ Best Recall@5: {best_recall*100:.1f}%")
    print(f"✓ Model saved: scout_bge.pth")
    
    return best_recall


# =============================================================================
# Step 3: Sherlock Holmes Stress Test
# =============================================================================

def run_sherlock_test(bge_model):
    """Run the Sherlock Holmes needle test with BGE embeddings."""
    print("\n" + "="*70)
    print("STEP 3: Sherlock Holmes Stress Test")
    print("="*70)
    
    # Load book
    cache_file = Path("sherlock_holmes.txt")
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        url = "https://www.gutenberg.org/files/1661/1661-0.txt"
        print(f"Downloading from {url}...")
        response = requests.get(url, timeout=30)
        text = response.text
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(text)
    
    # Clean Gutenberg headers
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0
    if end != -1:
        text = text[start:end]
    else:
        text = text[start:]
    
    print(f"Book: {len(text):,} characters")
    
    # Chunk into blocks (use simple word-based for BGE)
    # For consistency, use ~500 chars per block (roughly 128 tokens)
    BLOCK_CHARS = 500
    blocks = []
    for i in range(0, len(text), BLOCK_CHARS):
        block_text = text[i:i+BLOCK_CHARS].strip()
        if block_text:
            blocks.append({"id": len(blocks), "text": block_text})
    
    print(f"Blocks: {len(blocks)}")
    
    # Encode all blocks with BGE
    print("Encoding blocks with BGE...")
    block_texts = [b["text"] for b in blocks]
    
    start_time = time.time()
    block_embs = bge_model.encode(block_texts, convert_to_tensor=True, 
                                  device="cuda", batch_size=32, show_progress_bar=True)
    block_embs = block_embs.cpu().float()
    encode_time = time.time() - start_time
    print(f"✓ Encoded {len(blocks)} blocks in {encode_time:.2f}s")
    
    # Query
    QUESTION = "What was the speckled band?"
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    
    query_emb = bge_model.encode(QUERY_PREFIX + QUESTION, convert_to_tensor=True, device="cuda")
    query_emb = query_emb.cpu().float()
    
    # Load ScoutBGE
    device = torch.device("cuda")
    scout = ScoutBGE(input_dim=768, hidden_dim=128, dropout=0.3).to(device)
    scout.load_state_dict(torch.load("scout_bge.pth", weights_only=True))
    scout.eval()
    
    # Score all blocks
    print("Running ScoutBGE inference...")
    start_time = time.time()
    
    scores = []
    with torch.no_grad():
        for block_emb in block_embs:
            combined = torch.cat([query_emb, block_emb], dim=0).unsqueeze(0).to(device)
            logit = scout(combined)
            scores.append(torch.sigmoid(logit).item())
    
    scout_time = time.time() - start_time
    print(f"✓ Scout inference: {scout_time*1000:.2f}ms for {len(blocks)} blocks")
    
    # Top-5 blocks
    indexed = [(i, s) for i, s in enumerate(scores)]
    indexed.sort(key=lambda x: x[1], reverse=True)
    top5_ids = [idx for idx, _ in indexed[:5]]
    top5_ids.sort()
    
    print(f"\nTop-5 Block IDs: {top5_ids}")
    print("\nBlock Previews:")
    for bid in top5_ids:
        preview = blocks[bid]["text"][:80].replace("\n", " ")
        print(f"  Block {bid}: \"{preview}...\"")
    
    # Clear VRAM for Llama
    del block_embs, query_emb, scout
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load Llama-3 for generation
    print("\n" + "-"*70)
    print("Loading Llama-3 for answer generation...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config,
        device_map="cuda:0", trust_remote_code=True, torch_dtype=torch.float16
    )
    model.eval()
    
    # Generate answer
    context = "\n\n".join(blocks[bid]["text"] for bid in top5_ids)
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {QUESTION}

Answer concisely.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    print(f"\n{'='*70}")
    print("SHERLOCK RESULTS")
    print(f"{'='*70}")
    print(f"Query:        {QUESTION}")
    print(f"Expected:     A swamp adder (snake)")
    print(f"Scout Blocks: {top5_ids}")
    print(f"Scout Answer: {answer}")
    
    if "adder" in answer.lower() or "snake" in answer.lower():
        print("\n✓ SUCCESS! Scout-BGE found the needle!")
    else:
        print("\n⚠ Partial match (check context)")
    
    return answer, top5_ids


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("Scout-KV Phase 6: BGE Embeddings Upgrade")
    print("="*70)
    
    # Load BGE model
    print("\nLoading BGE model: BAAI/bge-small-en-v1.5")
    from sentence_transformers import SentenceTransformer
    bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    print(f"✓ BGE loaded (384 dim)")
    
    # Step 1: Re-generate training data
    training_data = regenerate_training_data(bge_model)
    
    # Step 2: Train ScoutBGE
    np.random.seed(42)
    torch.manual_seed(42)
    best_recall = train_scout_bge(training_data)
    
    # Step 3: Sherlock test
    answer, top5 = run_sherlock_test(bge_model)
    
    print("\n" + "="*70)
    print("PHASE 6 COMPLETE")
    print("="*70)
    print(f"Training Recall@5: {best_recall*100:.1f}%")
    print(f"Sherlock Answer:   {answer[:100]}...")
    print("="*70)


if __name__ == "__main__":
    main()
