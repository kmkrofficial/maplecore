# Scout-KV: Phase 1 - Sparsity Hypothesis Validation

This directory contains scripts for validating the **Sparsity Hypothesis** from the paper "Scout-KV: Speculative Paging for Infinite Context."

## Overview

The Sparsity Hypothesis states that for a given question about a long document, the model mostly attends to a few specific "Hot Blocks" of text, not the entire document.

## Hardware Requirements

- **GPU:** NVIDIA RTX 4090 Mobile (16GB VRAM) or equivalent
- **RAM:** 32GB System RAM
- **OS:** Windows 11 (WSL2) or Linux with CUDA support

## Quick Start

### 1. Install Dependencies

**For Linux / WSL2:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**For Windows (Native):**
```powershell
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install transformers accelerate datasets tqdm psutil

# IMPORTANT: bitsandbytes on Windows
# Use the Windows-compatible fork:
pip install bitsandbytes-windows
# OR if using WSL2, the standard package works:
# pip install bitsandbytes
```

### 2. Authenticate with Hugging Face

You need access to Llama-3 which requires accepting Meta's license:

1. Go to https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
2. Accept the license agreement
3. Login via CLI:

```bash
huggingface-cli login
```

### 3. Run the Script

```bash
python generate_oracle_data.py
```

## Output

The script generates `oracle_data.json` containing:

```json
{
  "metadata": {
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "quantization": "4-bit (bitsandbytes)",
    "block_size": 512,
    "num_samples": 50
  },
  "samples": [
    {
      "question": "What is the main character's name?",
      "answer": "John",
      "top_5_block_ids": [0, 14, 3, 7, 2],
      "all_block_scores": {"0": 0.15, "1": 0.02, ...},
      "num_blocks": 16,
      "context_tokens": 7500
    }
  ]
}
```

## Technical Details

### 4-bit Quantization

We use `bitsandbytes` with NF4 (Normal Float 4-bit) quantization to fit the 8B parameter model in 16GB VRAM:

- **Memory with FP16:** ~16GB (won't fit)
- **Memory with 4-bit:** ~5-6GB (fits comfortably)

### Attention Extraction

We use `attn_implementation="eager"` because FlashAttention and SDPA kernels often don't expose raw attention weights. Eager mode is slower but necessary for our analysis.

### Block Mapping

- Documents are divided into 512-token chunks
- Attention scores from the last token are summed per block
- Top 5 blocks with highest attention are recorded

## Troubleshooting

### "CUDA out of memory"
- Reduce `MAX_TOKENS` from 8000 to 6000
- Close other GPU applications

### "bitsandbytes not working on Windows"
- Use WSL2 instead of native Windows
- Or install `bitsandbytes-windows` fork

### "Model download requires authentication"
- Run `huggingface-cli login`
- Accept the Llama-3 license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

## License

Research use only. Llama-3 is subject to Meta's license terms.
