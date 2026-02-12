
import time
import psutil
import torch
import platform
import threading
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from maplecore import MapleIndexer, MapleScanner, MapleNet
from benchmarks.config import MODEL_PATH, DEFAULT_DEVICE

# ---------------------------------------------------------------------------
# Hardware Monitor Class
# ---------------------------------------------------------------------------
class HardwareMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.running = False
        self.cpu_samples = []
        self.ram_samples = []
        self.vram_samples = []
        self.monitor_thread = None

    def start(self):
        self.running = True
        self.cpu_samples = []
        self.ram_samples = []
        self.vram_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()

    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self):
        process = psutil.Process()
        while self.running:
            # CPU
            try:
                self.cpu_samples.append(process.cpu_percent(interval=None))
            except:
                pass
            
            # RAM
            try:
                mem = process.memory_info().rss / (1024 * 1024 * 1024) # GB
                self.ram_samples.append(mem)
            except:
                pass

            # VRAM (CUDA)
            if torch.cuda.is_available():
                try:
                    vram = torch.cuda.memory_allocated() / (1024 * 1024 * 1024) # GB
                    self.vram_samples.append(vram)
                except:
                    pass
            
            time.sleep(self.interval)

    def get_stats(self):
        stats = {}
        
        # CPU
        if self.cpu_samples:
            stats['cpu_peak'] = np.max(self.cpu_samples)
            stats['cpu_p95'] = np.percentile(self.cpu_samples, 95)
            stats['cpu_mean'] = np.mean(self.cpu_samples)
        else:
            stats['cpu_peak'] = 0
            stats['cpu_p95'] = 0
            stats['cpu_mean'] = 0

        # RAM
        if self.ram_samples:
            stats['ram_peak'] = np.max(self.ram_samples)
            stats['ram_p95'] = np.percentile(self.ram_samples, 95)
            stats['ram_mean'] = np.mean(self.ram_samples)
        else:
            stats['ram_peak'] = 0
            stats['ram_p95'] = 0
            stats['ram_mean'] = 0

        # VRAM
        if self.vram_samples:
            stats['vram_peak'] = np.max(self.vram_samples)
            stats['vram_p95'] = np.percentile(self.vram_samples, 95)
            stats['vram_mean'] = np.mean(self.vram_samples)
        else:
            stats['vram_peak'] = 0
            stats['vram_p95'] = 0
            stats['vram_mean'] = 0
            
        return stats

# ---------------------------------------------------------------------------
# System Info
# ---------------------------------------------------------------------------
def get_system_info():
    info = {}
    
    # CPU
    info['cpu_model'] = platform.processor()
    info['cpu_cores'] = psutil.cpu_count(logical=True)
    
    # RAM
    total_ram = psutil.virtual_memory().total / (1024 * 1024 * 1024) # GB
    info['ram_total'] = f"{total_ram:.1f} GB"
    
    # GPU
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024) # GB
        info['gpu_vram'] = f"{total_vram:.1f} GB"
    else:
        info['gpu_name'] = "None (CPU Only)"
        info['gpu_vram'] = "0 GB"
        
    # Software
    info['python_version'] = platform.python_version()
    info['torch_version'] = torch.__version__
    
    return info

# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------
def generate_haystack(length: int) -> str:
    # Alice text (Same locally generated logic)
    alices_adventures = (
        "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: "
        "once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "
        "'and what is the use of a book,' thought Alice 'without pictures or conversation?' "
        "So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), "
        "whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, "
        "when suddenly a White Rabbit with pink eyes ran close by her. "
    ) * 50 # Multiplier to ensure some base length
    
    multiplier = (length // len(alices_adventures)) + 1
    return (alices_adventures * multiplier)[:length]

def profile_workload():
    print("Initializing Profile Workload...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate Workload (100k chars ~ 200 blocks)
    text = generate_haystack(100000)
    
    # Init Components
    model = MapleNet.load(str(MODEL_PATH), device=device)
    indexer = MapleIndexer(device=device)
    scanner = MapleScanner(model, device=device)
    
    monitor = HardwareMonitor(interval=0.05)
    
    # 1. Indexing Profile
    print("Profiling Indexing...")
    monitor.start()
    index = indexer.create_index(text)
    monitor.stop()
    indexing_stats = monitor.get_stats()
    
    # 2. Search Profile (Batch of 10 queries)
    print("Profiling Searching...")
    query = "What did the White Rabbit say?"
    query_emb = indexer.encode_query(query)
    
    monitor.start()
    for _ in range(10):
        scanner.search(query_emb, index, k=5)
    monitor.stop()
    search_stats = monitor.get_stats()
    
    return indexing_stats, search_stats

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sys_info = get_system_info()
    
    print("\n" + "="*50)
    print("EXPERIMENTAL CONFIGURATION")
    print("="*50)
    print(f"CPU: {sys_info['cpu_model']} ({sys_info['cpu_cores']} Cores)")
    print(f"RAM: {sys_info['ram_total']}")
    print(f"GPU: {sys_info['gpu_name']} ({sys_info['gpu_vram']})")
    print(f"Software: Python {sys_info['python_version']}, PyTorch {sys_info['torch_version']}")
    print(f"Test Parameters: Chunk Size=500, Embedding=BGE-Small")
    print("="*50 + "\n")
    
    indexing_stats, search_stats = profile_workload()
    
    # Generate Report
    report_file = Path("benchmarks/results/hardware_profile.md")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    content = f"""# Hardware Resource Profile
**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}

## System Configuration
- **CPU:** {sys_info['cpu_model']} ({sys_info['cpu_cores']} Cores)
- **RAM:** {sys_info['ram_total']}
- **GPU:** {sys_info['gpu_name']} ({sys_info['gpu_vram']})
- **Software:** Python {sys_info['python_version']}, PyTorch {sys_info['torch_version']}

## Resource Usage Stats (P95 vs Peak)

| Task | CPU P95 (%) | CPU Peak (%) | RAM P95 (GB) | RAM Peak (GB) | VRAM Peak (GB) | VRAM P95 (GB) |
|------|-------------|--------------|--------------|---------------|----------------|---------------|
| **Indexing** | {indexing_stats['cpu_p95']:.1f}% | {indexing_stats['cpu_peak']:.1f}% | {indexing_stats['ram_p95']:.2f} | {indexing_stats['ram_peak']:.2f} | {indexing_stats['vram_peak']:.2f} | {indexing_stats['vram_p95']:.2f} |
| **Search** | {search_stats['cpu_p95']:.1f}% | {search_stats['cpu_peak']:.1f}% | {search_stats['ram_p95']:.2f} | {search_stats['ram_peak']:.2f} | {search_stats['vram_peak']:.2f} | {search_stats['vram_p95']:.2f} |

*Note: P95 represents the 95th percentile, indicating sustained load excluding transient spikes.*
"""
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"\nReport generated at {report_file}")
    
if __name__ == "__main__":
    main()
