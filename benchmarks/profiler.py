
import time
import psutil
import torch
import platform
import threading
import numpy as np
import os
import sys

class HardwareMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.running = False
        self.cpu_samples = []
        self.ram_samples = []
        self.vram_samples = []
        self.monitor_thread = None
        self.start_time = 0
        self.end_time = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.running = True
        self.cpu_samples = []
        self.ram_samples = []
        self.vram_samples = []
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.end_time = time.time()

    def _monitor_loop(self):
        process = psutil.Process()
        # Initial call to cpu_percent needs to be discarded or it returns 0.0
        process.cpu_percent(interval=None)
        
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
        duration = self.end_time - self.start_time if self.end_time > self.start_time else time.time() - self.start_time
        stats['duration_seconds'] = round(duration, 2)
        
        # CPU
        if self.cpu_samples:
            stats['cpu_peak_percent'] = float(np.max(self.cpu_samples))
            stats['cpu_p95_percent'] = float(np.percentile(self.cpu_samples, 95))
            stats['cpu_mean_percent'] = float(np.mean(self.cpu_samples))
        else:
            stats['cpu_peak_percent'] = 0.0
            
        # RAM
        if self.ram_samples:
            stats['ram_peak_gb'] = float(np.max(self.ram_samples))
            stats['ram_p95_gb'] = float(np.percentile(self.ram_samples, 95))
        else:
            stats['ram_peak_gb'] = 0.0

        # VRAM
        if self.vram_samples:
            stats['vram_peak_gb'] = float(np.max(self.vram_samples))
            stats['vram_p95_gb'] = float(np.percentile(self.vram_samples, 95))
        else:
            stats['vram_peak_gb'] = 0.0
            
        return stats

def get_system_info():
    info = {}
    
    # CPU
    info['cpu_model'] = platform.processor()
    info['cpu_cores'] = psutil.cpu_count(logical=True)
    
    # RAM
    total_ram = psutil.virtual_memory().total / (1024 * 1024 * 1024) # GB
    info['ram_total_gb'] = round(total_ram, 1)
    
    # GPU
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024) # GB
        info['gpu_vram_gb'] = round(total_vram, 1)
    else:
        info['gpu_name'] = "None (CPU Only)"
        info['gpu_vram_gb'] = 0.0
        
    # Software
    info['python_version'] = platform.python_version()
    info['torch_version'] = torch.__version__
    
    return info

def wrap_result(metrics: dict, monitor: HardwareMonitor) -> dict:
    """
    Standard format for benchmark results with hardware stats.
    """
    return {
        "metrics": metrics,
        "hardware": {
            "system": get_system_info(),
            "usage": monitor.get_stats()
        }
    }
