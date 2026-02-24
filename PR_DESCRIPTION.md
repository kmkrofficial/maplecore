# PR Description: MAPLE SDK Release & In-Situ Profiling

## Summary
This PR transitions the project from `svscout` experiments to the production-ready **MAPLE SDK**. It introduces a standardized benchmark suite with integrated hardware profiling, cleans up legacy scripts, and adds comprehensive documentation.

## Key Changes

### 1. In-Situ Hardware Profiling
- **New Module**: `benchmarks/profiler.py` with `HardwareMonitor` context manager.
- **Integration**: Automatically captures CPU, RAM, and VRAM usage for:
  - Recall Benchmarks (`01_recall_enhanced.py`)
  - Latency Scaling (`02_latency_scaling.py`)
  - Robustness Tests (`03_needle_robust.py`)
  - Model Training (`scripts/train_generalist.py`)
- **Output**: All results now follow a standardized `{ metrics: ..., hardware: ... }` JSON schema.

### 2. Benchmark Cleanup
- **Deleted**: Redundant/Legacy scripts (`01_recall.py`, `02_latency.py`, `03_needle.py`, `06_hardware_profile.py`).
- **Standardized**: Renamed and refactored core benchmarks to be modular and robust.

### 3. Documentation Overhaul
- **Root README**: completely rewritten to reflect **MAPLE** branding, features, and use cases.
- **Subfolder READMEs**: Added detailed documentation for `benchmarks/`, `maplecore/`, `scripts/`, `data/`, `models/`, `examples/`, and `paper_assets/`.
- **Git**: Forced inclusion of documentation for `.gitignore`d folders (`data`, `models`).

### 4. Branding
- Renamed project references from `Scout-KV` to **MAPLE**.
- Removed emojis for a professional research tone.

## Verification
- Ran all benchmarks to ensure `HardwareMonitor` integrates without error.
- Verified JSON output format.
- Checked generation of `README.md` in all subdirectories.

## Dependencies
- `psutil` (for hardware monitoring)
- `maplecore` (internal SDK)
