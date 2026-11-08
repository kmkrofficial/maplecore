# Scripts

Run from the repository root. Data layout and schemas: [docs/DATA.md](../docs/DATA.md).

| Script | Purpose |
|---|---|
| `download_data.py` | Download datasets, Gutenberg books, encoders and (with `--only teacher`) the gated teacher LLM; writes `data/MANIFEST.json` |
| `oracle_common.py` | Shared teacher loading, prompt construction, last-row attention hooks, evidence helpers, oracle I/O |
| `generate_oracle.py` | NarrativeQA oracle labels → `data/oracle/narrativeqa_{split}.json` |
| `generate_oracle_hotpot.py` | HotpotQA oracle labels (+ gold supporting-sentence blocks) → `data/oracle/hotpotqa_{split}.json` |
| `train_generalist.py` | **Canonical trainer**: NarrativeQA + HotpotQA oracle labels + optional synthetic needles → `models/maple_generalist.pth` |
| `train_maple.py` | Compatibility wrapper: NarrativeQA-only, no needles → `models/maple_v1.pth` |
