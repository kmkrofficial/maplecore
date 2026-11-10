# Examples

Run from the repository root. Scripts that load `models/maple_generalist.pth`
need a trained checkpoint (see the root README).

| Script | What it shows | Needs a checkpoint |
|---|---|---|
| `demo_infinite_context.py` | Index a full novel and query it with the high-level `Maple` client | yes |
| `benchmark_vs_rag.py` | Quick MAPLE vs cosine-similarity comparison on NarrativeQA | yes |
| `verify_custom_embedding.py` | Using a non-default sentence-transformer encoder | no |
| `verify_document_parsing.py` | PDF / DOCX / HTML ingestion (`pip install -e ".[parsers]"`) | no |
| `verify_onnx_export.py` | Export `MapleNet` to ONNX and search with `MapleONNXRunner` | no (random weights) |
