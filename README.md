# ArXiv Paper Summarizer

Fine-tuned BART-base on 2,596 ArXiv papers for abstractive summarization, with INT8 quantization, multi-model benchmarking, BERTScore evaluation, domain-shift analysis, and a Streamlit UI.

## Key Results

| Metric | Value |
|--------|-------|
| ROUGE-1 (fine-tuned) | **0.5596** |
| ROUGE-L (fine-tuned) | **0.5499** |
| ROUGE-1 improvement over baseline | **+139.6%** |
| BERTScore F1 (roberta-large) | **0.9187** |
| ROUGE–BERTScore correlation | 0.88 |
| INT8 size reduction | **46%** (532 MB → 286 MB) |
| INT8 inference speedup | **1.42x** |
| INT8 quality loss | 3.44% |
| Domain shift degradation | 2.4% (cs.CL → cs.CV/cs.LG/stat.ML) |

## Pipeline

```
scrape_arxiv.py        →  arxiv_papers.json        (2,596 cs.CL papers w/ PDFs)
train_bart.py          →  fine_tuned_arxiv_bart/    (fine-tuned BART-base)
evaluate.py            →  evaluation_results.json   (ROUGE vs baseline)
quantize.py            →  quantized_arxiv_bart/     (Dynamic INT8)
benchmark_models.py    →  model_comparison.json     (5-model comparison)
bertscore_eval.py      →  bertscore_results.json    (BERTScore + scatter plot)
domain_shift_eval.py   →  domain_shift_analysis.json (cross-domain ROUGE)
error_analysis.py      →  error_analysis.json       (failure mode taxonomy)
quantization_comparison.py → quantization_comparison.json (INT8 vs ONNX INT8)
app.py                 →  Streamlit UI
```

## Setup

```bash
git clone <repo>
cd paper-summarizer
pip install -r requirements.txt
cp .env.example .env   # add HF_TOKEN if you have one
```

## Running the App

```bash
streamlit run app.py
```

Enter an ArXiv ID in `arXiv:XXXX.XXXXX` format or paste paper text directly.

## Reproducing from Scratch

Run scripts in this order:

```bash
# 1. Scrape ArXiv papers (takes several hours, resumes on interrupt)
python3 scrape_arxiv.py

# 2. Fine-tune BART-base (~2 hours on GPU, ~8 hours on CPU)
python3 train_bart.py

# 3. Evaluate vs baseline
python3 evaluate.py

# 4. Quantize to INT8
python3 quantize.py

# 5. Benchmark against other pretrained models
python3 benchmark_models.py

# 6. BERTScore evaluation
python3 bertscore_eval.py

# 7. Domain-shift analysis (fetches cs.CV, cs.LG, stat.ML papers from ArXiv API)
python3 domain_shift_eval.py

# 8. Failure case analysis
python3 error_analysis.py

# 9. Quantization comparison (Dynamic INT8 vs ONNX INT8)
python3 quantization_comparison.py
```

## Evaluation Results

### Model Benchmark (200 validation papers)

| Model | ROUGE-1 | ROUGE-L | Latency (ms) | Size (MB) |
|-------|---------|---------|--------------|-----------|
| **BART-base (fine-tuned)** | **0.5596** | **0.5499** | 10,328 | 532 |
| T5-base | 0.3252 | 0.3037 | 4,272 | 850 |
| DistilBART-CNN-6-6 | 0.3034 | 0.2658 | 3,851 | 877 |
| LED-base-16384 | 0.2049 | 0.1517 | 7,894 | 617 |

### Quantization Comparison (100 validation papers)

| Method | Size | Reduction | Latency | Speedup | ROUGE-1 | Loss |
|--------|------|-----------|---------|---------|---------|------|
| FP32 (original) | 532 MB | — | 3,975 ms | 1.0x | 0.5570 | — |
| **Dynamic INT8** | 286 MB | 46% | 2,803 ms | **1.42x** | 0.5379 | 3.44% |
| ONNX INT8 | 212 MB | 60% | 20,881 ms | 0.19x | 0.4804 | 13.76% |

> ONNX INT8 is slower on Apple Silicon MPS because ONNX Runtime does not leverage MPS acceleration. Dynamic INT8 is the recommended deployment format for CPU/MPS inference.

### Domain Shift (200 papers per domain)

| Domain | ROUGE-1 | ROUGE-L | BERTScore F1 |
|--------|---------|---------|--------------|
| CS.CL (in-domain) | 0.5596 | 0.5499 | 0.9187 |
| CS.CV | 0.5477 | 0.5452 | 0.9237 |
| CS.LG | 0.5538 | 0.5520 | 0.9235 |
| STAT.ML | 0.5374 | 0.5346 | 0.9198 |

**Performance gap: 2.4%** — the model generalizes well across ML subfields despite being trained exclusively on cs.CL papers.

### Failure Mode Analysis (50 lowest-ROUGE papers)

| Failure Mode | Count | % |
|-------------|-------|---|
| Length mismatch | 40 | 80% |
| Hallucination (NLI-detected) | 10 | 20% |

Primary failure: generated summaries diverge from reference abstract length. Adding `length_penalty > 1.0` to beam search is the most direct fix.

## Project Structure

```
paper-summarizer/
├── app.py                        # Streamlit UI
├── scrape_arxiv.py               # ArXiv data collection (PDF + abstracts)
├── train_bart.py                 # BART fine-tuning
├── evaluate.py                   # ROUGE evaluation vs baseline
├── quantize.py                   # Dynamic INT8 quantization
├── benchmark_models.py           # Multi-model ROUGE benchmark
├── bertscore_eval.py             # BERTScore + ROUGE scatter plot
├── domain_shift_eval.py          # Cross-domain generalization eval
├── error_analysis.py             # Failure case taxonomy (NLI hallucination detection)
├── quantization_comparison.py    # INT8 vs ONNX INT8 comparison
├── requirements.txt
├── results/
│   ├── evaluation_results.json
│   ├── model_comparison.json
│   ├── bertscore_results.json
│   ├── domain_shift_analysis.json
│   ├── error_analysis.json
│   ├── quantization_results.json
│   └── quantization_comparison.json
└── plots/
    ├── rouge_vs_bertscore.png
    └── domain_shift_chart.png
```

## Technical Notes

- **Hardware:** Developed and benchmarked on Apple Silicon (MPS). INT8 uses `qnnpack` backend.
- **GPTQ / AWQ:** Not applicable — both target decoder-only LLMs (LLaMA, GPT) and require CUDA. Dynamic INT8 is the correct quantization method for encoder-decoder seq2seq models on CPU/MPS.
- **BERTScore model:** `roberta-large`. `deberta-xlarge-mnli` was tested but OOM'd on 16 GB unified memory.
- **Domain shift data:** cs.CV, cs.LG, and stat.ML papers are fetched live from the ArXiv API and cached to `domain_papers_cache.json`.
