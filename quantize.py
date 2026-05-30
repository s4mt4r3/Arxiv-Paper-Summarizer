"""
INT8 dynamic quantization of fine-tuned BART + inference speed benchmark.
"""
import json
import os
import time
import torch
from dotenv import load_dotenv
from transformers import BartTokenizer, BartForConditionalGeneration

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
from rouge_score import rouge_scorer

FINETUNED_DIR = "./fine_tuned_arxiv_bart"
QUANTIZED_DIR = "./quantized_arxiv_bart"
DATA_FILE = "arxiv_papers.json"
OUTPUT_FILE = "results/quantization_results.json"

MAX_INPUT_LEN = 512
MAX_SUMMARY_LEN = 150
MIN_SUMMARY_LEN = 50
NUM_BEAMS = 4
BENCH_RUNS = 10
ROUGE_SAMPLES = 50


def load_val_paper(path: str) -> dict:
    with open(path) as f:
        papers = json.load(f)["papers"]
    usable = [p for p in papers if p.get("full_text", "").strip() and p.get("summary", "").strip()]
    # Return one paper from the val split
    return usable[-(600 // 2)]


def load_val_papers(path: str, n: int = ROUGE_SAMPLES) -> list:
    with open(path) as f:
        papers = json.load(f)["papers"]
    usable = [p for p in papers if p.get("full_text", "").strip() and p.get("summary", "").strip()]
    return usable[-600:][:n]


def encode(tokenizer, text: str):
    return tokenizer(
        text,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        return_tensors="pt",
    )


def generate_summary(model, inputs: dict, tokenizer) -> str:
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=MAX_SUMMARY_LEN,
            min_length=MIN_SUMMARY_LEN,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def benchmark_latency(model, inputs: dict, tokenizer, runs: int = BENCH_RUNS) -> float:
    # Warmup
    generate_summary(model, inputs, tokenizer)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        generate_summary(model, inputs, tokenizer)
        times.append(time.perf_counter() - t0)
    return (sum(times) / len(times)) * 1000  # ms


def dir_size_mb(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 ** 2)


def rouge1_score(predictions: list, references: list) -> float:
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rouge1"].fmeasure for pred, ref in zip(predictions, references)]
    return sum(scores) / len(scores)


def main():
    # Always run on CPU — MPS doesn't support INT8 quantization
    device = torch.device("cpu")
    print(f"Device: {device} (quantization requires CPU)")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer from {FINETUNED_DIR}...")
    tokenizer = BartTokenizer.from_pretrained(FINETUNED_DIR)

    # ── Load fine-tuned FP32 model ────────────────────────────────────────────
    print(f"Loading FP32 model from {FINETUNED_DIR}...")
    fp32_model = BartForConditionalGeneration.from_pretrained(FINETUNED_DIR).to(device)
    fp32_model.eval()

    # ── Quantize to INT8 ──────────────────────────────────────────────────────
    print("Applying dynamic INT8 quantization...")
    torch.backends.quantized.engine = "qnnpack"  # required on Apple Silicon / macOS
    int8_model = torch.quantization.quantize_dynamic(
        fp32_model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    os.makedirs(QUANTIZED_DIR, exist_ok=True)
    # Save quantized model state dict + config
    torch.save(int8_model.state_dict(), os.path.join(QUANTIZED_DIR, "pytorch_model.bin"))
    fp32_model.config.save_pretrained(QUANTIZED_DIR)
    tokenizer.save_pretrained(QUANTIZED_DIR)
    print(f"Quantized model saved to {QUANTIZED_DIR}")

    # ── Benchmark latency ─────────────────────────────────────────────────────
    print(f"\nBenchmarking latency ({BENCH_RUNS} runs each)...")
    test_paper = load_val_paper(DATA_FILE)
    inputs = encode(tokenizer, test_paper["full_text"])

    fp32_latency = benchmark_latency(fp32_model, inputs, tokenizer)
    int8_latency = benchmark_latency(int8_model, inputs, tokenizer)
    speedup = fp32_latency / int8_latency if int8_latency > 0 else 0

    # ── Model size ────────────────────────────────────────────────────────────
    fp32_size_mb = dir_size_mb(FINETUNED_DIR)
    int8_size_mb = dir_size_mb(QUANTIZED_DIR)
    size_reduction_pct = ((fp32_size_mb - int8_size_mb) / fp32_size_mb) * 100

    # ── ROUGE quality check ───────────────────────────────────────────────────
    print(f"\nRunning ROUGE quality check on {ROUGE_SAMPLES} papers...")
    val_papers = load_val_papers(DATA_FILE)
    references = [p["summary"] for p in val_papers]

    fp32_summaries, int8_summaries = [], []
    for i, paper in enumerate(val_papers):
        enc = encode(tokenizer, paper["full_text"])
        fp32_summaries.append(generate_summary(fp32_model, enc, tokenizer))
        int8_summaries.append(generate_summary(int8_model, enc, tokenizer))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{ROUGE_SAMPLES}")

    fp32_rouge1 = rouge1_score(fp32_summaries, references)
    int8_rouge1 = rouge1_score(int8_summaries, references)
    accuracy_loss = ((fp32_rouge1 - int8_rouge1) / fp32_rouge1) * 100 if fp32_rouge1 > 0 else 0

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("QUANTIZATION RESULTS")
    print("=" * 50)
    print(f"  FP32 latency  : {fp32_latency:.1f} ms per inference")
    print(f"  INT8 latency  : {int8_latency:.1f} ms per inference")
    print(f"  Speedup       : {speedup:.2f}x")
    print()
    print(f"  Original model size : {fp32_size_mb:.1f} MB")
    print(f"  Quantized model size: {int8_size_mb:.1f} MB")
    print(f"  Size reduction      : {size_reduction_pct:.1f}%")
    print()
    print(f"  ROUGE-1 (FP32)     : {fp32_rouge1:.4f}")
    print(f"  ROUGE-1 (INT8)     : {int8_rouge1:.4f}")
    print(f"  Accuracy loss      : {accuracy_loss:.2f}%")
    print("=" * 50)

    results = {
        "latency": {
            "fp32_ms": fp32_latency,
            "int8_ms": int8_latency,
            "speedup": speedup,
        },
        "model_size": {
            "fp32_mb": fp32_size_mb,
            "int8_mb": int8_size_mb,
            "size_reduction_pct": size_reduction_pct,
        },
        "rouge": {
            "fp32_rouge1": fp32_rouge1,
            "int8_rouge1": int8_rouge1,
            "accuracy_loss_pct": accuracy_loss,
        },
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
