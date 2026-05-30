"""
Three-way quantization comparison for the fine-tuned BART model:
  FP32 (original)  →  Dynamic INT8  →  ONNX INT8

Note on GPTQ / AWQ:
  Both techniques target decoder-only LLMs (LLaMA, GPT, Mistral) and require
  a CUDA GPU. They have no support for encoder-decoder seq2seq models (BART,
  PEGASUS, T5) and cannot run on Apple Silicon MPS. ONNX INT8 is the
  production-grade alternative for this architecture and hardware.
"""
import json
import os
import shutil
import time
import torch
from dotenv import load_dotenv
from rouge_score import rouge_scorer as rouge_lib
from transformers import BartForConditionalGeneration, AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.exporters.onnx import main_export

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

FINETUNED_DIR = "./fine_tuned_arxiv_bart"
INT8_DIR = "./quantized_int8"
ONNX_DIR = "./quantized_onnx"
ONNX_INT8_DIR = "./quantized_onnx_int8"
BEST_DIR = "./quantized_arxiv_best"

DATA_FILE = "arxiv_papers.json"
CACHE_FILE = "val_summaries_cache.json"
OUTPUT_FILE = "results/quantization_comparison.json"

EVAL_SAMPLE = 100
LATENCY_RUNS = 20
MAX_INPUT_LEN = 512
MAX_SUMMARY_LEN = 150
MIN_SUMMARY_LEN = 50
NUM_BEAMS = 4


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_val_papers(n: int) -> list:
    with open(DATA_FILE) as f:
        papers = json.load(f)["papers"]
    usable = [p for p in papers if p.get("full_text", "").strip() and p.get("summary", "").strip()]
    val_start = len(usable) - 600
    return usable[val_start : val_start + n]


def dir_size_mb(path: str) -> float:
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(path)
        for f in files
    )
    return total / (1024 ** 2)


def param_size_mb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)


def compute_rouge1(predictions: list, references: list) -> float:
    scorer = rouge_lib.RougeScorer(["rouge1"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rouge1"].fmeasure for pred, ref in zip(predictions, references)]
    return sum(scores) / len(scores)


# ── Inference helpers ─────────────────────────────────────────────────────────

def summarize_torch(model, tokenizer, text: str, device) -> str:
    inputs = tokenizer(text, max_length=MAX_INPUT_LEN, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=MAX_SUMMARY_LEN,
            min_length=MIN_SUMMARY_LEN,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def summarize_onnx(model, tokenizer, text: str) -> str:
    inputs = tokenizer(text, max_length=MAX_INPUT_LEN, truncation=True, return_tensors="pt")
    out = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=MAX_SUMMARY_LEN,
        min_length=MIN_SUMMARY_LEN,
        num_beams=NUM_BEAMS,
        early_stopping=True,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def measure_latency_torch(model, tokenizer, text: str, device, runs: int) -> float:
    for _ in range(2):  # warm-up
        summarize_torch(model, tokenizer, text, device)
    t0 = time.perf_counter()
    for _ in range(runs):
        summarize_torch(model, tokenizer, text, device)
    return (time.perf_counter() - t0) / runs * 1000


def measure_latency_onnx(model, tokenizer, text: str, runs: int) -> float:
    for _ in range(2):
        summarize_onnx(model, tokenizer, text)
    t0 = time.perf_counter()
    for _ in range(runs):
        summarize_onnx(model, tokenizer, text)
    return (time.perf_counter() - t0) / runs * 1000


def generate_summaries_torch(model, tokenizer, papers: list, device) -> list:
    model.eval()
    summaries = []
    for i, p in enumerate(papers):
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(papers)}")
        summaries.append(summarize_torch(model, tokenizer, p["full_text"][:2000], device))
    return summaries


def generate_summaries_onnx(model, tokenizer, papers: list) -> list:
    summaries = []
    for i, p in enumerate(papers):
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(papers)}")
        summaries.append(summarize_onnx(model, tokenizer, p["full_text"][:2000]))
    return summaries


# ── Quantization builders ─────────────────────────────────────────────────────

def build_fp32(device):
    print(f"  Loading FP32 model from {FINETUNED_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
    model = BartForConditionalGeneration.from_pretrained(FINETUNED_DIR).to(device)
    model.eval()
    return model, tokenizer


def build_int8(fp32_model):
    """Dynamic INT8 via torch.quantization.quantize_dynamic."""
    print("  Applying Dynamic INT8 quantization...")
    torch.backends.quantized.engine = "qnnpack"
    model = torch.quantization.quantize_dynamic(
        fp32_model.cpu(), {torch.nn.Linear}, dtype=torch.qint8
    )
    model.eval()
    os.makedirs(INT8_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(INT8_DIR, "pytorch_model.bin"))
    return model


def build_onnx_int8(tokenizer):
    """Export to ONNX then apply INT8 quantization via onnxruntime."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if not os.path.exists(os.path.join(ONNX_DIR, "encoder_model.onnx")):
        print(f"  Exporting BART to ONNX → {ONNX_DIR} ...")
        main_export(
            model_name_or_path=FINETUNED_DIR,
            output=ONNX_DIR,
            task="text2text-generation",
            opset=14,
        )
    else:
        print(f"  ONNX export already exists at {ONNX_DIR}, skipping export.")

    os.makedirs(ONNX_INT8_DIR, exist_ok=True)

    # Copy non-model files (tokenizer configs, generation config)
    for fname in os.listdir(ONNX_DIR):
        if not fname.endswith(".onnx"):
            src = os.path.join(ONNX_DIR, fname)
            dst = os.path.join(ONNX_INT8_DIR, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

    # Quantize each ONNX model file
    for fname in os.listdir(ONNX_DIR):
        if not fname.endswith(".onnx"):
            continue
        src = os.path.join(ONNX_DIR, fname)
        dst = os.path.join(ONNX_INT8_DIR, fname)
        if os.path.exists(dst):
            print(f"    {fname}: already quantized, skipping.")
            continue
        print(f"    Quantizing {fname}...")
        quantize_dynamic(src, dst, weight_type=QuantType.QInt8)

    print(f"  Loading ONNX INT8 model from {ONNX_INT8_DIR}...")
    ort_model = ORTModelForSeq2SeqLM.from_pretrained(ONNX_INT8_DIR, use_cache=False)
    return ort_model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Force CPU for INT8 (qnnpack), MPS for FP32 inference
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device (FP32): {device}")
    print(f"Device (INT8): cpu (qnnpack required)\n")

    print(f"Loading {EVAL_SAMPLE} validation papers...")
    papers = load_val_papers(EVAL_SAMPLE)
    references = [p["summary"] for p in papers]
    bench_text = papers[0]["full_text"][:2000]
    print(f"  Loaded {len(papers)} papers.\n")

    results = {}

    # ── 1. FP32 baseline ──────────────────────────────────────────────────────
    print("=" * 55)
    print("[1/3] FP32 Original")
    fp32_model, tokenizer = build_fp32(device)
    fp32_size = param_size_mb(fp32_model)
    fp32_latency = measure_latency_torch(fp32_model, tokenizer, bench_text, device, LATENCY_RUNS)
    print(f"  Size: {fp32_size:.0f} MB  |  Latency: {fp32_latency:.0f} ms")
    print(f"  Generating {EVAL_SAMPLE} summaries...")
    fp32_summaries = generate_summaries_torch(fp32_model, tokenizer, papers, device)
    fp32_rouge = compute_rouge1(fp32_summaries, references)
    print(f"  ROUGE-1: {fp32_rouge:.4f}")
    results["original"] = {
        "size_mb": round(fp32_size, 1),
        "latency_ms": round(fp32_latency, 1),
        "rouge1": round(fp32_rouge, 4),
        "speedup": "1.0x",
        "size_reduction": "—",
        "accuracy_loss": "—",
    }

    # ── 2. Dynamic INT8 ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("[2/3] Dynamic INT8")
    int8_model = build_int8(fp32_model)
    del fp32_model
    int8_size = dir_size_mb(INT8_DIR)
    int8_latency = measure_latency_torch(int8_model, tokenizer, bench_text, torch.device("cpu"), LATENCY_RUNS)
    size_red_int8 = (1 - int8_size / fp32_size) * 100
    speedup_int8 = fp32_latency / int8_latency
    print(f"  Size: {int8_size:.0f} MB  |  Latency: {int8_latency:.0f} ms  |  Speedup: {speedup_int8:.2f}x")
    print(f"  Generating {EVAL_SAMPLE} summaries...")
    int8_summaries = generate_summaries_torch(int8_model, tokenizer, papers, torch.device("cpu"))
    int8_rouge = compute_rouge1(int8_summaries, references)
    acc_loss_int8 = (fp32_rouge - int8_rouge) / fp32_rouge * 100
    print(f"  ROUGE-1: {int8_rouge:.4f}  |  Accuracy loss: {acc_loss_int8:.2f}%")
    results["dynamic_int8"] = {
        "size_mb": round(int8_size, 1),
        "size_reduction": f"{size_red_int8:.1f}%",
        "latency_ms": round(int8_latency, 1),
        "speedup": f"{speedup_int8:.2f}x",
        "rouge1": round(int8_rouge, 4),
        "accuracy_loss": f"{acc_loss_int8:.2f}%",
    }
    del int8_model

    # ── 3. ONNX INT8 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("[3/3] ONNX INT8")
    ort_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
    ort_model = build_onnx_int8(ort_tokenizer)
    onnx_size = dir_size_mb(ONNX_INT8_DIR)
    onnx_latency = measure_latency_onnx(ort_model, ort_tokenizer, bench_text, LATENCY_RUNS)
    size_red_onnx = (1 - onnx_size / fp32_size) * 100
    speedup_onnx = fp32_latency / onnx_latency
    print(f"  Size: {onnx_size:.0f} MB  |  Latency: {onnx_latency:.0f} ms  |  Speedup: {speedup_onnx:.2f}x")
    print(f"  Generating {EVAL_SAMPLE} summaries...")
    onnx_summaries = generate_summaries_onnx(ort_model, ort_tokenizer, papers)
    onnx_rouge = compute_rouge1(onnx_summaries, references)
    acc_loss_onnx = (fp32_rouge - onnx_rouge) / fp32_rouge * 100
    print(f"  ROUGE-1: {onnx_rouge:.4f}  |  Accuracy loss: {acc_loss_onnx:.2f}%")
    results["onnx_int8"] = {
        "size_mb": round(onnx_size, 1),
        "size_reduction": f"{size_red_onnx:.1f}%",
        "latency_ms": round(onnx_latency, 1),
        "speedup": f"{speedup_onnx:.2f}x",
        "rouge1": round(onnx_rouge, 4),
        "accuracy_loss": f"{acc_loss_onnx:.2f}%",
    }

    # ── Recommendation ────────────────────────────────────────────────────────
    # Best = highest speedup with accuracy loss < 5%
    candidates = [
        ("dynamic_int8", speedup_int8, acc_loss_int8),
        ("onnx_int8", speedup_onnx, acc_loss_onnx),
    ]
    recommended = max(
        [c for c in candidates if c[2] < 5.0],
        key=lambda c: c[1],
        default=candidates[0],
    )
    rec_key = recommended[0]
    rec_data = results[rec_key]

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Method':<16} | {'Size(MB)':>8} | {'Size↓':>7} | {'Latency(ms)':>11} | {'Speedup':>7} | {'ROUGE-1':>7} | {'Loss':>7}")
    print("-" * 80)
    rows = [
        ("FP32 (original)", results["original"]),
        ("Dynamic INT8",    results["dynamic_int8"]),
        ("ONNX INT8",       results["onnx_int8"]),
    ]
    for label, r in rows:
        print(
            f"{label:<16} | {r['size_mb']:>8.0f} | {r['size_reduction']:>7} | "
            f"{r['latency_ms']:>11.0f} | {r['speedup']:>7} | {r['rouge1']:>7.4f} | {r['accuracy_loss']:>7}"
        )
    print("=" * 80)

    print(f"\nBest overall: {rec_key.replace('_', ' ').upper()} with "
          f"{rec_data['speedup']} speedup, {rec_data['size_reduction']} size reduction, "
          f"{rec_data['accuracy_loss']} quality loss")
    print("For production: balance speed/quality tradeoff based on deployment constraints.")
    print("\nNote: GPTQ and AWQ were not applicable — both target decoder-only LLMs "
          "(LLaMA/GPT/Mistral) and require CUDA. ONNX INT8 is the correct "
          "production-grade alternative for encoder-decoder seq2seq models on CPU/MPS.")

    # ── Copy best quantized model ─────────────────────────────────────────────
    best_src = ONNX_INT8_DIR if rec_key == "onnx_int8" else INT8_DIR
    if os.path.exists(BEST_DIR):
        shutil.rmtree(BEST_DIR)
    shutil.copytree(best_src, BEST_DIR)
    print(f"\nBest model copied to {BEST_DIR}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        **results,
        "recommended": rec_key,
        "reason": f"Highest speedup ({rec_data['speedup']}) with <5% quality loss",
        "gptq_awq_note": (
            "GPTQ and AWQ are decoder-only LLM quantization techniques (LLaMA, GPT). "
            "They do not support encoder-decoder seq2seq models (BART, PEGASUS, T5) "
            "and require CUDA. ONNX INT8 is the correct alternative for this architecture."
        ),
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
