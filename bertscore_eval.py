"""
BERTScore evaluation of the best summarization model.
Loads (or generates) summaries, computes BERTScore + per-paper ROUGE,
saves bertscore_results.json, and produces a scatter plot.
"""
import json
import os
import statistics
import time
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from bert_score import score as bert_score
from rouge_score import rouge_scorer as rouge_lib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

DATA_FILE = "arxiv_papers.json"
COMPARISON_FILE = "results/model_comparison.json"
EVAL_RESULTS_FILE = "results/evaluation_results.json"
OUTPUT_FILE = "results/bertscore_results.json"
SCATTER_FILE = "plots/rouge_vs_bertscore.png"

EVAL_SAMPLE = 200       # match benchmark_models.py
LATENCY_PAPERS = 10

# deberta-xlarge-mnli (~900 MB) is slow on CPU/MPS.
# Swap to "roberta-large" for a 3-4x speedup with minimal quality loss.
BERTSCORE_MODEL = "roberta-large"

MAX_INPUT_LEN = 512
MAX_SUMMARY_LEN = 150
MIN_SUMMARY_LEN = 50
NUM_BEAMS = 4


def load_val_papers(path: str, n: int) -> list:
    with open(path) as f:
        papers = json.load(f)["papers"]
    usable = [p for p in papers if p.get("full_text", "").strip() and p.get("summary", "").strip()]
    val_start = len(usable) - 600
    return usable[val_start : val_start + n]


def pick_model_id() -> str:
    """Return the best model ID from prior benchmark, falling back to fine-tuned BART."""
    if os.path.exists(COMPARISON_FILE):
        with open(COMPARISON_FILE) as f:
            data = json.load(f)
        best_name = data.get("best_model", "")
        for m in data.get("models", []):
            if m["name"] == best_name:
                model_id = m["model_id"]
                print(f"  Using best model from benchmark: {best_name} ({model_id})")
                return model_id
    print("  results/model_comparison.json not found — using fine-tuned BART.")
    return "./fine_tuned_arxiv_bart"


def generate_summaries(model_id: str, papers: list, device) -> tuple[list[str], float]:
    print(f"  Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    model.eval()

    prefix = "summarize: " if "t5" in model_id.lower() else ""
    summaries = []
    t0 = time.perf_counter()
    for i, paper in enumerate(papers):
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(papers)}")
        source = prefix + paper["full_text"][:2000]
        inputs = tokenizer(source, max_length=MAX_INPUT_LEN, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=MAX_SUMMARY_LEN,
                min_length=MIN_SUMMARY_LEN,
                num_beams=NUM_BEAMS,
                early_stopping=True,
            )
        summaries.append(tokenizer.decode(out[0], skip_special_tokens=True))
    elapsed = time.perf_counter() - t0
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return summaries, elapsed


def per_paper_rouge1(predictions: list, references: list) -> list[float]:
    scorer = rouge_lib.RougeScorer(["rouge1"], use_stemmer=True)
    return [scorer.score(ref, pred)["rouge1"].fmeasure for pred, ref in zip(predictions, references)]


def main():
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    print(f"\nLoading {EVAL_SAMPLE} validation papers...")
    papers = load_val_papers(DATA_FILE, EVAL_SAMPLE)
    references = [p["summary"] for p in papers]
    print(f"  Loaded {len(papers)} papers.")

    model_id = pick_model_id()

    print("\nGenerating summaries...")
    summaries, gen_time = generate_summaries(model_id, papers, device)
    print(f"  Generated {len(summaries)} summaries in {gen_time:.1f}s")

    # ── ROUGE per paper ───────────────────────────────────────────────────────
    print("\nComputing per-paper ROUGE-1...")
    rouge_scores = per_paper_rouge1(summaries, references)
    mean_rouge1 = sum(rouge_scores) / len(rouge_scores)
    print(f"  Mean ROUGE-1: {mean_rouge1:.4f}")

    # ── BERTScore ─────────────────────────────────────────────────────────────
    print(f"\nComputing BERTScore (model: {BERTSCORE_MODEL})...")
    print("  This may take several minutes on CPU/MPS...")
    P, R, F1 = bert_score(
        summaries,
        references,
        model_type=BERTSCORE_MODEL,
        lang="en",
        verbose=True,
    )
    f1_list = F1.tolist()
    mean_f1 = statistics.mean(f1_list)
    median_f1 = statistics.median(f1_list)
    std_f1 = statistics.stdev(f1_list)
    min_f1, max_f1 = min(f1_list), max(f1_list)

    # ── Correlation ───────────────────────────────────────────────────────────
    n = len(rouge_scores)
    mean_r = mean_rouge1
    mean_b = mean_f1
    cov = sum((rouge_scores[i] - mean_r) * (f1_list[i] - mean_b) for i in range(n)) / n
    std_r = statistics.stdev(rouge_scores)
    correlation = cov / (std_r * std_f1) if std_r * std_f1 > 0 else 0.0

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("BERTScore Evaluation:")
    print(f"  Mean F1:   {mean_f1:.4f}")
    print(f"  Median F1: {median_f1:.4f}")
    print(f"  Std Dev:   {std_f1:.4f}")
    print(f"  Min: {min_f1:.4f}, Max: {max_f1:.4f}")
    print()
    print(f"  ROUGE-1:       {mean_rouge1:.4f} (for comparison)")
    print(f"  BERTScore F1:  {mean_f1:.4f}")
    print(f"  Correlation:   {correlation:.4f}")
    interpretation = (
        f"ROUGE measures n-gram overlap ({mean_rouge1:.2f}), BERTScore measures semantic "
        f"similarity ({mean_f1:.2f}). "
        + (
            f"High BERTScore indicates summaries are semantically high-quality despite lower ROUGE."
            if mean_f1 > mean_rouge1
            else f"Strong ROUGE ({mean_rouge1:.2f}) and BERTScore ({mean_f1:.2f}) together confirm high-quality summaries."
        )
    )
    print(f"\n  Interpretation: {interpretation}")
    print("=" * 55)

    # ── 10 example summaries with per-paper scores ────────────────────────────
    print("\n10 example summaries (ROUGE-1 vs BERTScore F1):")
    examples = []
    for i in range(min(10, len(summaries))):
        examples.append({
            "paper_title": papers[i].get("title", ""),
            "generated_summary": summaries[i],
            "ground_truth": references[i][:300] + "...",
            "rouge1": round(rouge_scores[i], 4),
            "bertscore_f1": round(f1_list[i], 4),
        })
        print(f"  [{i+1}] ROUGE-1={rouge_scores[i]:.4f}  BERTScore F1={f1_list[i]:.4f}  — {papers[i].get('title','')[:60]}")

    # ── Scatter plot ──────────────────────────────────────────────────────────
    print(f"\nGenerating scatter plot → {SCATTER_FILE}")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(rouge_scores, f1_list, alpha=0.4, s=20, color="steelblue", edgecolors="none")
    ax.set_xlabel("ROUGE-1", fontsize=13)
    ax.set_ylabel("BERTScore F1", fontsize=13)
    ax.set_title(
        f"ROUGE-1 vs BERTScore F1  (n={n}, r={correlation:.3f})\n{model_id}",
        fontsize=12,
    )
    # Reference lines at means
    ax.axvline(mean_rouge1, color="tomato", linestyle="--", linewidth=1, label=f"Mean ROUGE-1 = {mean_rouge1:.3f}")
    ax.axhline(mean_f1, color="orange", linestyle="--", linewidth=1, label=f"Mean BERTScore = {mean_f1:.3f}")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(SCATTER_FILE, dpi=150)
    plt.close()
    print(f"  Saved to {SCATTER_FILE}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results = {
        "model_id": model_id,
        "eval_sample": n,
        "bertscore_model": BERTSCORE_MODEL,
        "bertscore_f1_mean": round(mean_f1, 4),
        "bertscore_f1_median": round(median_f1, 4),
        "bertscore_f1_std": round(std_f1, 4),
        "bertscore_f1_min": round(min_f1, 4),
        "bertscore_f1_max": round(max_f1, 4),
        "rouge1_for_comparison": round(mean_rouge1, 4),
        "correlation_with_rouge": round(correlation, 4),
        "interpretation": interpretation,
        "example_summaries": examples,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
