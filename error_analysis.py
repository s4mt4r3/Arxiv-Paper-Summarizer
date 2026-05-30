"""
Failure-case analysis on the 50 lowest-ROUGE validation papers.

Pipeline:
  1. Regenerate summaries for all val papers (cached to val_summaries_cache.json)
  2. Compute per-paper ROUGE-1, sort ascending, take bottom 50
  3. Classify each failure into ONE category (priority order):
       hallucination → length_mismatch → missing_contribution → scope_creep → over_generalization
  4. Print table, save error_analysis.json
"""
import json
import os
import re
import time
import torch
from dotenv import load_dotenv
from rouge_score import rouge_scorer as rouge_lib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

DATA_FILE = "arxiv_papers.json"
CACHE_FILE = "val_summaries_cache.json"
OUTPUT_FILE = "results/error_analysis.json"

FINETUNED_DIR = "./fine_tuned_arxiv_bart"
NLI_MODEL = "typeform/distilbert-base-uncased-mnli"  # 66 MB, fast

VAL_SIZE = 600
EVAL_SAMPLE = 200
BOTTOM_N = 50

MAX_INPUT_LEN = 512
MAX_SUMMARY_LEN = 150
MIN_SUMMARY_LEN = 50
NUM_BEAMS = 4

# Failure thresholds
LENGTH_MISMATCH_THRESHOLD = 50   # chars
KEYWORD_COVERAGE_THRESHOLD = 0.3  # fraction of title keywords that must appear
NLI_CONTRADICTION_THRESHOLD = 0.6

SCOPE_CREEP_PATTERNS = re.compile(
    r"\b(in recent years|this paper (reviews|surveys|provides an overview)|"
    r"background|introduction|traditionally|historically|it is well.known|"
    r"prior work|previous (work|studies|research))\b",
    re.IGNORECASE,
)

STOP_WORDS = {
    "a","an","the","of","in","on","for","to","and","or","with","is","are",
    "that","this","from","by","as","at","be","can","it","its","we","our",
    "based","using","via","new","deep","large","small","paper","model",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_val_papers() -> list:
    with open(DATA_FILE) as f:
        papers = json.load(f)["papers"]
    usable = [p for p in papers if p.get("full_text", "").strip() and p.get("summary", "").strip()]
    val_start = len(usable) - VAL_SIZE
    return usable[val_start : val_start + EVAL_SAMPLE]


# ── Summary generation (with cache) ──────────────────────────────────────────

def generate_all_summaries(papers: list, device) -> list[str]:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        if len(cache.get("summaries", [])) == len(papers):
            print(f"  Loaded {len(papers)} summaries from cache.")
            return cache["summaries"]

    print(f"  Loading model from {FINETUNED_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED_DIR).to(device)
    model.eval()

    summaries = []
    for i, paper in enumerate(papers):
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(papers)}")
        inputs = tokenizer(
            paper["full_text"], max_length=MAX_INPUT_LEN, truncation=True, return_tensors="pt"
        ).to(device)
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

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    with open(CACHE_FILE, "w") as f:
        json.dump({"summaries": summaries}, f, indent=2)
    print(f"  Summaries cached to {CACHE_FILE}")
    return summaries


# ── Per-paper ROUGE-1 ─────────────────────────────────────────────────────────

def per_paper_rouge1(summaries: list, references: list) -> list[float]:
    scorer = rouge_lib.RougeScorer(["rouge1"], use_stemmer=True)
    return [scorer.score(ref, pred)["rouge1"].fmeasure for pred, ref in zip(summaries, references)]


# ── Failure classification ────────────────────────────────────────────────────

def title_keywords(title: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\b\w+\b", title) if w.lower() not in STOP_WORDS and len(w) > 2}


def classify_hallucination(nli_pipe, source: str, generated: str) -> bool:
    try:
        # Premise = source text (truncated), hypothesis = generated summary
        result = nli_pipe(
            {"text": source[:512], "text_pair": generated},
            truncation=True,
        )
        label = result["label"].lower()
        score = result["score"]
        return label == "contradiction" and score > NLI_CONTRADICTION_THRESHOLD
    except Exception:
        return False


def classify_failure(paper: dict, generated: str, nli_pipe) -> str:
    source = paper["full_text"][:2000]
    ground_truth = paper["summary"]
    title = paper.get("title", "")

    # 1. Hallucination — NLI contradiction between source and generated summary
    if classify_hallucination(nli_pipe, source, generated):
        return "hallucination"

    # 2. Length mismatch
    if abs(len(generated) - len(ground_truth)) > LENGTH_MISMATCH_THRESHOLD:
        return "length_mismatch"

    # 3. Missing contribution — key title terms absent from summary
    keywords = title_keywords(title)
    if keywords:
        gen_lower = generated.lower()
        covered = sum(1 for kw in keywords if kw in gen_lower)
        if covered / len(keywords) < KEYWORD_COVERAGE_THRESHOLD:
            return "missing_contribution"

    # 4. Scope creep — background/intro language dominates
    if SCOPE_CREEP_PATTERNS.search(generated):
        return "scope_creep"

    # 5. Default — vague, loses specifics
    return "over_generalization"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}\n")

    print("Loading validation papers...")
    papers = load_val_papers()
    references = [p["summary"] for p in papers]
    print(f"  {len(papers)} papers loaded.\n")

    print("Generating/loading summaries...")
    summaries = generate_all_summaries(papers, device)

    print("\nComputing per-paper ROUGE-1...")
    rouge_scores = per_paper_rouge1(summaries, references)

    # Sort by ROUGE-1 ascending, take bottom N
    ranked = sorted(zip(rouge_scores, papers, summaries), key=lambda x: x[0])
    bottom = ranked[:BOTTOM_N]
    print(f"  Lowest ROUGE-1 in bottom {BOTTOM_N}: {bottom[0][0]:.4f}")
    print(f"  Threshold (paper #{BOTTOM_N}): {bottom[-1][0]:.4f}")

    # Load NLI pipeline
    print(f"\nLoading NLI model ({NLI_MODEL})...")
    nli_pipe = pipeline(
        "text-classification",
        model=NLI_MODEL,
        device=0 if device.type == "cuda" else -1,
    )

    # Classify each failure
    print(f"\nClassifying {BOTTOM_N} failure cases...")
    categories = ["hallucination", "length_mismatch", "missing_contribution", "scope_creep", "over_generalization"]
    counts = {c: 0 for c in categories}
    annotated = []

    for i, (score, paper, generated) in enumerate(bottom):
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{BOTTOM_N}")
        failure = classify_failure(paper, generated, nli_pipe)
        counts[failure] += 1
        annotated.append({
            "rank": i + 1,
            "rouge1": round(score, 4),
            "title": paper.get("title", ""),
            "failure_mode": failure,
            "generated_length": len(generated),
            "ground_truth_length": len(paper["summary"]),
            "length_delta": len(generated) - len(paper["summary"]),
            "generated": generated,
            "ground_truth": paper["summary"],
        })

    total = BOTTOM_N
    failure_modes = {
        c: {"count": counts[c], "percent": round(counts[c] / total * 100, 1)}
        for c in categories
    }

    # ── Print table ───────────────────────────────────────────────────────────
    label_map = {
        "hallucination": "Hallucination",
        "length_mismatch": "Length mismatch",
        "missing_contribution": "Missing contribution",
        "scope_creep": "Scope creep",
        "over_generalization": "Over-generalization",
    }
    sorted_modes = sorted(failure_modes.items(), key=lambda x: x[1]["count"], reverse=True)

    print(f"\nError Analysis of {BOTTOM_N} Lowest-ROUGE Papers:")
    print(f"\n  {'Category':<24} | {'Count':>5} | {'%':>6}")
    print("  " + "-" * 42)
    for mode, stats in sorted_modes:
        print(f"  {label_map[mode]:<24} | {stats['count']:>5} | {stats['percent']:>5.1f}%")

    top1, top2 = sorted_modes[0], sorted_modes[1]
    insight1 = f"Model most often fails due to {label_map[top1[0]].lower()} ({top1[1]['percent']}% of errors)."
    insight2 = f"Second most common: {label_map[top2[0]].lower()} ({top2[1]['percent']}%)."
    print(f"\n  Top insight:    {insight1}")
    print(f"  Second insight: {insight2}")

    # ── 3 example failures ────────────────────────────────────────────────────
    examples = annotated[:3]
    print("\n" + "=" * 65)
    print("3 Example Failures:")
    for ex in examples:
        print(f"\n  [{ex['rank']}] ROUGE-1={ex['rouge1']:.4f}  Failure: {label_map[ex['failure_mode']]}")
        print(f"  Title:      {ex['title'][:70]}")
        print(f"  Generated:  {ex['generated'][:200]}...")
        print(f"  Ground truth: {ex['ground_truth'][:200]}...")
    print("=" * 65)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    recommendations = [
        "Add length penalty to generation to enforce conciseness (length_penalty > 1.0)",
        "Pre-identify key contributions using keyword extraction before summarizing",
        "Implement NLI-based hallucination filtering as a post-processing step",
        "Fine-tune on contrastive pairs (generic vs. specific summaries) to reduce over-generalization",
        "Add introduction-vs-contribution classifier to steer attention away from background",
    ]

    output = {
        "total_errors_analyzed": total,
        "model": FINETUNED_DIR,
        "failure_modes": failure_modes,
        "insights": [insight1, insight2],
        "recommendations": recommendations,
        "example_failures": examples[:3],
        "all_failures": annotated,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Full per-paper summaries cached at {CACHE_FILE}")


if __name__ == "__main__":
    main()
