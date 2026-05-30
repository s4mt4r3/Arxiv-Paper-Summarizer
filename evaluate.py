"""
Evaluate fine-tuned BART vs baseline facebook/bart-base using ROUGE metrics.
"""
import json
import os
import time
import torch
from dotenv import load_dotenv
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

FINETUNED_DIR = "./fine_tuned_arxiv_bart"
BASELINE_MODEL = "facebook/bart-base"
DATA_FILE = "arxiv_papers.json"
OUTPUT_FILE = "results/evaluation_results.json"

VAL_SIZE = 600
MAX_INPUT_LEN = 512
MAX_SUMMARY_LEN = 150
MIN_SUMMARY_LEN = 50
NUM_BEAMS = 4
EVAL_SAMPLE = 200   # evaluate on first N val papers (full 600 is slow on CPU/MPS)


def load_val_papers(path: str, val_size: int = VAL_SIZE) -> list:
    with open(path) as f:
        papers = json.load(f)["papers"]
    usable = [p for p in papers if p.get("full_text", "").strip() and p.get("summary", "").strip()]
    train_end = len(usable) - val_size
    return usable[train_end : train_end + val_size]


def generate_summaries(model, tokenizer, papers: list, device, label: str) -> list:
    model.eval()
    summaries = []
    for i, paper in enumerate(papers):
        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i+1}/{len(papers)}")
        inputs = tokenizer(
            paper["full_text"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=MAX_SUMMARY_LEN,
                min_length=MIN_SUMMARY_LEN,
                num_beams=NUM_BEAMS,
                early_stopping=True,
            )
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries


def compute_rouge(predictions: list, references: list) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1_scores, rl_scores = [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_scores.append(scores["rouge1"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)
    return {
        "rouge1": sum(r1_scores) / len(r1_scores),
        "rougeL": sum(rl_scores) / len(rl_scores),
    }


def load_model(model_path: str, device):
    print(f"  Loading model from {model_path}...")
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
    return model, tokenizer


def main():
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    print(f"\nLoading validation papers...")
    val_papers = load_val_papers(DATA_FILE)
    eval_papers = val_papers[:EVAL_SAMPLE]
    references = [p["summary"] for p in eval_papers]
    print(f"  Evaluating on {len(eval_papers)} papers.")

    # ── Baseline ──────────────────────────────────────────────────────────────
    print(f"\n[1/2] Baseline: {BASELINE_MODEL}")
    baseline_model, baseline_tokenizer = load_model(BASELINE_MODEL, device)
    t0 = time.time()
    baseline_summaries = generate_summaries(baseline_model, baseline_tokenizer, eval_papers, device, "baseline")
    baseline_time = time.time() - t0
    baseline_rouge = compute_rouge(baseline_summaries, references)
    del baseline_model  # free memory

    # ── Fine-tuned ────────────────────────────────────────────────────────────
    print(f"\n[2/2] Fine-tuned: {FINETUNED_DIR}")
    ft_model, ft_tokenizer = load_model(FINETUNED_DIR, device)
    t0 = time.time()
    ft_summaries = generate_summaries(ft_model, ft_tokenizer, eval_papers, device, "fine-tuned")
    ft_time = time.time() - t0
    ft_rouge = compute_rouge(ft_summaries, references)
    del ft_model

    # ── Results ───────────────────────────────────────────────────────────────
    def improvement(base, ft):
        return ((ft - base) / base) * 100 if base > 0 else 0.0

    r1_imp = improvement(baseline_rouge["rouge1"], ft_rouge["rouge1"])
    rl_imp = improvement(baseline_rouge["rougeL"], ft_rouge["rougeL"])

    print("\n" + "=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Baseline  ROUGE-1 : {baseline_rouge['rouge1']:.4f}")
    print(f"  Fine-tuned ROUGE-1: {ft_rouge['rouge1']:.4f}")
    print(f"  Improvement       : {r1_imp:+.2f}%")
    print()
    print(f"  Baseline  ROUGE-L : {baseline_rouge['rougeL']:.4f}")
    print(f"  Fine-tuned ROUGE-L: {ft_rouge['rougeL']:.4f}")
    print(f"  Improvement       : {rl_imp:+.2f}%")
    print()
    print(f"  Baseline  inference time : {baseline_time:.1f}s ({len(eval_papers)} papers)")
    print(f"  Fine-tuned inference time: {ft_time:.1f}s ({len(eval_papers)} papers)")
    print("=" * 55)

    results = {
        "num_eval_papers": len(eval_papers),
        "baseline": {
            "model": BASELINE_MODEL,
            "rouge1": baseline_rouge["rouge1"],
            "rougeL": baseline_rouge["rougeL"],
            "inference_time_s": baseline_time,
            "summaries": baseline_summaries[:5],  # sample
        },
        "finetuned": {
            "model": FINETUNED_DIR,
            "rouge1": ft_rouge["rouge1"],
            "rougeL": ft_rouge["rougeL"],
            "inference_time_s": ft_time,
            "summaries": ft_summaries[:5],
        },
        "improvement": {
            "rouge1_pct": r1_imp,
            "rougeL_pct": rl_imp,
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
