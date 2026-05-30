"""
Benchmark 5 pretrained summarization models against the fine-tuned BART.
Saves results to model_comparison.json and prints a comparison table.
"""
import json
import os
import time
import torch
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

DATA_FILE = "arxiv_papers.json"
OUTPUT_FILE = "results/model_comparison.json"
FINETUNED_DIR = "./fine_tuned_arxiv_bart"

EVAL_SAMPLE = 200   # increase to 1000 if you have time (each model takes ~15-30 min on MPS)
LATENCY_SAMPLE = 10
MAX_SUMMARY_LEN = 150
MIN_SUMMARY_LEN = 50
NUM_BEAMS = 4

MODELS = [
    {
        "name": "BART-base (fine-tuned)",
        "id": FINETUNED_DIR,
        "max_input": 512,
        "prefix": None,
        "led": False,
    },
    {
        "name": "PEGASUS-ArXiv",
        "id": "google/pegasus-arxiv",
        "max_input": 512,
        "prefix": None,
        "led": False,
    },
    {
        "name": "T5-base",
        "id": "google-t5/t5-base",
        "max_input": 512,
        "prefix": "summarize: ",
        "led": False,
    },
    {
        "name": "DistilBART-CNN-6-6",
        "id": "sshleifer/distilbart-cnn-6-6",
        "max_input": 512,
        "prefix": None,
        "led": False,
    },
    {
        "name": "LED-base-16384",
        "id": "allenai/led-base-16384",
        "max_input": 1024,
        "prefix": None,
        "led": True,
    },
]


def load_val_papers(path: str, eval_sample: int) -> list:
    with open(path) as f:
        papers = json.load(f)["papers"]
    usable = [p for p in papers if p.get("full_text", "").strip() and p.get("summary", "").strip()]
    # Use the tail of the dataset (same split as evaluate.py uses VAL_SIZE=600)
    val_start = len(usable) - 600
    val_papers = usable[val_start:]
    return val_papers[:eval_sample]


def model_size_mb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)


def generate_summary(model, tokenizer, text: str, cfg: dict, device) -> str:
    source = (cfg["prefix"] + text) if cfg["prefix"] else text
    inputs = tokenizer(
        source,
        max_length=cfg["max_input"],
        truncation=True,
        return_tensors="pt",
    ).to(device)

    gen_kwargs = dict(
        max_length=MAX_SUMMARY_LEN,
        min_length=MIN_SUMMARY_LEN,
        num_beams=NUM_BEAMS,
        early_stopping=True,
    )

    # LED requires global attention on the first token
    if cfg["led"]:
        global_mask = torch.zeros_like(inputs["input_ids"])
        global_mask[:, 0] = 1
        gen_kwargs["global_attention_mask"] = global_mask

    with torch.no_grad():
        output_ids = model.generate(inputs["input_ids"], **gen_kwargs)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def compute_rouge(predictions: list, references: list) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1, rl = [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {"rouge1": sum(r1) / len(r1), "rougeL": sum(rl) / len(rl)}


def benchmark_model(cfg: dict, papers: list, device) -> dict | None:
    print(f"\n{'='*60}")
    print(f"  Model: {cfg['name']}  ({cfg['id']})")
    print(f"{'='*60}")
    try:
        print("  Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(cfg["id"])
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg["id"]).to(device)
        model.eval()
        size_mb = model_size_mb(model)
        print(f"  Model size: {size_mb:.0f} MB")

        # Latency on first LATENCY_SAMPLE papers
        print(f"  Measuring latency ({LATENCY_SAMPLE} papers)...")
        t0 = time.perf_counter()
        for paper in papers[:LATENCY_SAMPLE]:
            generate_summary(model, tokenizer, paper["full_text"][:2000], cfg, device)
        latency_ms = (time.perf_counter() - t0) / LATENCY_SAMPLE * 1000

        # Full ROUGE evaluation
        print(f"  Generating summaries ({len(papers)} papers)...")
        summaries = []
        for i, paper in enumerate(papers):
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(papers)}")
            summaries.append(
                generate_summary(model, tokenizer, paper["full_text"][:2000], cfg, device)
            )

        references = [p["summary"] for p in papers]
        rouge = compute_rouge(summaries, references)

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

        return {
            "name": cfg["name"],
            "model_id": cfg["id"],
            "rouge1": rouge["rouge1"],
            "rougeL": rouge["rougeL"],
            "latency_ms": latency_ms,
            "size_mb": size_mb,
            "example_summaries": summaries[:5],
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Skipping {cfg['name']}.")
        return None


def print_table(results: list) -> None:
    print("\n" + "=" * 72)
    print(f"{'Model':<26} | {'ROUGE-1':>7} | {'ROUGE-L':>7} | {'Latency(ms)':>11} | {'Size(MB)':>8}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['name']:<26} | {r['rouge1']:>7.4f} | {r['rougeL']:>7.4f} | "
            f"{r['latency_ms']:>11.0f} | {r['size_mb']:>8.0f}"
        )
    print("=" * 72)


def main():
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    print(f"\nLoading {EVAL_SAMPLE} validation papers...")
    papers = load_val_papers(DATA_FILE, EVAL_SAMPLE)
    print(f"  Loaded {len(papers)} papers.")

    results = []
    for cfg in MODELS:
        result = benchmark_model(cfg, papers, device)
        if result:
            results.append(result)

    if not results:
        print("No models completed successfully.")
        return

    results.sort(key=lambda r: r["rouge1"], reverse=True)
    best = results[0]

    print_table(results)
    print(f"\nBest model: {best['name']} with ROUGE-1: {best['rouge1']:.4f}")

    output = {
        "eval_sample": EVAL_SAMPLE,
        "device": str(device),
        "models": results,
        "best_model": best["name"],
        "best_rouge1": best["rouge1"],
        "best_summaries": best["example_summaries"],
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
