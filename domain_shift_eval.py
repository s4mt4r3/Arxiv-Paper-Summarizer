"""
Domain-shift evaluation: benchmark the fine-tuned BART across
CS.CL (in-domain), CS.CV, CS.LG, and STAT.ML (out-of-domain).

Papers for out-of-domain categories are fetched live from the ArXiv API
(abstracts only — no PDFs needed) and cached to domain_papers_cache.json.
"""
import json
import os
import time
import sys
import requests
import xml.etree.ElementTree as ET
import statistics
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from rouge_score import rouge_scorer as rouge_lib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

DATA_FILE = "arxiv_papers.json"
CACHE_FILE = "domain_papers_cache.json"
COMPARISON_FILE = "results/model_comparison.json"
OUTPUT_FILE = "results/domain_shift_analysis.json"
CHART_FILE = "plots/domain_shift_chart.png"

PAPERS_PER_DOMAIN = 200
MAX_INPUT_LEN = 512
MAX_SUMMARY_LEN = 150
MIN_SUMMARY_LEN = 50
NUM_BEAMS = 4
LATENCY_SAMPLE = 10

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}
API_PAGE_SIZE = 100
API_DELAY = 15  # seconds between pages (ArXiv rate limit)

DOMAINS = [
    {"key": "cs.CL",  "label": "CS.CL (in-domain)",  "query": "cat:cs.CL"},
    {"key": "cs.CV",  "label": "CS.CV",               "query": "cat:cs.CV"},
    {"key": "cs.LG",  "label": "CS.LG",               "query": "cat:cs.LG"},
    {"key": "stat.ML","label": "STAT.ML",              "query": "cat:stat.ML"},
]


# ── ArXiv fetcher ─────────────────────────────────────────────────────────────

def fetch_arxiv_abstracts(query: str, n: int) -> list[dict]:
    """Fetch up to n papers (title + abstract) from the ArXiv API."""
    papers, offset = [], 0
    while len(papers) < n:
        want = min(API_PAGE_SIZE, n - len(papers))
        try:
            resp = requests.get(
                ARXIV_API,
                params={"search_query": query, "sortBy": "submittedDate",
                        "sortOrder": "descending", "start": offset, "max_results": want},
                timeout=30,
            )
        except requests.RequestException as e:
            print(f"  [Network error] {e}", file=sys.stderr)
            break

        if resp.status_code != 200:
            print(f"  [API {resp.status_code}] sleeping 30s...", file=sys.stderr)
            time.sleep(30)
            continue

        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", NS)
        if not entries:
            break

        for entry in entries:
            title = (entry.findtext("atom:title", default="", namespaces=NS) or "").strip()
            abstract = (entry.findtext("atom:summary", default="", namespaces=NS) or "").strip()
            if title and abstract:
                papers.append({"title": title, "full_text": abstract[:2000], "summary": abstract})

        offset += want
        if len(papers) < n and offset < n + API_PAGE_SIZE:
            print(f"    fetched {len(papers)}/{n} — sleeping {API_DELAY}s...")
            time.sleep(API_DELAY)

    return papers[:n]


def load_or_fetch_domain_papers() -> dict[str, list]:
    """Load cached domain papers, fetching any missing categories."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
    else:
        cache = {}

    changed = False
    for domain in DOMAINS:
        key = domain["key"]
        if key == "cs.CL":
            # Use the existing scraped dataset for in-domain
            continue
        if key in cache and len(cache[key]) >= PAPERS_PER_DOMAIN:
            print(f"  {key}: {len(cache[key])} papers loaded from cache.")
            continue
        print(f"  Fetching {PAPERS_PER_DOMAIN} papers for {key}...")
        papers = fetch_arxiv_abstracts(domain["query"], PAPERS_PER_DOMAIN)
        print(f"    Got {len(papers)} papers.")
        cache[key] = papers
        changed = True

    if changed:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"  Cache saved to {CACHE_FILE}")

    return cache


def load_cscl_papers(n: int) -> list[dict]:
    """Load CS.CL papers from the training dataset (val split tail)."""
    with open(DATA_FILE) as f:
        papers = json.load(f)["papers"]
    usable = [p for p in papers if p.get("full_text", "").strip() and p.get("summary", "").strip()]
    val_start = max(0, len(usable) - 600)
    return usable[val_start : val_start + n]


def pick_model_id() -> str:
    if os.path.exists(COMPARISON_FILE):
        with open(COMPARISON_FILE) as f:
            data = json.load(f)
        best_name = data.get("best_model", "")
        for m in data.get("models", []):
            if m["name"] == best_name:
                print(f"  Best model from benchmark: {best_name} ({m['model_id']})")
                return m["model_id"]
    print("  results/model_comparison.json not found — using fine-tuned BART.")
    return "./fine_tuned_arxiv_bart"


# ── Inference ─────────────────────────────────────────────────────────────────

def build_model(model_id: str, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    model.eval()
    prefix = "summarize: " if "t5" in model_id.lower() else ""
    return model, tokenizer, prefix


def summarize_batch(model, tokenizer, prefix: str, papers: list, device) -> tuple[list[str], float]:
    summaries, t0 = [], time.perf_counter()
    for i, paper in enumerate(papers):
        if (i + 1) % 50 == 0:
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
    latency_ms = (time.perf_counter() - t0) / len(papers) * 1000
    return summaries, latency_ms


def compute_rouge(predictions: list, references: list) -> dict:
    scorer = rouge_lib.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1, rl = [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {"rouge1": sum(r1) / len(r1), "rougeL": sum(rl) / len(rl)}


def compute_bertscore(predictions: list, references: list) -> float:
    try:
        from bert_score import score as bscore
        _, _, F1 = bscore(predictions, references, model_type="roberta-large", lang="en", verbose=False)
        return float(F1.mean())
    except Exception as e:
        print(f"  BERTScore skipped: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}\n")

    # Fetch / load domain papers
    print("Loading domain papers...")
    cache = load_or_fetch_domain_papers()

    domain_papers = {}
    for domain in DOMAINS:
        key = domain["key"]
        if key == "cs.CL":
            domain_papers[key] = load_cscl_papers(PAPERS_PER_DOMAIN)
            print(f"  cs.CL: {len(domain_papers[key])} papers from local dataset.")
        else:
            available = cache.get(key, [])
            domain_papers[key] = available[:PAPERS_PER_DOMAIN]
            print(f"  {key}: {len(domain_papers[key])} papers.")

    # Load model once
    model_id = pick_model_id()
    print(f"\nLoading model: {model_id}")
    model, tokenizer, prefix = build_model(model_id, device)

    # Evaluate each domain
    domain_results = []
    for domain in DOMAINS:
        key = domain["key"]
        papers = domain_papers.get(key, [])
        if not papers:
            print(f"\n  Skipping {key} — no papers available.")
            continue

        print(f"\n[{domain['label']}] — {len(papers)} papers")
        summaries, latency_ms = summarize_batch(model, tokenizer, prefix, papers, device)
        references = [p["summary"] for p in papers]
        rouge = compute_rouge(summaries, references)

        print(f"  ROUGE-1: {rouge['rouge1']:.4f}  ROUGE-L: {rouge['rougeL']:.4f}  Latency: {latency_ms:.0f}ms")
        print(f"  Computing BERTScore...")
        bert_f1 = compute_bertscore(summaries, references)

        domain_results.append({
            "key": key,
            "label": domain["label"],
            "num_papers": len(papers),
            "rouge1": round(rouge["rouge1"], 4),
            "rougeL": round(rouge["rougeL"], 4),
            "bertscore_f1": round(bert_f1, 4) if bert_f1 else None,
            "latency_ms": round(latency_ms, 1),
            "example_summaries": summaries[:3],
        })

    # ── Analysis ──────────────────────────────────────────────────────────────
    cscl = next((r for r in domain_results if r["key"] == "cs.CL"), None)
    out_of_domain = [r for r in domain_results if r["key"] != "cs.CL"]

    indomain_rouge = cscl["rouge1"] if cscl else None
    ood_rouge_avg = (
        sum(r["rouge1"] for r in out_of_domain) / len(out_of_domain)
        if out_of_domain else None
    )
    gap_pct = (
        (indomain_rouge - ood_rouge_avg) / indomain_rouge * 100
        if indomain_rouge and ood_rouge_avg else None
    )

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'Domain':<22} | {'Papers':>6} | {'ROUGE-1':>7} | {'ROUGE-L':>7} | {'BERTScore-F1':>12} | {'Latency(ms)':>11}")
    print("-" * 75)
    for r in domain_results:
        bs = f"{r['bertscore_f1']:.4f}" if r["bertscore_f1"] else "   N/A"
        print(f"{r['label']:<22} | {r['num_papers']:>6} | {r['rouge1']:>7.4f} | {r['rougeL']:>7.4f} | {bs:>12} | {r['latency_ms']:>11.0f}")
    print("=" * 75)

    analysis_text = (
        f"Domain shift analysis:\n"
        f"  In-domain (CS.CL) ROUGE-1:   {indomain_rouge:.4f}\n"
        f"  Out-of-domain average ROUGE-1: {ood_rouge_avg:.4f}\n"
        f"  Performance gap: {gap_pct:.1f}% degradation on unfamiliar domains\n"
        f"  Suggests model overfits to NLP-specific language patterns."
    ) if gap_pct is not None else "Insufficient data for gap analysis."
    print(f"\n{analysis_text}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    labels = [r["label"] for r in domain_results]
    rouge1_vals = [r["rouge1"] for r in domain_results]
    rougeL_vals = [r["rougeL"] for r in domain_results]
    colors = ["steelblue" if r["key"] == "cs.CL" else "tomato" for r in domain_results]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar([i - width/2 for i in x], rouge1_vals, width, label="ROUGE-1", color=colors, alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], rougeL_vals, width, label="ROUGE-L",
                   color=[c if c == "steelblue" else "salmon" for c in colors], alpha=0.65)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("ROUGE Score", fontsize=12)
    ax.set_title("ROUGE by ArXiv Domain  (blue = in-domain, red = out-of-domain)", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150)
    plt.close()
    print(f"\nChart saved to {CHART_FILE}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "model_id": model_id,
        "papers_per_domain": PAPERS_PER_DOMAIN,
        "indomain_rouge1": indomain_rouge,
        "ood_avg_rouge1": ood_rouge_avg,
        "performance_gap_pct": round(gap_pct, 2) if gap_pct else None,
        "analysis": analysis_text,
        "domains": domain_results,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
