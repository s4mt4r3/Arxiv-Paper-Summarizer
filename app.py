import time
import os
import json
import torch
import streamlit as st
from dotenv import load_dotenv
from transformers import AutoTokenizer, BartForConditionalGeneration
import arxiv as arxiv_lib

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

# Try HF Hub first (for Streamlit Cloud), fall back to local
HF_MODEL_NAME = "st4r3/arxiv-bart-finetuned-int8"
FINETUNED_DIR = "./fine_tuned_arxiv_bart"
QUANTIZED_DIR = "./quantized_arxiv_bart"
MAX_INPUT_LEN = 512
MAX_SUMMARY_LEN = 150
MIN_SUMMARY_LEN = 50
NUM_BEAMS = 4


def model_size_mb(path: str) -> float:
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(path)
        for f in files
    )
    return total / (1024 ** 2)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    try:
        hf_token = st.secrets.get("HF_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, token=hf_token)
        model = BartForConditionalGeneration.from_pretrained(HF_MODEL_NAME, token=hf_token)
        model.eval()
        model_label = "INT8 Quantized (HF Hub)"
        size_mb = 286
        return model, tokenizer, model_label, size_mb
    except Exception as e:
        st.error(f"Failed to load model from HF Hub: {e}")
        st.info("Make sure HF_TOKEN is set in Streamlit secrets.")
        st.stop()


def summarize(model, tokenizer, text: str) -> tuple[str, float]:
    inputs = tokenizer(
        text,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        return_tensors="pt",
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=MAX_SUMMARY_LEN,
            min_length=MIN_SUMMARY_LEN,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary, latency_ms


def fetch_arxiv_paper(arxiv_id: str) -> dict | None:
    try:
        client = arxiv_lib.Client()
        search = arxiv_lib.Search(id_list=[arxiv_id.strip()])
        results = list(client.results(search))
        if not results:
            return None
        r = results[0]
        return {
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "abstract": r.summary,
            "text": r.summary[:2000],
        }
    except Exception as e:
        st.error(f"ArXiv fetch error: {e}")
        return None


def main():
    st.set_page_config(page_title="ArXiv Paper Summarizer", layout="wide")

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("ArXiv Paper Summarizer")
    st.markdown("**Fine-Tuned BART on ArXiv Papers**")

    model, tokenizer, model_label, size_mb = load_model()

    # Model info bar
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "BART-base")
    col2.metric("Mode", model_label)
    col3.metric("Model size", f"{size_mb:.0f} MB")

    st.divider()

    # ── Benchmark results (if available) ─────────────────────────────────────
    with st.expander("Model details & benchmark results"):
        if os.path.exists("results/quantization_results.json"):
            with open("results/quantization_results.json") as f:
                qr = json.load(f)
            c1, c2, c3 = st.columns(3)
            c1.metric("Speedup", f"{qr['latency']['speedup']:.1f}x")
            c2.metric("Size reduction",
                      f"{qr['model_size']['size_reduction_pct']:.0f}%")
            c3.metric("ROUGE-1 accuracy loss",
                      f"{qr['rouge']['accuracy_loss_pct']:.2f}%")
        if os.path.exists("results/evaluation_results.json"):
            with open("results/evaluation_results.json") as f:
                er = json.load(f)
            st.markdown("**ROUGE vs baseline (facebook/bart-base):**")
            c1, c2 = st.columns(2)
            c1.metric(
                "ROUGE-1 improvement",
                f"{er['finetuned']['rouge1']:.4f}",
                delta=f"{er['improvement']['rouge1_pct']:+.2f}%",
            )
            c2.metric(
                "ROUGE-L improvement",
                f"{er['finetuned']['rougeL']:.4f}",
                delta=f"{er['improvement']['rougeL_pct']:+.2f}%",
            )
        if not os.path.exists("results/quantization_results.json") and not os.path.exists("results/evaluation_results.json"):
            st.info(
                "Run evaluate.py and quantize.py to populate benchmark results here.")

    # ── Input ─────────────────────────────────────────────────────────────────
    input_mode = st.radio(
        "Input method",
        ["Paste ArXiv ID", "Paste paper text"],
        horizontal=True,
    )

    paper_title = None
    paper_authors = None
    ground_truth = None
    input_text = ""

    if input_mode == "Paste ArXiv ID":
        arxiv_id = st.text_input(
            "ArXiv ID",
            placeholder="e.g. arXiv:2605.25593",
            help="Enter the ArXiv paper ID — accepts arXiv:XXXX.XXXXX or plain XXXX.XXXXX format",
        )
        if arxiv_id:
            # Strip the arXiv: prefix if present (case-insensitive)
            clean_id = arxiv_id.strip()
            if clean_id.lower().startswith("arxiv:"):
                clean_id = clean_id[6:]
            with st.spinner("Fetching paper from ArXiv..."):
                paper = fetch_arxiv_paper(clean_id)
            if paper:
                paper_title = paper["title"]
                paper_authors = paper["authors"]
                ground_truth = paper["abstract"]
                input_text = paper["text"]
                st.success(f"**{paper_title}**")
                st.caption("Authors: " + ", ".join(paper_authors[:5]))
            else:
                st.error("Paper not found. Check the ID and try again.")
    else:
        input_text = st.text_area(
            "Paste paper text (max 2000 chars)",
            height=200,
            max_chars=2000,
            placeholder="Paste the introduction or abstract of a paper here...",
        )

    # ── Generate ──────────────────────────────────────────────────────────────
    st.divider()

    generate_btn = st.button(
        "Generate Summary",
        type="primary",
        disabled=not input_text.strip(),
    )

    if generate_btn and input_text.strip():
        with st.spinner("Generating summary..."):
            summary, latency_ms = summarize(model, tokenizer, input_text)

        st.subheader("Generated Summary")
        st.write(summary)

        col1, col2 = st.columns(2)
        col1.metric("Inference time", f"{latency_ms:.0f} ms")
        col2.metric("Timestamp", time.strftime("%H:%M:%S"))

        if ground_truth:
            st.divider()
            st.subheader("Ground Truth Abstract")
            st.write(ground_truth)


if __name__ == "__main__":
    main()
