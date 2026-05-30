"""
Upload the fine-tuned INT8 quantized BART model to Hugging Face Hub.

Before running:
  1. Create a Hugging Face account at https://huggingface.co
  2. Generate an access token: https://huggingface.co/settings/tokens
  3. Run: huggingface-cli login (or hf auth login)
  4. Update MODEL_NAME below to your desired username/model-name

Then run:
  python3 upload_to_hf.py
"""
from transformers import BartForConditionalGeneration, AutoTokenizer
import torch
import os
from dotenv import load_dotenv

load_dotenv()

# Update this to your HF username and desired model name
# Format: "username/model-name"
HF_MODEL_NAME = "s4mt4r3/arxiv-bart-finetuned-int8"

# Local directories
QUANTIZED_DIR = "./quantized_arxiv_bart"
FINETUNED_DIR = "./fine_tuned_arxiv_bart"


def main():
    print(f"Uploading quantized BART to {HF_MODEL_NAME}...")

    # Load tokenizer from fine-tuned model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)

    # Load INT8 quantized model
    print("Loading INT8 quantized model...")
    torch.backends.quantized.engine = "qnnpack"
    model = BartForConditionalGeneration.from_pretrained(FINETUNED_DIR)
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    model.eval()

    # Get HF token from environment or use cached login
    hf_token = os.getenv("HF_TOKEN")

    # Push to HF Hub
    print("\nPushing model to Hugging Face Hub...")
    model.push_to_hub(HF_MODEL_NAME, private=False, token=hf_token)

    print("Pushing tokenizer to Hugging Face Hub...")
    tokenizer.push_to_hub(HF_MODEL_NAME, private=False, token=hf_token)

    # Create a model card
    model_card = f"""---
license: apache-2.0
tags:
  - summarization
  - arxiv
  - bart
  - int8
  - quantization
---

# ArXiv Paper Summarizer (BART Fine-tuned + INT8)

Fine-tuned `facebook/bart-base` on 2,596 ArXiv papers (cs.CL) with dynamic INT8 quantization.

## Model Details

- **Base model:** facebook/bart-base
- **Training data:** 2,000 ArXiv papers from cs.CL
- **Quantization:** Dynamic INT8 via torch.quantization
- **Size:** 286 MB (vs 532 MB original)
- **Speedup:** 1.42x faster inference
- **Quality:** 3.44% ROUGE-1 loss

## Performance

| Metric | Value |
|--------|-------|
| ROUGE-1 | 0.5596 |
| ROUGE-L | 0.5499 |
| BERTScore F1 | 0.9187 |
| Inference latency | 2,803 ms (100 papers) |

## Usage

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="{HF_MODEL_NAME}")
result = summarizer("Your paper abstract or text here...", max_length=150, min_length=50)
print(result[0]["summary_text"])
```

Or with the Streamlit app:

```bash
git clone https://github.com/yourusername/paper-summarizer
cd paper-summarizer
pip install -r requirements.txt
streamlit run app.py
```

## License

Apache License 2.0
"""

    print("\nSaving model card...")
    with open("MODEL_CARD.md", "w") as f:
        f.write(model_card)

    print("\n✅ Upload complete!")
    print(f"Model available at: https://huggingface.co/{HF_MODEL_NAME}")
    print("\nTo use in your code:")
    print(
        f"  model = BartForConditionalGeneration.from_pretrained('{HF_MODEL_NAME}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{HF_MODEL_NAME}')")


if __name__ == "__main__":
    main()
