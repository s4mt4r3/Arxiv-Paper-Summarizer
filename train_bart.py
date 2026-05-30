import json
import os
import torch

# Force CPU — MPS is slower than CPU for BART seq2seq on Apple Silicon
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

from torch.utils.data import Dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
)

DATA_FILE = "arxiv_papers.json"
OUTPUT_DIR = "./fine_tuned_arxiv_bart"
MODEL_NAME = "facebook/bart-base"

MAX_INPUT_LEN = 256
MAX_TARGET_LEN = 96
TRAIN_SIZE = 1000
VAL_SIZE = 300
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2   # effective batch = 4 * 2 = 8
LEARNING_RATE = 3e-5
EPOCHS = 2
WARMUP_STEPS = 100


class ArxivDataset(Dataset):
    def __init__(self, papers, tokenizer):
        self.tokenizer = tokenizer
        # Filter out papers missing text or summary
        self.papers = [
            p for p in papers
            if p.get("full_text", "").strip() and p.get("summary", "").strip()
        ]

    def __len__(self):
        return len(self.papers)

    def __getitem__(self, idx):
        paper = self.papers[idx]
        inputs = self.tokenizer(
            paper["full_text"],
            max_length=MAX_INPUT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            paper["summary"],
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = targets["input_ids"].squeeze()
        # Replace padding token id with -100 so loss ignores pad positions
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
        }


def load_papers(path: str):
    with open(path) as f:
        data = json.load(f)
    return data["papers"]


def main():
    print(f"Loading data from {DATA_FILE}...")
    papers = load_papers(DATA_FILE)
    print(f"  Total papers loaded: {len(papers)}")

    if len(papers) < TRAIN_SIZE + VAL_SIZE:
        print(
            f"  Warning: only {len(papers)} papers available; "
            f"adjusting split to {len(papers) - VAL_SIZE} train / {VAL_SIZE} val."
        )
        actual_train = len(papers) - VAL_SIZE
    else:
        actual_train = TRAIN_SIZE

    train_papers = papers[:actual_train]
    val_papers = papers[actual_train : actual_train + VAL_SIZE]
    print(f"  Train: {len(train_papers)}  |  Val: {len(val_papers)}")

    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = ArxivDataset(train_papers, tokenizer)
    val_dataset = ArxivDataset(val_papers, tokenizer)
    print(f"  Effective train samples: {len(train_dataset)}")
    print(f"  Effective val samples:   {len(val_dataset)}")

    print(f"Loading model from {MODEL_NAME}...")
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        fp16=False,
        logging_steps=25,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting training...\n")
    train_result = trainer.train()

    print("\nSaving best model and tokenizer...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total steps:        {train_result.global_step}")
    print(f"  Final train loss:   {train_result.training_loss:.4f}")

    # Pull per-epoch metrics from trainer log history
    epoch_stats = []
    for entry in trainer.state.log_history:
        if "eval_loss" in entry:
            epoch_stats.append(
                {
                    "epoch": entry.get("epoch"),
                    "eval_loss": entry["eval_loss"],
                }
            )
        if "loss" in entry and "epoch" in entry:
            epoch_stats.append(
                {
                    "epoch": entry["epoch"],
                    "train_loss": entry["loss"],
                }
            )

    # Write loss curve to JSON for later plotting
    curve_path = os.path.join(OUTPUT_DIR, "loss_curve.json")
    with open(curve_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"\n  Loss curve saved to: {curve_path}")

    # Print epoch-by-epoch eval losses
    print("\n  Eval loss per epoch:")
    best_epoch, best_loss = None, float("inf")
    for entry in trainer.state.log_history:
        if "eval_loss" in entry:
            epoch = entry.get("epoch")
            loss = entry["eval_loss"]
            print(f"    Epoch {epoch:.0f}: eval_loss = {loss:.4f}")
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch

    print(f"\n  Best epoch: {best_epoch:.0f}  |  Best eval_loss: {best_loss:.4f}")
    print(f"  Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
