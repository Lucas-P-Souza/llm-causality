import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.dataset import CausalityDataset
from src.models.causality_classifier import CausalityClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="roberta-large")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/subtask1_llm_only")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_dataset = CausalityDataset("train", tokenizer)
    model = CausalityClassifier(args.model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        seed=args.seed,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    train_result = trainer.train()

    dev_dataset = CausalityDataset("dev", tokenizer)
    output = trainer.predict(dev_dataset)
    logits = torch.tensor(output.predictions)
    y_pred = logits.argmax(dim=-1).tolist()
    y_true = output.label_ids.tolist()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    metrics = {
        "task": "subtask1",
        "model_name": args.model,
        "seed": args.seed,
        "dev_macro_f1": f1_score(y_true, y_pred, average="macro"),
        "dev_accuracy": accuracy_score(y_true, y_pred),
        "dev_precision": precision.tolist(),
        "dev_recall": recall.tolist(),
        "dev_f1": f1.tolist(),
        "train_loss": train_result.training_loss,
    }
    output_dir = Path(args.output_dir)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(
        {
            "index": [row["index"] for row in dev_dataset.rows],
            "y_true": y_true,
            "y_pred": y_pred,
            "probs": logits.softmax(dim=-1).tolist(),
        }
    ).to_json(output_dir / "dev_predictions.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
