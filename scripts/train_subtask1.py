import argparse

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
    trainer.train()


if __name__ == "__main__":
    main()
