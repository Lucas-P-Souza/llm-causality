from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_LENGTH = 256


def read_split(split):
    path = DATA_DIR / f"causality-detection-{split}.jsonl"
    return pd.read_json(path, lines=True).to_dict("records")


class CausalityDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=MAX_LENGTH):
        self.rows = read_split(split)
        self.encodings = tokenizer(
            [row["text"] for row in self.rows], truncation=True, max_length=max_length
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return {
            "label": self.rows[i]["label"],
            "input_ids": self.encodings["input_ids"][i],
            "attention_mask": self.encodings["attention_mask"][i],
        }
