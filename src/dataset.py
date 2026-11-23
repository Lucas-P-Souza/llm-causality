from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def read_split(split):
    path = DATA_DIR / f"causality-detection-{split}.jsonl"
    return pd.read_json(path, lines=True).to_dict("records")
