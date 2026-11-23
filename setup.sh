#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || python3 -m pip install --user uv

uv venv --python 3.10 .venv
source .venv/bin/activate

export UV_HTTP_TIMEOUT=600
# cu130 needs NVIDIA driver 580+ (CERV -> 580)
uv pip install "torch==2.14.1+cu130" \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple
uv pip install -r requirements.txt
python -m spacy download en_core_web_sm

mkdir -p data
for task in causality-detection causal-candidate-extraction causality-identification; do
  for split in train dev; do
    file="$task-$split.jsonl"
    [ -f "data/$file" ] || curl -fsSL -o "data/$file" \
      "https://zenodo.org/api/records/18183665/files/$file/content"
  done
done
