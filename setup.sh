#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || python3 -m pip install --user uv

uv venv --python 3.10 .venv
source .venv/bin/activate
