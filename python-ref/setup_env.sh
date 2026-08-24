#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "Creating virtual environment in $(pwd)/.venv..."
python3 -m venv .venv --system-site-packages

echo "Installing required dependencies..."
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY .venv/bin/pip install -r requirements.txt

echo "Verifying environment..."
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY .venv/bin/python -c "
import torch, transformers, onnxruntime, numpy, scipy
print('All core dependencies imported successfully!')
print(f'  torch: {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  onnxruntime: {onnxruntime.__version__}')
print(f'  numpy: {numpy.__version__}')
print(f'  scipy: {scipy.__version__}')
"
echo "Environment setup complete!"
