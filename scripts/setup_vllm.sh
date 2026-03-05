#!/bin/bash
# filepath: scripts/setup_vllm.sh

echo "Setting up vLLM for pathology analysis..."

# Install vLLM
pip install vllm

# Download and cache vision model
python -c "
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

print('Downloading LLaVA model for pathology analysis...')
model_name = 'llava-hf/llava-1.5-7b-hf'
processor = LlavaNextProcessor.from_pretrained(model_name)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
print('Model downloaded and cached!')
"

echo "vLLM setup complete!"
echo "Start server with: python scripts/start_vllm_server.py"