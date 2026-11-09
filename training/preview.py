"""Preview a saved LoRA adapter locally without changing inference files."""
import argparse
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--adapter', type=Path, required=True)
p.add_argument('--model', type=Path, default=Path('models/SmolLM2-135M-Instruct'))
p.add_argument('--prompt', default='What is the demonstration robot called?')
a = p.parse_args()
assert torch.cuda.is_available(), 'CUDA required'
torch.set_num_threads(2)
tokenizer = AutoTokenizer.from_pretrained(a.adapter, local_files_only=True)
base = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
    dtype=torch.float32, attn_implementation='eager')
model = PeftModel.from_pretrained(base, a.adapter, local_files_only=True).to('cuda').eval()
inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': a.prompt}],
    add_generation_prompt=True, tokenize=True, return_tensors='pt').to('cuda')
with torch.no_grad():
    output = model.generate(inputs, attention_mask=torch.ones_like(inputs),
        max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True))
