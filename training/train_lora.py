#!/usr/bin/env python3
"""Offline, completion-only LoRA fine-tuning for the downloaded SmolLM2 weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def examples(path, tokenizer, max_length):
    result = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        messages = json.loads(line)['messages']
        if len(messages) < 2 or messages[-1]['role'] != 'assistant' or not messages[-1]['content'].strip():
            raise ValueError(f'{path}:{number}: end each conversation with an assistant answer')
        if any(m['role'] not in ('system', 'user', 'assistant') or not isinstance(m['content'], str) for m in messages):
            raise ValueError(f'{path}:{number}: invalid message')
        prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if tokens[:len(prefix)] != prefix or len(tokens) <= len(prefix):
            raise ValueError(f'{path}:{number}: chat template boundary mismatch')
        if len(tokens) > max_length:
            raise ValueError(f'{path}:{number}: {len(tokens)} tokens exceeds --max-length {max_length}; shorten the example')
        labels = [-100] * len(prefix) + tokens[len(prefix):]
        result.append((tokens, labels))
    if not result:
        raise ValueError(f'{path}: no examples')
    return result


def batch(example):
    tokens, labels = example
    return {
        'input_ids': torch.tensor([tokens], device='cuda'),
        'attention_mask': torch.ones((1, len(tokens)), device='cuda', dtype=torch.long),
        'labels': torch.tensor([labels], device='cuda'),
    }


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    total = count = 0
    for example in data:
        n = sum(x != -100 for x in example[1][1:])
        total += model(**batch(example)).loss.item() * n
        count += n
    return total / count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('models/SmolLM2-135M-Instruct'))
    p.add_argument('--data', type=Path, default=Path('training/demo.jsonl'))
    p.add_argument('--validation', type=Path, default=Path('training/demo-validation.jsonl'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.steps < 1 or args.accumulation < 1 or args.max_length < 2 or args.learning_rate <= 0:
        p.error('steps, accumulation, max-length and learning-rate must be positive')
    if args.output.exists():
        p.error('output already exists; choose a new directory to preserve previous runs')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: refusing CPU fallback')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda,
          'GPU:', torch.cuda.get_device_name(), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    data = examples(args.data, tokenizer, args.max_length)
    validation = examples(args.validation, tokenizer, args.max_length)
    # FP32 is inexpensive for 135M parameters and avoids FP16 gradient scaling.
    base = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32,
        attn_implementation='eager',
    ).to('cuda')
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        task_type='CAUSAL_LM', r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=['q_proj', 'v_proj'], bias='none',
    ))
    model.print_trainable_parameters()
    trainable = [v for v in model.parameters() if v.requires_grad]
    initial = [v.detach().cpu().clone() for v in trainable]
    before = evaluate(model, validation)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    history = []
    for step in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.accumulation):
            loss = model(**batch(random.choice(data))).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss_sum += loss.item()
            (loss / args.accumulation).backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append(loss_sum / args.accumulation)
        if step == 0 or (step + 1) % 5 == 0 or step + 1 == args.steps:
            print(f'step={step + 1} loss={history[-1]:.4f} grad_norm={norm.item():.4f}', flush=True)
    torch.cuda.synchronize()
    changed = any(not torch.equal(a, b.detach().cpu()) for a, b in zip(initial, trainable))
    if not changed:
        raise RuntimeError('Adapter parameters did not change')
    after = evaluate(model, validation)
    if not math.isfinite(after):
        raise RuntimeError('Non-finite validation loss')
    args.output.mkdir(parents=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    report = {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'base_source': json.loads((args.model / 'download-source.json').read_text()) if (args.model / 'download-source.json').exists() else None,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'adapter_parameters_changed': changed,
        'train_examples': len(data), 'validation_examples': len(validation),
        'validation_loss_before': before, 'validation_loss_after': after,
        'training_loss': history, 'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
        'dataset_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'validation_sha256': hashlib.sha256(args.validation.read_bytes()).hexdigest(),
    }
    (args.output / 'training-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('arguments', 'training_loss')}, indent=2))
    print('Adapter saved:', args.output, '\nInference GGUF has not been changed.')


if __name__ == '__main__':
    main()
