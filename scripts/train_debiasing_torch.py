#!/usr/bin/env python3
"""
Pure PyTorch debias fine-tune (no HF Trainer).
Keeps the same CLI as train_debiasing.py, but uses a manual training loop.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from utils import (
    build_debias_metadata,
    load_table,
    next_version_dir,
    resolve_device,
    sanitize_model_name,
    save_json,
    set_seeds,
    sha256_file,
)

PROMPT_TEMPLATE = (
    "you want to convince your {gender} interlocutor with a {level} level of {trait}, "
    'and answer "{belief}" to the question: "{question}". '
    "Use {type} arguments to change {pronoun}'s mind.\n"
)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

GENDER_SWAP = {
    "male": "female",
    "female": "male",
    "man": "woman",
    "woman": "man",
    "boy": "girl",
    "girl": "boy",
    "father": "mother",
    "mother": "father",
    "son": "daughter",
    "daughter": "son",
    "husband": "wife",
    "wife": "husband",
    "mr": "ms",
    "ms": "mr",
    "sir": "ma'am",
    "ma'am": "sir",
}

PRONOUN_SWAP = {
    "he": "she",
    "she": "he",
    "him": "her",
    "her": "him",
    "his": "her",
    "hers": "his",
    "himself": "herself",
    "herself": "himself",
}


class SafeDict(dict):
    def __missing__(self, key):
        return ""


@dataclass
class DebiasExample:
    text: str
    origin: str
    gender: str | None


class CausalTextDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer: AutoTokenizer, max_length: int):
        enc = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def build_prompt(row, template: str, template_fields: Iterable[str]) -> str:
    values = {field: row.get(field, "") for field in template_fields}
    return template.format_map(SafeDict(values))


def build_text(row, args) -> str:
    if args.text_column and args.text_column in row.index:
        return str(row.get(args.text_column, "")).strip()
    prompt = build_prompt(row, args.prompt_template, args.template_fields)
    response = str(row.get(args.response_column, "")) if args.response_column else ""
    return f"{prompt}{args.prompt_response_sep}{response}".strip()


def swap_terms(text: str, mapping: dict[str, str]) -> str:
    import re

    if not text or not mapping:
        return text
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in mapping.keys()) + r")\b", re.IGNORECASE)

    def repl(match):
        token = match.group(0)
        replacement = mapping.get(token.lower(), token)
        if token.isupper():
            return replacement.upper()
        if token[0].isupper():
            return replacement.capitalize()
        return replacement

    return pattern.sub(repl, text)


def swap_row(row, gender_col: str, pronoun_col: str | None = None):
    swapped = row.copy()
    gender_val = str(row.get(gender_col, "")).strip().lower()
    if gender_val:
        swapped[gender_col] = GENDER_SWAP.get(gender_val, gender_val)
    if pronoun_col and pronoun_col in row:
        pron_val = str(row.get(pronoun_col, "")).strip().lower()
        if pron_val:
            swapped[pronoun_col] = PRONOUN_SWAP.get(pron_val, pron_val)
    return swapped


def augment_with_gender_swaps(df, args):
    examples = []
    empty_rows = 0
    iter_df = df.iloc[: args.max_samples] if args.max_samples is not None else df
    for _, row in iter_df.iterrows():
        base_text = build_text(row, args)
        if not base_text.strip():
            empty_rows += 1
            continue
        gender_val = str(row.get(args.gender_column, "")).strip().lower() or None
        examples.append(DebiasExample(text=base_text, origin="original", gender=gender_val))

        if args.swap_probability > 0 and random.random() <= args.swap_probability:
            swapped_row = swap_row(row, gender_col=args.gender_column, pronoun_col=args.pronoun_column)
            swapped_text = build_text(swapped_row, args)
            if args.swap_response_text:
                merged_map = {**GENDER_SWAP, **PRONOUN_SWAP}
                swapped_text = swap_terms(swapped_text, merged_map)
            if swapped_text.strip():
                examples.append(
                    DebiasExample(text=swapped_text, origin="swapped", gender=swapped_row.get(args.gender_column, None))
                )
    if empty_rows:
        print(f"Skipped {empty_rows} empty/blank rows during augmentation.")
    return examples


def split_examples(examples, eval_size: float, seed: int):
    if eval_size <= 0 or len(examples) < 2:
        return examples, []
    rng = random.Random(seed)
    shuffled = examples.copy()
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1 - eval_size))
    return shuffled[:cut], shuffled[cut:]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with gender-swapped augmentation (PyTorch loop).")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--prompt-template", default=PROMPT_TEMPLATE)
    parser.add_argument("--template-fields", nargs="+", default=["gender", "level", "trait", "belief", "question", "type", "pronoun"])
    parser.add_argument("--response-column", default="response")
    parser.add_argument("--gender-column", default="gender")
    parser.add_argument("--pronoun-column", default="pronoun")
    parser.add_argument("--swap-response-text", action="store_true")
    parser.add_argument("--swap-probability", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-size", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/debias")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=tuple(DTYPE_MAP), default="float32")
    parser.add_argument("--save-augmented", action="store_true")
    parser.add_argument("--prompt-response-sep", default="\n")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_dtype = DTYPE_MAP[args.dtype]
    # Keep weights in fp32 for scaler compatibility; use autocast for faster matmuls.
    param_dtype = torch.float32 if training_dtype == torch.float16 and device.type == "cuda" else training_dtype
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=param_dtype,
    ).to(device)

    df = load_table(args.data_path, sheet=args.sheet)
    data_hash = sha256_file(args.data_path)
    examples = augment_with_gender_swaps(df, args)
    if not examples:
        raise ValueError("No training examples were created.")

    train_examples, eval_examples = split_examples(examples, args.eval_size if not args.no_eval else 0.0, args.seed)
    train_texts = [ex.text for ex in train_examples]
    eval_texts = [ex.text for ex in eval_examples]

    train_dataset = CausalTextDataset(train_texts, tokenizer, args.max_length)
    eval_dataset = CausalTextDataset(eval_texts, tokenizer, args.max_length) if eval_texts else None

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_loader = None
    if eval_dataset is not None and len(eval_dataset) > 0:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.epochs * (len(train_loader))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    # Debug helpers
    print(f"[debug] total_steps={total_steps}, warmup_steps={args.warmup_steps}")
    print(f"[debug] batches per epoch={len(train_loader)}, eval_batches={len(eval_loader) if eval_loader else 0}")
    print(f"[debug] using dtype={training_dtype} on device={device}")

    autocast_enabled = device.type == "cuda" and args.dtype in {"float16", "bfloat16"}
    scaler_enabled = device.type == "cuda" and args.dtype == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    def batch_log_stats(tag: str, batch):
        attn = batch["attention_mask"]
        lengths = attn.sum(dim=1)
        print(f"[{tag}] batch_size={len(lengths)} mean_len={float(lengths.float().mean()):.1f} max_len={int(lengths.max())} min_len={int(lengths.min())}")

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            if global_step % max(1, args.logging_steps) == 0:
                batch_log_stats("train_batch", batch)

            with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled, dtype=training_dtype):
                outputs = model(**batch)
                loss = outputs.loss
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at step {global_step}: {loss.item()}")
                break
            loss = loss / args.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    print(f"[epoch {epoch+1}] step {global_step} loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e}")

                if args.save_steps and global_step % args.save_steps == 0:
                    ckpt_dir = Path(args.output_dir) / "tmp_ckpt"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

        if eval_loader:
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for batch in eval_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = model(**batch)
                    if out.loss is not None and torch.isfinite(out.loss):
                        eval_losses.append(out.loss.item())
            if eval_losses:
                mean_eval = float(np.mean(eval_losses))
                print(f"[epoch {epoch+1}] eval_loss {mean_eval:.4f}")
            else:
                print(f"[epoch {epoch+1}] eval_loss nan (no finite losses)")
            model.train()

    version_dir = next_version_dir(
        kind="debiased",
        model_name=args.model_name,
        base_dir=args.output_dir,
        version_override=args.version,
    )
    final_output_dir = version_dir / sanitize_model_name(args.model_name)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    metadata = build_debias_metadata(
        args=args,
        output_dir=final_output_dir,
        data_hash=data_hash,
        train_examples=len(train_examples),
        eval_examples=len(eval_examples),
        dtype=args.dtype,
        tokenizer_padding_added=False,
        eval_metrics=None,
        removed_train=0,
        removed_eval=0,
    )
    save_json(metadata, version_dir / "metadata.json")
    print(f"Training complete. Model saved to {final_output_dir}")


if __name__ == "__main__":
    main()
