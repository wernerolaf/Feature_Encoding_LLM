#!/usr/bin/env python3
"""
Debias fine-tune using TRL's SFTTrainer (optionally with LoRA + 8bit).

CLI is intentionally similar to train_debiasing.py/torch for consistency.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

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
    parser = argparse.ArgumentParser(description="Fine-tune with TRL SFTTrainer + optional LoRA/8bit.")
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-size", type=float, default=0.2)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/debias")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=tuple(DTYPE_MAP), default="float16")
    parser.add_argument("--prompt-response-sep", default="\n")
    # LoRA / quantization knobs
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-peft", action="store_true", help="Force full-precision fine-tuning even if use-lora set.")
    parser.add_argument("--save-augmented", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)
    device = resolve_device(args.device)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Data
    df = load_table(args.data_path, sheet=args.sheet)
    data_hash = sha256_file(args.data_path)
    examples = augment_with_gender_swaps(df, args)
    if not examples:
        raise ValueError("No training examples were created.")

    train_examples, eval_examples = split_examples(examples, args.eval_size if not args.no_eval else 0.0, args.seed)
    train_texts = [ex.text for ex in train_examples]
    eval_texts = [ex.text for ex in eval_examples]

    # TRL expects "text" field.
    train_ds = Dataset.from_dict({"text": train_texts})
    eval_ds = Dataset.from_dict({"text": eval_texts}) if eval_texts else None

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Quantization / LoRA configs
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE_MAP[args.dtype])
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    peft_config = None
    if args.use_lora and not args.no_peft:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto" if device.type == "cuda" else None,
        quantization_config=quantization_config,
        torch_dtype=DTYPE_MAP[args.dtype],
    )
    model.config.use_cache = False

    version_dir = next_version_dir(
        kind="debiased",
        model_name=args.model_name,
        base_dir=args.output_dir,
        version_override=args.version,
    )
    final_output_dir = version_dir / sanitize_model_name(args.model_name)

    # TRL SFT config
    sft_config = SFTConfig(
        output_dir=str(final_output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no" if args.no_eval or not eval_texts else "steps",
        eval_steps=args.save_steps if (not args.no_eval and eval_texts) else None,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        report_to=["none"],
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds if not args.no_eval and eval_texts else None,
        args=sft_config,
        data_collator=data_collator,
        peft_config=peft_config,
    )

    print(f"[debug] train_samples={len(train_ds)} eval_samples={len(eval_ds) if eval_ds else 0}")
    trainer.train()

    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))

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
