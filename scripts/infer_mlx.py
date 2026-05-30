#!/usr/bin/env python3
"""Run a small Hy-MT2 MLX translation sample."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler


DEFAULT_PROMPT = (
    "Translate the following text into English. Note that you should only output "
    "the translated result without any additional explanation:\n\n"
    "今天天气真好。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local MLX checkpoint path.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--repetition-context-size", type=int, default=20)
    parser.add_argument("--ignore-chat-template", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default=None, help="Optional text output file.")
    return parser.parse_args()


def build_prompt(tokenizer, args: argparse.Namespace):
    if args.ignore_chat_template or not tokenizer.has_chat_template:
        return args.prompt

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.prompt})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def main() -> None:
    args = parse_args()
    model, tokenizer = load(args.model)
    prompt = build_prompt(tokenizer, args)
    sampler = make_sampler(
        args.temperature,
        args.top_p,
        min_p=0.0,
        min_tokens_to_keep=1,
        top_k=args.top_k,
    )
    logits_processors = make_logits_processors(
        repetition_penalty=args.repetition_penalty,
        repetition_context_size=args.repetition_context_size,
    )
    response = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=args.verbose,
    )
    print(response)
    if args.output:
        Path(args.output).write_text(response + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
