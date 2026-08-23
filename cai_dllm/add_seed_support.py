#!/usr/bin/env python3
"""
add_seed_support.py — Adds --seed argument to eval_commonsense.py and eval_math_code.py
Run once on the server: python cai_dllm/add_seed_support.py
"""
import os, sys

def add_seed_to_file(filepath, argparse_anchor, set_seed_code):
    with open(filepath, 'r') as f:
        content = f.read()

    if '--seed' in content:
        print(f"  {filepath}: seed already added")
        return

    # Add argparse argument
    old = argparse_anchor
    new = ('    p.add_argument("--seed", type=int, default=42,\n'
           '                   help="Random seed for reproducibility")\n'
           + argparse_anchor)
    if old not in content:
        print(f"  {filepath}: anchor not found — skipping")
        return
    content = content.replace(old, new, 1)

    # Add seed setting at start of main()
    old2 = 'def main():\n    args = parse_args()'
    new2 = ('def main():\n'
            '    args = parse_args()\n'
            '    # Set random seed for reproducibility\n'
            '    import random, numpy as np, torch\n'
            '    random.seed(args.seed)\n'
            '    np.random.seed(args.seed)\n'
            '    torch.manual_seed(args.seed)\n'
            '    if torch.cuda.is_available():\n'
            '        torch.cuda.manual_seed_all(args.seed)\n'
            '    print(f"Seed: {args.seed}", flush=True)')
    if old2 in content:
        content = content.replace(old2, new2, 1)

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  {filepath}: seed support added")

# eval_commonsense.py
add_seed_to_file(
    "cai_dllm/eval_commonsense.py",
    '    p.add_argument("--output_dir"',
    '    p.add_argument("--output_dir"'
)

# eval_math_code.py
add_seed_to_file(
    "cai_dllm/eval_math_code.py",
    '    p.add_argument("--output_dir"',
    '    p.add_argument("--output_dir"'
)

print("Done.")
