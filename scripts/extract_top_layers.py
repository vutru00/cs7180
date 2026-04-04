"""Extract the top causal layers from Phase 1 NIE results.

Usage:
    python scripts/extract_top_layers.py \
        --inputs results/nie_gender_unet.json results/nie_gender_textenc.json \
        --top-k 10 \
        --output results/top_causal_layers.json

The script averages NIE values across all prompts, ranks layers by mean NIE,
and saves the top-k layer names as a JSON list.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="NIE JSON files from run_tracing.py")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top layers to keep")
    parser.add_argument("--output", type=str, default="results/top_causal_layers.json")
    parser.add_argument("--print-all", action="store_true",
                        help="Print the full ranked list before saving")
    return parser.parse_args()


def main():
    args = parse_args()

    # Accumulate NIE values per layer across all prompts and input files
    layer_values = defaultdict(list)

    for path in args.inputs:
        with open(path) as f:
            data = json.load(f)
        # data: {prompt_text: {layer_name: nie_value}}
        for prompt_text, layer_nies in data.items():
            for layer, nie in layer_nies.items():
                layer_values[layer].append(nie)

    # Compute mean NIE per layer
    mean_nies = {layer: sum(vals) / len(vals) for layer, vals in layer_values.items()}

    # Sort descending by absolute mean NIE (captures both male- and female-mediating layers)
    ranked = sorted(mean_nies.items(), key=lambda x: abs(x[1]), reverse=True)

    if args.print_all:
        print(f"\nAll layers ranked by mean NIE ({len(ranked)} total):")
        for i, (layer, nie) in enumerate(ranked):
            print(f"  {i+1:3d}. {nie:+.4f}  {layer}")

    top_layers = [layer for layer, _ in ranked[: args.top_k]]

    print(f"\nTop {args.top_k} causal layers:")
    for i, layer in enumerate(top_layers):
        print(f"  {i+1:2d}. {mean_nies[layer]:+.4f}  {layer}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(top_layers, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
