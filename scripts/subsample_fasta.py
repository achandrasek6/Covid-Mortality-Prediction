#!/usr/bin/env python3
"""
subsample_fasta.py

Utility to create a reproducible random subset of sequences from a large FASTA file
using reservoir sampling (constant memory, one-pass). Intended for fast iteration / testing
without needing to process the full dataset.

Main responsibilities:
  1. Read through the input FASTA.
  2. Select `k` sequences uniformly at random via reservoir sampling.
  3. Write the subset to an output FASTA.

Features:
  * Fixed default seed (42) for deterministic subsamples unless overridden.
  * Works on arbitrarily large FASTA without loading all sequences into memory.

CLI arguments:
  -i, --input    : Path to the input FASTA (full dataset).
  -o, --output   : Path to write the subsampled FASTA.
  -k            : Number of sequences to sample.
  --seed        : (Optional) Random seed for reproducibility; default is 42.

Outputs:
  Subsampled FASTA containing `k` sequences.

Example usage:
  python3 scripts/subsample_fasta.py \
    -i raw_data/variant_samples.fasta \
    -o transformed_data/variant_samples_small.fasta \
    -k 100

  # With explicit seed (different subset)
  python3 scripts/subsample_fasta.py \
    -i raw_data/variant_samples.fasta \
    -o transformed_data/variant_samples_small.fasta \
    -k 100 --seed 123
"""

import argparse
import random
from Bio import SeqIO

def reservoir_sample_fasta(input_fasta, output_fasta, k, seed=42):
    random.seed(seed)

    reservoir = []
    for i, record in enumerate(SeqIO.parse(input_fasta, "fasta")):
        if i < k:
            reservoir.append(record)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = record

    if len(reservoir) < k:
        print(f"Warning: input has only {len(reservoir)} records (requested {k}).")
    SeqIO.write(reservoir, output_fasta, "fasta")
    print(f"Wrote {len(reservoir)} sequences to {output_fasta} (seed={seed})")

def main():
    parser = argparse.ArgumentParser(description="Subsample k sequences from a FASTA via reservoir sampling.")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file.")
    parser.add_argument("-o", "--output", required=True, help="Output FASTA with subsample.")
    parser.add_argument("-k", type=int, required=True, help="Number of sequences to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default=42).")
    args = parser.parse_args()
    reservoir_sample_fasta(args.input, args.output, args.k, args.seed)

if __name__ == "__main__":
    main()
