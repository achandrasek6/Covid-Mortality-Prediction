#!/usr/bin/env python3
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
