#!/usr/bin/env python3
"""
extract_features.py

Extracts feature names from the header of a training feature-matrix CSV,
excluding identifier columns, and writes them as a single-row CSV
(either to stdout or to an output file).
"""

import argparse
import csv
import sys

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract feature column names from training matrix CSV header."
    )
    p.add_argument(
        "--train-feature-matrix", "-i",
        required=True,
        help="Path to the training feature-matrix CSV."
    )
    p.add_argument(
        "--output", "-o",
        help="Path to write the single-row CSV. If omitted, prints to stdout."
    )
    return p.parse_args()

def main():
    args = parse_args()

    # Read only the header row
    with open(args.train_feature_matrix, newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader)

    # Drop the non-feature columns
    exclude = {"SampleID", "Variant", "Global CFR"}
    features = [col for col in header if col not in exclude]

    # Prepare writer: to file or stdout
    out_fh = open(args.output, "w", newline='') if args.output else sys.stdout
    writer = csv.writer(out_fh)
    writer.writerow(features)

    if args.output:
        out_fh.close()

if __name__ == "__main__":
    main()
