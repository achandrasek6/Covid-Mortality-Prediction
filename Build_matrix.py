#!/usr/bin/env python3
"""
Script: Build_matrix.py

Description:
    This script processes a multiple‐sequence alignment of SARS-CoV-2 genomes (including the NC_045512.2 reference)
    to generate binary feature matrices for machine learning. It:
      1. Defines gene coordinate annotations based on the reference genome.
      2. Maps alignment columns to raw reference positions and groups those by gene.
      3. Extracts, for each sample, a binary vector indicating nucleotide differences from the reference at each gene position.
      4. Attaches metadata: SampleID, Variant (parsed from sequence header), and known global CFR per variant.
      5. Splits the dataset by variant into 70% training and 30% testing sets.
      6. Drops any feature columns that are constant zeros in the training set.
      7. Writes out two CSV files:
         - feature_matrix_train.csv
         - feature_matrix_test.csv

Usage:
    python Build_matrix.py

Outputs:
    - feature_matrix_train.csv : Training set with non-constant binary features
    - feature_matrix_test.csv  : Testing set with the same selected features

Dependencies:
    - Python 3
    - Biopython (`pip install biopython`) for AlignIO
    - Standard library modules: csv, re, random, numpy, pandas (if used)
"""
from Bio import AlignIO
import csv
import re
import random

# ---------------------------
# Define gene annotations (raw coordinates based on NC_045512.2)
# ---------------------------
gene_annotations = [
    {"gene": "ORF1ab", "start": 265, "end": 21555, "strand": 1},
    {"gene": "S", "start": 21562, "end": 25384, "strand": 1},
    {"gene": "ORF3a", "start": 25392, "end": 26220, "strand": 1},
    {"gene": "E", "start": 26244, "end": 26472, "strand": 1},
    {"gene": "M", "start": 26522, "end": 27191, "strand": 1},
    {"gene": "ORF6", "start": 27201, "end": 27387, "strand": 1},
    {"gene": "ORF7a", "start": 27393, "end": 27759, "strand": 1},
    {"gene": "ORF7b", "start": 27755, "end": 27887, "strand": 1},
    {"gene": "ORF8", "start": 27893, "end": 28259, "strand": 1},
    {"gene": "N", "start": 28273, "end": 29533, "strand": 1},
    {"gene": "ORF10", "start": 29557, "end": 29674, "strand": 1}
]

# ---------------------------
# Global CFR values for each variant as decimals.
# ---------------------------
global_cfr = {
    "WildType": 0.036,  # 3.6% -> 0.036
    "Alpha": 0.026,     # 2.6% -> 0.026
    "Beta": 0.042,      # 4.2% -> 0.042
    "Gamma": 0.036,     # 3.6% -> 0.036
    "Delta": 0.020,     # 2.0% -> 0.020
    "Omicron": 0.007    # 0.7% -> 0.007
}

# ---------------------------
# File names
# ---------------------------
alignment_file = "aligned.fasta"  # Combined alignment (reference + samples)
train_csv = "feature_matrix_train.csv"
test_csv = "feature_matrix_test.csv"

# ---------------------------
# Read the alignment file
# ---------------------------
alignment = AlignIO.read(alignment_file, "fasta")

# Identify the reference sequence (assumes its header contains "NC_045512.2")
ref_record = None
for record in alignment:
    if "NC_045512.2" in record.id:
        ref_record = record
        break
if ref_record is None:
    raise Exception("Reference sequence 'NC_045512.2' not found in alignment.")

# ---------------------------
# Build mapping: alignment column index -> reference raw coordinate.
# ---------------------------
mapping = {}  # mapping[i] = reference raw position (or None if gap)
raw_pos = 0
for i, nt in enumerate(ref_record.seq):
    if nt != "-":
        raw_pos += 1
        mapping[i] = raw_pos
    else:
        mapping[i] = None

# ---------------------------
# Determine which alignment columns correspond to each annotated gene.
# ---------------------------
gene_alignment_indices = {}  # gene -> list of alignment indices
for gene in gene_annotations:
    gene_name = gene["gene"]
    indices = []
    for i, pos in mapping.items():
        if pos is not None and gene["start"] <= pos <= gene["end"]:
            indices.append(i)
    gene_alignment_indices[gene_name] = indices

# Build column labels for the feature matrix.
# These labels correspond to each nucleotide position in the annotated regions.
column_labels = []
for gene in gene_annotations:
    gene_name = gene["gene"]
    for j in range(1, len(gene_alignment_indices[gene_name]) + 1):
        column_labels.append(f"{gene_name}_{j}")

# ---------------------------
# Function to extract variant name from header (assumes variant is in square brackets)
# ---------------------------
def extract_variant(header):
    match = re.search(r'\[([^\]]+)\]', header)
    if match:
        return match.group(1).strip()
    else:
        return "unknown"

# ---------------------------
# Build the full feature matrix rows for each sample.
# Each row: [SampleID, Variant, Global CFR, feature vector...]
# ---------------------------
all_rows = []
for record in alignment:
    # Skip the reference sequence.
    if "NC_045512.2" in record.id:
        continue

    sample_id = record.id
    variant = extract_variant(record.description)
    cfr = global_cfr.get(variant, "N/A")
    features = []
    for gene in gene_annotations:
        gene_name = gene["gene"]
        for idx in gene_alignment_indices[gene_name]:
            ref_nt = ref_record.seq[idx].upper()
            sample_nt = record.seq[idx].upper()
            features.append(0 if sample_nt == ref_nt else 1)
    all_rows.append([sample_id, variant, cfr] + features)

# ---------------------------
# Group the rows by variant.
# ---------------------------
variant_groups = {}
for row in all_rows:
    variant = row[1]
    if variant not in variant_groups:
        variant_groups[variant] = []
    variant_groups[variant].append(row)

# ---------------------------
# For each variant group, split into 70% training and 30% testing.
# ---------------------------
train_rows = []
test_rows = []
for variant, rows in variant_groups.items():
    random.shuffle(rows)
    split_point = int(0.7 * len(rows))
    train_rows.extend(rows[:split_point])
    test_rows.extend(rows[split_point:])

# ---------------------------
# Drop any feature columns (i.e. columns after the first 3) that are all zeros in the training set.
# ---------------------------
# header: ["SampleID", "Variant", "Global CFR"] + column_labels
feature_start = 3  # feature columns start at index 3
num_features = len(column_labels)
# Determine which feature columns are non-constant (i.e. not all zeros) in training data.
keep_feature_indices = []
for j in range(num_features):
    col_idx = feature_start + j
    # Check if at least one training row has a non-zero value for this feature.
    if any(row[col_idx] != 0 for row in train_rows):
        keep_feature_indices.append(col_idx)

# Update header: keep first 3 columns and the kept feature columns.
new_header = ["SampleID", "Variant", "Global CFR"] + [f for i, f in enumerate(column_labels) if feature_start + i in keep_feature_indices]

def filter_features(rows, keep_indices, feature_start=3):
    new_rows = []
    for row in rows:
        new_row = row[:feature_start]  # first three columns
        new_row += [row[i] for i in keep_indices]
        new_rows.append(new_row)
    return new_rows

train_rows = filter_features(train_rows, keep_feature_indices, feature_start)
test_rows = filter_features(test_rows, keep_feature_indices, feature_start)

# ---------------------------
# Write the training and testing matrices to CSV files.
# ---------------------------
with open(train_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(new_header)
    for row in train_rows:
        writer.writerow(row)
print(f"Training feature matrix saved to '{train_csv}'.")

with open(test_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(new_header)
    for row in test_rows:
        writer.writerow(row)
print(f"Testing feature matrix saved to '{test_csv}'.")
