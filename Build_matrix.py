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
      5. Splits the dataset by variant into 70% training and 30% testing sets, placing any singleton‐variant samples directly into training.
      6. Drops any feature columns that are constant zeros in the training set.
      7. Writes out two CSV files:
         - feature_matrix_train.csv
         - feature_matrix_test.csv

Usage:
    python3 Build_matrix.py
"""

from Bio import AlignIO
import re
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# reproducibility
random.seed(42)

# ---------------------------
# Gene annotations (NC_045512.2 coordinates)
# ---------------------------
gene_annotations = [
    {"gene": "ORF1ab", "start": 265, "end": 21555},
    {"gene": "S",      "start": 21562, "end": 25384},
    {"gene": "ORF3a",  "start": 25392, "end": 26220},
    {"gene": "E",      "start": 26244, "end": 26472},
    {"gene": "M",      "start": 26522, "end": 27191},
    {"gene": "ORF6",   "start": 27201, "end": 27387},
    {"gene": "ORF7a",  "start": 27393, "end": 27759},
    {"gene": "ORF7b",  "start": 27755, "end": 27887},
    {"gene": "ORF8",   "start": 27893, "end": 28259},
    {"gene": "N",      "start": 28273, "end": 29533},
    {"gene": "ORF10",  "start": 29557, "end": 29674}
]

# ---------------------------
# Known global CFRs by variant
# ---------------------------
global_cfr = {
    "WildType": 0.036,
    "Alpha":     0.026,
    "Beta":      0.042,
    "Gamma":     0.036,
    "Delta":     0.020,
    "Omicron":   0.007
}

# ---------------------------
# File paths
# ---------------------------
alignment_file = "aligned.fasta"
train_csv      = "feature_matrix_train.csv"
test_csv       = "feature_matrix_test.csv"

# ---------------------------
# Load alignment & identify reference
# ---------------------------
alignment = AlignIO.read(alignment_file, "fasta")
ref_record = next((r for r in alignment if "NC_045512.2" in r.id), None)
if ref_record is None:
    raise RuntimeError("Reference 'NC_045512.2' not found in alignment.")

# ---------------------------
# Map alignment columns to raw positions
# ---------------------------
mapping = {}
raw_pos = 0
for i, nt in enumerate(ref_record.seq):
    if nt != "-":
        raw_pos += 1
        mapping[i] = raw_pos
    else:
        mapping[i] = None

# ---------------------------
# Determine which columns belong to each gene
# ---------------------------
gene_alignment_indices = {}
for ga in gene_annotations:
    name = ga["gene"]
    idxs = [
        i
        for i, pos in mapping.items()
        if pos is not None and ga["start"] <= pos <= ga["end"]
    ]
    gene_alignment_indices[name] = idxs

# ---------------------------
# Build feature column labels
# ---------------------------
column_labels = []
for ga in gene_annotations:
    name = ga["gene"]
    for j in range(1, len(gene_alignment_indices[name]) + 1):
        column_labels.append(f"{name}_{j}")

# ---------------------------
# Helper to parse variant from header
# ---------------------------
def extract_variant(header: str) -> str:
    m = re.search(r"\[([^\]]+)\]", header)
    return m.group(1) if m else "unknown"

# ---------------------------
# Construct rows: [SampleID, Variant, CFR, feat1, feat2, …]
# ---------------------------
all_rows = []
for rec in alignment:
    if "NC_045512.2" in rec.id:
        continue
    sid = rec.id
    var = extract_variant(rec.description)
    cfr = global_cfr.get(var, "N/A")
    feats = []
    for ga in gene_annotations:
        for idx in gene_alignment_indices[ga["gene"]]:
            ref_nt    = ref_record.seq[idx].upper()
            sample_nt = rec.seq[idx].upper()
            feats.append(int(sample_nt != ref_nt))
    all_rows.append([sid, var, cfr] + feats)

# ---------------------------
# Create DataFrame
# ---------------------------
df = pd.DataFrame(
    all_rows,
    columns=["SampleID", "Variant", "Global CFR"] + column_labels
)

# ---------------------------
# Handle singleton variants & stratified split
# ---------------------------
# Identify variants with <2 samples
counts = df["Variant"].value_counts()
singletons = counts[counts < 2].index.tolist()

# Pull singletons entirely into training
df_single = df[df["Variant"].isin(singletons)]
df_main   = df[~df["Variant"].isin(singletons)]

# Stratified 70/30 on the remaining variants
df_train_main, df_test = train_test_split(
    df_main,
    test_size=0.30,
    random_state=42,
    stratify=df_main["Variant"]
)

# Combine singleton rows into training
df_train = pd.concat([df_train_main, df_single], axis=0).reset_index(drop=True)

# ---------------------------
# Drop zero‐only features (based on training set)
# ---------------------------
keep_feats = [col for col in column_labels if df_train[col].any()]
final_cols = ["SampleID", "Variant", "Global CFR"] + keep_feats

df_train = df_train[final_cols]
df_test  = df_test[final_cols]

# ---------------------------
# Save to CSV
# ---------------------------
df_train.to_csv(train_csv, index=False)
print(f"Training feature matrix saved to '{train_csv}'.")

df_test.to_csv(test_csv, index=False)
print(f"Testing feature matrix saved to '{test_csv}'.")
