#!/usr/bin/env python3
"""
preprocess_all.py

Full preprocessing pipeline for raw SARS-CoV-2 sample FASTA(s). Given unaligned
sample sequences and a reference genome, this script performs:

  1. Reference discovery or validation (with optional explicit reference FASTA).
  2. Concatenation of reference + samples and multiple sequence alignment via MAFFT.
  3. Reordering the alignment to place the reference first.
  4. Filtering samples based on percent identity to the reference; low-identity
     sequences are rejected and saved separately.
  5. Building a binary variant matrix from the filtered alignment (encoding
     deviations from reference at each position).
  6. Writing out:
       - Filtered & reordered alignment (FASTA)
       - Identity summary (per-sample percent identity and pass/reject)
       - Rejected sample FASTAs (for those below threshold)
       - Binary variant matrix (CSV)

Expected CLI inputs:
  --samples              : Unaligned sample FASTA file.
  --reference-fasta      : Optional path to reference FASTA (auto-detected if omitted).
  --identity-threshold   : Minimum percent identity (e.g., 92.0) for keeping a sample.
  --out-dir              : Output directory to write preprocessing results.
  --mafft-args          : Extra arguments to pass to MAFFT (optional).

Primary outputs (under out-dir):
  aligned_filtered.fasta         : Filtered alignment (reference + passing samples).
  identity_summary.tsv          : Table with percent identity and status per sample.
  rejected/                     : Directory containing FASTA(s) of filtered-out samples.
  variant_binary_matrix.csv     : Binary matrix encoding variants for downstream modeling.

Example usage:
  python3 scripts/preprocess_all.py \
    --samples raw_data/variant_samples.fasta \
    --reference-fasta raw_data/NC_045512.2_sequence.fasta \
    --identity-threshold 92.0 \
    --out-dir preprocessed
"""
import argparse
import os
import sys
import logging
import tempfile
from Bio import SeqIO
import subprocess
import pandas as pd

# -------- helpers --------
def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def find_reference(ref_path_hint=None):
    if ref_path_hint:
        if os.path.isfile(ref_path_hint):
            return ref_path_hint
        else:
            raise FileNotFoundError(f"Reference file hint provided but not found: {ref_path_hint}")
    candidates = [f for f in os.listdir("..") if f.startswith("NC_045512.2") and f.endswith((".fasta", ".fa", ".fna"))]
    if not candidates:
        raise FileNotFoundError("Could not auto-find reference FASTA (looking for files starting with NC_045512.2*.fasta/.fa/.fna).")
    logging.info(f"Auto-located reference FASTA: {candidates[0]}")
    return candidates[0]

def extract_reference_id(ref_fasta):
    records = list(SeqIO.parse(ref_fasta, "fasta"))
    if not records:
        raise RuntimeError(f"No sequences in reference FASTA {ref_fasta}")
    if len(records) > 1:
        logging.warning(f"Reference FASTA has multiple records; using first: {records[0].id}")
    return records[0].id

def run_mafft(input_fasta, output_fasta, extra_args=None):
    cmd = ["mafft", "--auto"]
    if extra_args:
        cmd += extra_args
    cmd += [input_fasta]
    logging.info("Running MAFFT: " + " ".join(cmd))
    with open(output_fasta, "w") as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        logging.error("MAFFT failed:\n" + proc.stderr)
        raise RuntimeError("MAFFT alignment failed.")
    logging.info(f"MAFFT alignment written to {output_fasta}")

def reorder_with_reference_first(aligned_fasta, reference_id, out_fasta):
    records = list(SeqIO.parse(aligned_fasta, "fasta"))
    ref = [r for r in records if r.id == reference_id]
    others = [r for r in records if r.id != reference_id]
    if not ref:
        raise KeyError(f"Reference ID '{reference_id}' not in alignment.")
    ordered = ref + others
    SeqIO.write(ordered, out_fasta, "fasta")
    logging.info(f"Reordered alignment with reference first to {out_fasta}")

def compute_percent_identity(ref_seq, sample_seq):
    if len(ref_seq) != len(sample_seq):
        raise ValueError("Sequences must be same length for identity.")
    matches = 0
    comparables = 0
    for r, s in zip(ref_seq, sample_seq):
        if r == "-" or s == "-":
            continue
        comparables += 1
        if r == s:
            matches += 1
    return (matches / comparables * 100.0) if comparables > 0 else 0.0

def filter_by_identity(aligned_fasta, reference_id, threshold, original_samples_fasta, rejected_dir, filtered_output_fasta, identity_summary_path):
    records = list(SeqIO.parse(aligned_fasta, "fasta"))
    ref_record = [r for r in records if r.id == reference_id]
    if not ref_record:
        raise KeyError(f"Reference {reference_id} missing from alignment.")
    ref_record = ref_record[0]
    others = [r for r in records if r.id != reference_id]

    passed = []
    failed = []
    identity_table = {}

    for rec in others:
        pid = compute_percent_identity(str(ref_record.seq), str(rec.seq))
        identity_table[rec.id] = pid
        if pid >= threshold:
            passed.append(rec)
        else:
            failed.append(rec)

    # write filtered alignment: reference + passed
    filtered = [ref_record] + passed
    SeqIO.write(filtered, filtered_output_fasta, "fasta")
    logging.info(f"{len(passed)} sequences passed identity filter; {len(failed)} rejected.")

    # write rejected originals (unaligned) to rejected_dir
    os.makedirs(rejected_dir, exist_ok=True)
    orig = {rec.id: rec for rec in SeqIO.parse(original_samples_fasta, "fasta")}
    for rec in failed:
        if rec.id in orig:
            outp = os.path.join(rejected_dir, f"{rec.id}.fasta")
            SeqIO.write(orig[rec.id], outp, "fasta")
        else:
            logging.warning(f"Rejected sample {rec.id} not found in original FASTA to save.")

    # write identity summary
    with open(identity_summary_path, "w") as f:
        f.write("SampleID\tPercentIdentity\tStatus\n")
        for sid, pid in sorted(identity_table.items()):
            status = "PASS" if pid >= threshold else "REJECT"
            f.write(f"{sid}\t{pid:.2f}\t{status}\n")
    logging.info(f"Identity summary written to {identity_summary_path}")

    return filtered_output_fasta

def build_binary_variant_matrix(seqs_fasta, reference_id, drop_invariant=True):
    seqs = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(seqs_fasta, "fasta")}
    ref_seq = seqs[reference_id]
    length = len(ref_seq)
    samples = sorted(seqs.keys())
    variant_dict = {}

    for pos in range(length):
        ref_base = ref_seq[pos]
        if ref_base == "-":
            continue
        for sample in samples:
            base = seqs[sample][pos]
            if base == "-" or base == ref_base:
                continue
            col = f"pos{pos+1}_{ref_base}>{base}"
            variant_dict.setdefault(col, {})[sample] = 1

    # DataFrame assembly
    cols = sorted(variant_dict.keys())
    data = []
    for sample in samples:
        row = [variant_dict[col].get(sample, 0) for col in cols]
        data.append(row)
    df = pd.DataFrame(data, index=samples, columns=cols, dtype=int)

    if drop_invariant:
        df = df.loc[:, df.nunique() > 1]

    return df

# -------- main pipeline --------
def main():
    parser = argparse.ArgumentParser(description="Full preprocessing: align, filter by identity, encode binary variants.")
    parser.add_argument("--samples", required=True, help="Unaligned sample FASTA.")
    parser.add_argument("--reference-fasta", help="Reference FASTA (auto-detect if omitted).")
    parser.add_argument("--identity-threshold", type=float, default=90.0, help="Minimum % identity to keep sample.")
    parser.add_argument("--out-dir", default="preprocessed_full", help="Root output directory.")
    parser.add_argument("--mafft-args", nargs="*", help="Extra arguments to MAFFT.")
    args = parser.parse_args()

    setup_logger()

    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)
    alignment_raw = os.path.join(out_root, "aligned_raw.fasta")
    alignment_filtered = os.path.join(out_root, "aligned_filtered.fasta")
    identity_summary = os.path.join(out_root, "identity_summary.tsv")
    variant_matrix_path = os.path.join(out_root, "variant_binary_matrix.csv")
    rejected_dir = os.path.join(out_root, "rejected")

    # find reference
    try:
        ref_path = find_reference(args.reference_fasta)
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)
    reference_id = extract_reference_id(ref_path)
    logging.info(f"Using reference: {ref_path} (ID: {reference_id})")

    # concatenate and align
    with tempfile.TemporaryDirectory() as tmp:
        combined = os.path.join(tmp, "combined.fasta")
        # merge reference + samples
        with open(combined, "w") as outf:
            for p in [ref_path, args.samples]:
                with open(p) as inf:
                    outf.write(inf.read())
        run_mafft(combined, alignment_raw, extra_args=args.mafft_args)

    # reorder so reference is first
    reorder_with_reference_first(alignment_raw, reference_id, alignment_raw + ".reordered")
    os.replace(alignment_raw + ".reordered", alignment_raw)

    # filter by identity
    filtered_fasta = filter_by_identity(
        alignment_raw,
        reference_id,
        threshold=args.identity_threshold,
        original_samples_fasta=args.samples,
        rejected_dir=rejected_dir,
        filtered_output_fasta=alignment_filtered,
        identity_summary_path=identity_summary
    )

    # encode variants
    logging.info("Building binary variant matrix from filtered alignment")
    variant_df = build_binary_variant_matrix(filtered_fasta, reference_id, drop_invariant=True)
    variant_df.to_csv(variant_matrix_path)
    logging.info(f"Variant binary matrix saved to {variant_matrix_path} (shape: {variant_df.shape})")

if __name__ == "__main__":
    main()
