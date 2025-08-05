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
"""

import argparse
import os
import sys
import logging
import tempfile
from Bio import SeqIO
import subprocess
import pandas as pd
import glob

# Determine project root (one level up from scripts/)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))


def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def find_reference(ref_path_hint=None):
    """
    Resolve the reference FASTA. If the user provided --reference-fasta,
    we treat it as either an absolute path or relative to PROJECT_DIR.
    Otherwise we search PROJECT_DIR/raw_data for NC_045512.2*.fasta.
    """
    if ref_path_hint:
        # if absolute, use it; otherwise resolve under project root
        ref = ref_path_hint if os.path.isabs(ref_path_hint) \
              else os.path.join(PROJECT_DIR, ref_path_hint)
        if os.path.isfile(ref):
            logging.info(f"Using provided reference FASTA: {ref}")
            return ref
        else:
            raise FileNotFoundError(f"Reference file hint provided but not found: {ref}")
    # no hint: search in raw_data/
    search_pattern = os.path.join(PROJECT_DIR, "raw_data", "NC_045512.2*.fasta")
    candidates = glob.glob(search_pattern)
    if not candidates:
        raise FileNotFoundError(
            f"Could not auto-find reference FASTA in {PROJECT_DIR}/raw_data "
            f"(looking for NC_045512.2*.fasta)"
        )
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
    cmd = ["mafft", "--auto"] + (extra_args or []) + [input_fasta]
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
    SeqIO.write(ref + others, out_fasta, "fasta")
    logging.info(f"Reordered alignment with reference first to {out_fasta}")


def compute_percent_identity(ref_seq, sample_seq):
    if len(ref_seq) != len(sample_seq):
        raise ValueError("Sequences must be same length for identity.")
    matches = comparables = 0
    for r, s in zip(ref_seq, sample_seq):
        if r == "-" or s == "-":
            continue
        comparables += 1
        if r == s:
            matches += 1
    return (matches / comparables * 100.0) if comparables else 0.0


def filter_by_identity(aligned_fasta, reference_id, threshold,
                       original_samples_fasta, rejected_dir,
                       filtered_output_fasta, identity_summary_path):
    records = list(SeqIO.parse(aligned_fasta, "fasta"))
    ref_record = next((r for r in records if r.id == reference_id), None)
    if not ref_record:
        raise KeyError(f"Reference {reference_id} missing from alignment.")
    others = [r for r in records if r.id != reference_id]

    passed, failed, identity_table = [], [], {}
    for rec in others:
        pid = compute_percent_identity(str(ref_record.seq), str(rec.seq))
        identity_table[rec.id] = pid
        (passed if pid >= threshold else failed).append(rec)

    SeqIO.write([ref_record] + passed, filtered_output_fasta, "fasta")
    logging.info(f"{len(passed)} passed; {len(failed)} rejected.")

    os.makedirs(rejected_dir, exist_ok=True)
    orig = {r.id: r for r in SeqIO.parse(original_samples_fasta, "fasta")}
    for rec in failed:
        if rec.id in orig:
            SeqIO.write(orig[rec.id],
                        os.path.join(rejected_dir, f"{rec.id}.fasta"), "fasta")
        else:
            logging.warning(f"Could not find original record for {rec.id} to reject.")

    with open(identity_summary_path, "w") as f:
        f.write("SampleID\tPercentIdentity\tStatus\n")
        for sid, pid in sorted(identity_table.items()):
            status = "PASS" if pid >= threshold else "REJECT"
            f.write(f"{sid}\t{pid:.2f}\t{status}\n")
    logging.info(f"Identity summary -> {identity_summary_path}")

    return filtered_output_fasta


def build_binary_variant_matrix(seqs_fasta, reference_id, drop_invariant=True):
    seqs = {r.id: str(r.seq).upper() for r in SeqIO.parse(seqs_fasta, "fasta")}
    ref_seq = seqs[reference_id]
    length = len(ref_seq)
    samples = sorted(seqs.keys())
    variant_dict = {}

    for pos in range(length):
        ref_base = ref_seq[pos]
        if ref_base == "-": continue
        for sample in samples:
            base = seqs[sample][pos]
            if base == "-" or base == ref_base:
                continue
            col = f"pos{pos+1}_{ref_base}>{base}"
            variant_dict.setdefault(col, {})[sample] = 1

    df = pd.DataFrame(
        [[variant_dict[col].get(s, 0) for col in sorted(variant_dict)]
         for s in samples],
        index=samples, columns=sorted(variant_dict), dtype=int
    )
    if drop_invariant:
        df = df.loc[:, df.nunique() > 1]
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Full preprocessing: align, filter, and encode variants."
    )
    parser.add_argument("--samples",            required=True)
    parser.add_argument("--reference-fasta",    help="Path or relative path to reference FASTA")
    parser.add_argument("--identity-threshold", type=float, default=90.0)
    parser.add_argument("--out-dir",            default="preprocessed_full")
    parser.add_argument("--mafft-args", nargs="*")
    args = parser.parse_args()

    setup_logger()
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    # 1) Resolve reference
    try:
        ref_path = find_reference(args.reference_fasta)
    except FileNotFoundError as e:
        logging.error(str(e)); sys.exit(1)
    ref_id = extract_reference_id(ref_path)
    logging.info(f"Reference -> {ref_path} (ID: {ref_id})")

    # 2) Align
    raw_align = os.path.join(out, "aligned_raw.fasta")
    with tempfile.TemporaryDirectory() as tmp:
        combo = os.path.join(tmp, "combo.fasta")
        with open(combo, "w") as w:
            for p in (ref_path, args.samples):
                w.write(open(p).read())
        run_mafft(combo, raw_align, extra_args=args.mafft_args)

    # 3) Reorder
    reordered = raw_align + ".reordered"
    reorder_with_reference_first(raw_align, ref_id, reordered)
    os.replace(reordered, raw_align)

    # 4) Filter
    filt = os.path.join(out, "aligned_filtered.fasta")
    summary = os.path.join(out, "identity_summary.tsv")
    rej_dir = os.path.join(out, "rejected")
    filtered = filter_by_identity(
        raw_align, ref_id, args.identity_threshold,
        args.samples, rej_dir, filt, summary
    )

    # 5) Encode variants
    bin_mat = build_binary_variant_matrix(filtered, ref_id)
    csv_out = os.path.join(out, "variant_binary_matrix.csv")
    bin_mat.to_csv(csv_out)
    logging.info(f"Variant matrix saved to {csv_out} (shape={bin_mat.shape})")


if __name__ == "__main__":
    main()
