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
    we check multiple locations in this order:
    1. Current working directory (for Nextflow staging)
    2. Absolute path (if provided)
    3. Relative to PROJECT_DIR
    Otherwise we search PROJECT_DIR/raw_data for NC_045512.2*.fasta.
    """
    if ref_path_hint:
        # First, check if it exists in current working directory (Nextflow staging)
        if os.path.isfile(ref_path_hint):
            abs_path = os.path.abspath(ref_path_hint)
            logging.info(f"Using provided reference FASTA (current dir): {abs_path}")
            return abs_path

        # Second, check if it's an absolute path
        if os.path.isabs(ref_path_hint) and os.path.isfile(ref_path_hint):
            logging.info(f"Using provided reference FASTA (absolute): {ref_path_hint}")
            return ref_path_hint

        # Third, try relative to project root
        project_relative = os.path.join(PROJECT_DIR, ref_path_hint)
        if os.path.isfile(project_relative):
            logging.info(f"Using provided reference FASTA (project relative): {project_relative}")
            return project_relative

        # If none of the above work, raise error with detailed information
        current_dir_path = os.path.abspath(ref_path_hint)
        raise FileNotFoundError(
            f"Reference file not found. Tried:\n"
            f"  - Current directory: {current_dir_path} (exists: {os.path.isfile(current_dir_path)})\n"
            f"  - Absolute path: {ref_path_hint} (exists: {os.path.isfile(ref_path_hint) if os.path.isabs(ref_path_hint) else 'N/A - not absolute'})\n"
            f"  - Project relative: {project_relative} (exists: {os.path.isfile(project_relative)})\n"
            f"  - Current working directory: {os.getcwd()}\n"
            f"  - Files in current directory: {os.listdir('.')}"
        )

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
    """Extract the reference sequence ID from the reference FASTA file."""
    records = list(SeqIO.parse(ref_fasta, "fasta"))
    if not records:
        raise RuntimeError(f"No sequences in reference FASTA {ref_fasta}")
    if len(records) > 1:
        logging.warning(f"Reference FASTA has multiple records; using first: {records[0].id}")
    return records[0].id


def run_mafft(input_fasta, output_fasta, extra_args=None):
    """Run MAFFT alignment on the input FASTA file."""
    cmd = ["mafft", "--auto"] + (extra_args or []) + [input_fasta]
    logging.info("Running MAFFT: " + " ".join(cmd))

    try:
        with open(output_fasta, "w") as out:
            proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)

        if proc.returncode != 0:
            logging.error("MAFFT failed:\n" + proc.stderr)
            raise RuntimeError("MAFFT alignment failed.")

        logging.info(f"MAFFT alignment written to {output_fasta}")

    except FileNotFoundError:
        raise RuntimeError(
            "MAFFT not found. Please ensure MAFFT is installed and in your PATH.\n"
            "You can install it with: conda install -c bioconda mafft"
        )


def reorder_with_reference_first(aligned_fasta, reference_id, out_fasta):
    """Reorder alignment to place reference sequence first."""
    records = list(SeqIO.parse(aligned_fasta, "fasta"))
    ref = [r for r in records if r.id == reference_id]
    others = [r for r in records if r.id != reference_id]

    if not ref:
        available_ids = [r.id for r in records]
        raise KeyError(
            f"Reference ID '{reference_id}' not found in alignment.\n"
            f"Available sequence IDs: {available_ids}"
        )

    SeqIO.write(ref + others, out_fasta, "fasta")
    logging.info(f"Reordered alignment with reference first to {out_fasta}")


def compute_percent_identity(ref_seq, sample_seq):
    """Compute percent identity between reference and sample sequences."""
    if len(ref_seq) != len(sample_seq):
        raise ValueError("Sequences must be same length for identity calculation.")

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
    """Filter sequences by percent identity to reference."""
    records = list(SeqIO.parse(aligned_fasta, "fasta"))
    ref_record = next((r for r in records if r.id == reference_id), None)

    if not ref_record:
        available_ids = [r.id for r in records]
        raise KeyError(
            f"Reference {reference_id} missing from alignment.\n"
            f"Available sequence IDs: {available_ids}"
        )

    others = [r for r in records if r.id != reference_id]

    passed, failed, identity_table = [], [], {}
    for rec in others:
        pid = compute_percent_identity(str(ref_record.seq), str(rec.seq))
        identity_table[rec.id] = pid
        (passed if pid >= threshold else failed).append(rec)

    # Write filtered alignment
    SeqIO.write([ref_record] + passed, filtered_output_fasta, "fasta")
    logging.info(f"Identity filtering: {len(passed)} passed, {len(failed)} rejected (threshold: {threshold}%)")

    # Save rejected sequences in original (unaligned) form
    os.makedirs(rejected_dir, exist_ok=True)
    if os.path.isfile(original_samples_fasta):
        orig = {r.id: r for r in SeqIO.parse(original_samples_fasta, "fasta")}
        for rec in failed:
            if rec.id in orig:
                rejected_file = os.path.join(rejected_dir, f"{rec.id}.fasta")
                SeqIO.write(orig[rec.id], rejected_file, "fasta")
            else:
                logging.warning(f"Could not find original record for {rec.id} to save as rejected.")
    else:
        logging.warning(f"Original samples file {original_samples_fasta} not found. Cannot save rejected sequences.")

    # Write identity summary
    with open(identity_summary_path, "w") as f:
        f.write("SampleID\tPercentIdentity\tStatus\n")
        for sid, pid in sorted(identity_table.items()):
            status = "PASS" if pid >= threshold else "REJECT"
            f.write(f"{sid}\t{pid:.2f}\t{status}\n")

    logging.info(f"Identity summary written to {identity_summary_path}")
    return filtered_output_fasta


def build_binary_variant_matrix(seqs_fasta, reference_id, drop_invariant=True):
    """Build binary variant matrix from aligned sequences."""
    seqs = {r.id: str(r.seq).upper() for r in SeqIO.parse(seqs_fasta, "fasta")}

    if reference_id not in seqs:
        available_ids = list(seqs.keys())
        raise KeyError(
            f"Reference ID '{reference_id}' not found in sequences.\n"
            f"Available sequence IDs: {available_ids}"
        )

    ref_seq = seqs[reference_id]
    length = len(ref_seq)
    samples = sorted(seqs.keys())
    variant_dict = {}

    # Build variant dictionary
    for pos in range(length):
        ref_base = ref_seq[pos]
        if ref_base == "-":
            continue  # Skip gap positions in reference

        for sample in samples:
            if sample == reference_id:
                continue  # Skip reference itself

            base = seqs[sample][pos]
            if base == "-" or base == ref_base:
                continue  # Skip gaps and matches

            col = f"pos{pos + 1}_{ref_base}>{base}"
            variant_dict.setdefault(col, {})[sample] = 1

    # Convert to DataFrame
    if not variant_dict:
        logging.warning("No variants found. Creating empty variant matrix.")
        df = pd.DataFrame(index=samples, dtype=int)
    else:
        df = pd.DataFrame(
            [[variant_dict[col].get(s, 0) for col in sorted(variant_dict)]
             for s in samples],
            index=samples, columns=sorted(variant_dict), dtype=int
        )

    # Optionally drop invariant columns
    if drop_invariant and not df.empty:
        original_cols = df.shape[1]
        df = df.loc[:, df.nunique() > 1]
        dropped_cols = original_cols - df.shape[1]
        if dropped_cols > 0:
            logging.info(f"Dropped {dropped_cols} invariant columns")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Full preprocessing: align, filter, and encode variants."
    )
    parser.add_argument("--samples", required=True,
                        help="Path to FASTA file containing sample sequences")
    parser.add_argument("--reference-fasta",
                        help="Path to reference FASTA file (absolute, relative to current dir, or relative to project root)")
    parser.add_argument("--identity-threshold", type=float, default=90.0,
                        help="Minimum percent identity to reference for inclusion (default: 90.0)")
    parser.add_argument("--out-dir", default="preprocessed_full",
                        help="Output directory (default: preprocessed_full)")
    parser.add_argument("--mafft-args", nargs="*",
                        help="Additional arguments to pass to MAFFT")
    args = parser.parse_args()

    setup_logger()

    # Log input parameters
    logging.info("=== PREPROCESSING PARAMETERS ===")
    logging.info(f"Samples file: {args.samples}")
    logging.info(f"Reference file hint: {args.reference_fasta}")
    logging.info(f"Identity threshold: {args.identity_threshold}%")
    logging.info(f"Output directory: {args.out_dir}")
    logging.info(f"Current working directory: {os.getcwd()}")
    logging.info(f"Project directory: {PROJECT_DIR}")
    logging.info("=== END PARAMETERS ===")

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    # 1) Resolve reference
    try:
        ref_path = find_reference(args.reference_fasta)
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    try:
        ref_id = extract_reference_id(ref_path)
        logging.info(f"Reference resolved: {ref_path} (ID: {ref_id})")
    except Exception as e:
        logging.error(f"Failed to extract reference ID: {e}")
        sys.exit(1)

    # 2) Align
    raw_align = os.path.join(out, "aligned_raw.fasta")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            combo = os.path.join(tmp, "combo.fasta")

            # Combine reference and samples
            with open(combo, "w") as w:
                # Write reference first
                with open(ref_path, "r") as ref_file:
                    w.write(ref_file.read())
                # Write samples
                with open(args.samples, "r") as samples_file:
                    w.write(samples_file.read())

            logging.info(f"Combined reference and samples into {combo}")
            run_mafft(combo, raw_align, extra_args=args.mafft_args)

    except Exception as e:
        logging.error(f"Alignment failed: {e}")
        sys.exit(1)

    # 3) Reorder alignment to put reference first
    try:
        reordered = raw_align + ".reordered"
        reorder_with_reference_first(raw_align, ref_id, reordered)
        os.replace(reordered, raw_align)
    except Exception as e:
        logging.error(f"Reordering failed: {e}")
        sys.exit(1)

    # 4) Filter by identity
    try:
        filt = os.path.join(out, "aligned_filtered.fasta")
        summary = os.path.join(out, "identity_summary.tsv")
        rej_dir = os.path.join(out, "rejected")

        filtered = filter_by_identity(
            raw_align, ref_id, args.identity_threshold,
            args.samples, rej_dir, filt, summary
        )
    except Exception as e:
        logging.error(f"Identity filtering failed: {e}")
        sys.exit(1)

    # 5) Build binary variant matrix
    try:
        bin_mat = build_binary_variant_matrix(filtered, ref_id)
        csv_out = os.path.join(out, "variant_binary_matrix.csv")
        bin_mat.to_csv(csv_out)
        logging.info(f"Variant matrix saved to {csv_out} (shape: {bin_mat.shape})")
    except Exception as e:
        logging.error(f"Variant matrix creation failed: {e}")
        sys.exit(1)

    logging.info("=== PREPROCESSING COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()