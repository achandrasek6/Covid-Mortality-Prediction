#!/usr/bin/env python3
from Bio import SeqIO
from Bio.Align.Applications import MafftCommandline
import os

# File names (adjust paths if needed)
ref_file = "NC_045512.2_sequence.fasta"
variants_file = "variant_samples.fasta"
combined_file = "combined.fasta"
aligned_file = "aligned.fasta"

# Step 1: Combine the reference and variant samples into one FASTA file.
# Read the reference and variant records
ref_records = list(SeqIO.parse(ref_file, "fasta"))
variant_records = list(SeqIO.parse(variants_file, "fasta"))

# Combine (place reference first for clarity)
combined_records = ref_records + variant_records

# Write combined records to a new FASTA file
with open(combined_file, "w") as out_handle:
    SeqIO.write(combined_records, out_handle, "fasta")

print(f"Combined FASTA file saved as '{combined_file}'.")

# Step 2: Run MAFFT on the combined file using Biopython's MAFFT wrapper.
# This command uses the --auto option so MAFFT chooses the best algorithm automatically.
mafft_cline = MafftCommandline(input=combined_file, auto=True)
print("Running MAFFT alignment...")
stdout, stderr = mafft_cline()

# Write the aligned sequences to the output file.
with open(aligned_file, "w") as out_f:
    out_f.write(stdout)

print(f"Alignment complete. Aligned sequences saved to '{aligned_file}'.")

