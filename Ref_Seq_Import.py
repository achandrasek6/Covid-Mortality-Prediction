#!/usr/bin/env python3
"""
Script: Ref_Seq_Import.py

Description:
    This script uses Biopython’s Entrez utilities to download the SARS-CoV-2 reference genome
    (accession NC_045512.2) from NCBI. It performs two main tasks:
      1. Parses the GenBank record to extract gene annotations (gene name, start, end, strand)
         and writes them to a tab-delimited text file.
      2. Writes the full nucleotide sequence for NC_045512.2 to a FASTA file.

Usage:
    Ensure Biopython is installed (`pip install biopython`), then run:
        python Ref_Seq_Import.py

Outputs:
    - NC_045512.2_gene_annotations.txt : Tab-delimited list of gene names with coordinates and strand
    - NC_045512.2_sequence.fasta      : FASTA file of the complete reference genome sequence

Dependencies:
    - Python 3
    - Biopython (Bio.Entrez, Bio.SeqIO)
"""

from Bio import Entrez, SeqIO

# Set your email address (required by NCBI)
Entrez.email = "aravind_plano@yahoo.com"

# Define the reference accession
ref_acc = "NC_045512.2"

# ---------------------------
# 1. Fetch the GenBank Record for the Reference Sequence
# ---------------------------
with Entrez.efetch(db="nuccore", id=ref_acc, rettype="gb", retmode="text") as handle:
    record = SeqIO.read(handle, "genbank")

# ---------------------------
# 2. Write Gene Annotations to a File
# ---------------------------
annotations_file = "NC_045512.2_gene_annotations.txt"
with open(annotations_file, "w") as f:
    f.write("Gene\tStart\tEnd\tStrand\n")
    for feature in record.features:
        if feature.type == "gene":
            # Get gene name; if not available, use "unknown"
            gene_name = feature.qualifiers.get("gene", ["unknown"])[0]
            start = int(feature.location.start)
            end = int(feature.location.end)
            strand = feature.location.strand  # +1 for forward, -1 for reverse
            f.write(f"{gene_name}\t{start}\t{end}\t{strand}\n")
print(f"Gene annotations saved to {annotations_file}")

# ---------------------------
# 3. Write the Nucleotide Sequence to a FASTA File
# ---------------------------
fasta_file = "NC_045512.2_sequence.fasta"
with open(fasta_file, "w") as f:
    SeqIO.write(record, f, "fasta")
print(f"Nucleotide sequence saved to {fasta_file}")


