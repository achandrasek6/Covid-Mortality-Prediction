#!/usr/bin/env python3
"""
Script: NCBI_variant_sampling.py

Description:
    This script queries NCBI’s nuccore database for complete SARS-CoV-2 genomes
    of specified pangolin lineages (Alpha, Beta, Gamma, Delta, Omicron, and the
    Wuhan-Hu-1 reference “WildType”), retrieves all matching FASTA records, and
    annotates each sequence header with its variant name. All fetched records
    are combined and written to a single output FASTA file.

Features:
    1. Uses Entrez.esearch to count and list all record IDs for each variant.
    2. Uses Entrez.efetch to download full genome FASTA sequences in batch.
    3. Appends “[VariantName]” to each sequence description for downstream
       identification.
    4. Respects NCBI usage guidelines by including an email address and
       rate-limiting requests with a short pause between variant fetches.

Usage:
    Ensure you have Biopython installed (`pip install biopython`). Then run:
        python fetch_sars_cov2_variants.py

    The script will create a file named `variant_samples.fasta` in the
    current directory containing all annotated sequences.

Dependencies:
    - Python 3
    - Biopython (Bio.Entrez, Bio.SeqIO)
"""

from Bio import Entrez, SeqIO
import time
from io import StringIO

# Set your email address (NCBI requires this)
Entrez.email = "aravind_plano@yahoo.com"

# Define the variants with their associated pangolin lineage queries.
# For wild type, we use the reference "Wuhan-Hu-1".
variants = {
    "Alpha": "B.1.1.7",
    "Beta": "B.1.351",
    "Gamma": "P.1",
    "Delta": "B.1.617.2",
    "Omicron": "B.1.1.529",
    "WildType": "Wuhan-Hu-1"
}

# This list will store all sampled SeqRecord objects from all variants.
combined_records = []

# Loop over each variant.
for variant_name, lineage in variants.items():
    # Build a query string: We require SARS-CoV-2, complete genome, and the specific lineage.
    query = f'SARS-CoV-2[Organism] AND "complete genome"[Title] AND "{lineage}"'
    print(f"\nSearching for {variant_name} with query:\n  {query}")

    # Step 1: Get the total count of records matching the query.
    search_handle = Entrez.esearch(db="nuccore", term=query, retmax=0)
    search_record = Entrez.read(search_handle)
    search_handle.close()

    total_records = int(search_record["Count"])
    print(f"{variant_name}: Total records found = {total_records}")

    if total_records == 0:
        print(f"No records found for {variant_name}; skipping...")
        continue

    # Step 2: Retrieve all IDs matching the query.
    # (If total_records is very large, consider batching.)
    search_handle = Entrez.esearch(db="nuccore", term=query, retmax=total_records)
    record = Entrez.read(search_handle)
    search_handle.close()
    candidate_ids = record["IdList"]
    print(f"{variant_name}: Retrieved {len(candidate_ids)} candidate IDs.")

    # Step 3: Fetch the FASTA records for these IDs.
    # Join all IDs into a comma-separated string.
    id_string = ",".join(candidate_ids)
    fetch_handle = Entrez.efetch(db="nuccore", id=id_string, rettype="fasta", retmode="text")
    fasta_data = fetch_handle.read()
    fetch_handle.close()

    # Parse multiple FASTA records from the fetched data.
    records = list(SeqIO.parse(StringIO(fasta_data), "fasta"))

    # Step 4: Append the variant name in square brackets to each record's header.
    for rec in records:
        rec.description = rec.description + f" [{variant_name}]"
        combined_records.append(rec)

    print(f"{variant_name}: Added {len(records)} records.")
    time.sleep(0.5)  # Pause to be respectful to NCBI servers

# Step 5: Write all combined records to one FASTA file.
output_file = "variant_samples.fasta"
with open(output_file, "w") as out_f:
    SeqIO.write(combined_records, out_f, "fasta")

print(f"\nAll sampled records with appended variant names have been saved to '{output_file}'.")
