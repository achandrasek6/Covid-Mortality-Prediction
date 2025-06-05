from Bio import Entrez, SeqIO

# Set your email address (required by NCBI)
Entrez.email = "your.email@example.com"

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


