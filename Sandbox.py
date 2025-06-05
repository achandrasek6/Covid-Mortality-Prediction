from Bio import SeqIO
import random
random.seed(42)

# Specify the path to your FASTA file
fasta_file = "random_sars_cov2.fasta"

# Iterate over each record in the FASTA file
for record in SeqIO.parse(fasta_file, "fasta"):
    print(f"{record.id}: {len(record.seq)}")

from Bio import Entrez, SeqIO
import random

# Set your email address (required by NCBI)
Entrez.email = "your.email@example.com"

# Step 1: Search for SARS-CoV-2 sequences in NCBI's nuccore database
# You can adjust 'retmax' to retrieve more IDs if needed.
search_handle = Entrez.esearch(db="nuccore", term="SARS-CoV-2[Organism]", retmax=1000)
search_record = Entrez.read(search_handle)
search_handle.close()

# Extract the list of IDs
id_list = search_record["IdList"]
print(f"Found {len(id_list)} sequences.")

# Step 2: Randomly select a subset (e.g., 10 sequences)
sample_size = 10
if len(id_list) < sample_size:
    sample_ids = id_list
else:
    sample_ids = random.sample(id_list, sample_size)
print(f"Randomly selected IDs: {sample_ids}")

# Step 3: Fetch the FASTA records for the selected IDs
fetch_handle = Entrez.efetch(db="nuccore", id=",".join(sample_ids), rettype="fasta", retmode="text")
fasta_data = fetch_handle.read()
fetch_handle.close()

# Optionally, save the FASTA data to a file
output_file = "random_sars_cov2.fasta"
with open(output_file, "w") as f:
    f.write(fasta_data)

print(f"Fetched FASTA sequences saved to {output_file}")

# You can also parse and process the sequences:
# for record in SeqIO.parse(output_file, "fasta"):
#     print(record.id, len(record.seq))