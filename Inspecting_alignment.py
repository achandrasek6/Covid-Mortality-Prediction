from Bio import AlignIO

alignment_file = "aligned.fasta"
alignment = AlignIO.read(alignment_file, "fasta")
print(alignment)
