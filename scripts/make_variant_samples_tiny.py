#!/usr/bin/env python3
"""
Create variant_samples_tiny.fasta in root/test_samples/ by:
- randomly sampling 7 FASTA records from variant_samples_small.fasta (no replacement)
- appending 3 made-up random DNA records (A/C/G/T only) that should fail alignment
  to the SARS-CoV-2 reference because they are not COVID-like.

Expected repo layout:
  root/
    scripts/   <-- this script lives here
    test_samples/
      variant_samples_small.fasta
      variant_samples_tiny.fasta  <-- output
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import random

Record = Tuple[str, str]  # (header_without_>, sequence_no_whitespace)


def read_fasta(path: Path) -> List[Record]:
  records: List[Record] = []
  header: str | None = None
  seq_chunks: List[str] = []

  with path.open("r", encoding="utf-8") as f:
    for raw in f:
      line = raw.strip()
      if not line:
        continue
      if line.startswith(">"):
        if header is not None:
          seq = "".join(seq_chunks).replace(" ", "").replace("\t", "")
          records.append((header, seq))
        header = line[1:].strip()
        seq_chunks = []
      else:
        seq_chunks.append(line)

  if header is not None:
    seq = "".join(seq_chunks).replace(" ", "").replace("\t", "")
    records.append((header, seq))

  if not records:
    raise ValueError(f"No FASTA records found in {path}")

  return records


def wrap_seq(seq: str, width: int = 60) -> str:
  return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def write_fasta(records: List[Record], out_path: Path) -> None:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w", encoding="utf-8", newline="\n") as f:
    for h, s in records:
      f.write(f">{h}\n")
      f.write(wrap_seq(s) + "\n")


def random_dna(length: int, rng: random.Random) -> str:
  alphabet = "ACGT"
  return "".join(rng.choice(alphabet) for _ in range(length))


def make_random_failure_records(rng: random.Random) -> List[Record]:
  """
  Three valid DNA sequences that are essentially random, so they should align poorly
  (or be rejected) against a SARS-CoV-2 reference in typical pipelines.
  """
  lengths = [1200, 1800, 2500]
  recs: List[Record] = []
  for i, L in enumerate(lengths, start=1):
    header = f"reject_tiny_{i:02d}_random_non_covid_len{L}"
    seq = random_dna(L, rng)
    recs.append((header, seq))
  return recs


def main() -> None:
  repo_root = Path(__file__).resolve().parents[1]
  test_samples_dir = repo_root / "test_samples"

  in_path = test_samples_dir / "variant_samples_small.fasta"
  out_path = test_samples_dir / "variant_samples_tiny.fasta"

  records = read_fasta(in_path)
  if len(records) < 7:
    raise ValueError(f"Expected at least 7 records in {in_path}, found {len(records)}")

  # deterministic randomness so the file is reproducible
  rng = random.Random(1337)

  # sample 7 distinct real records (order randomized)
  sampled7 = rng.sample(records, k=7)

  # add 3 "reject" records (valid DNA alphabet, but random/non-covid)
  rejects = make_random_failure_records(rng)

  tiny_records = sampled7 + rejects
  write_fasta(tiny_records, out_path)

  print(f"Wrote {len(tiny_records)} records to: {out_path}")
  print("  - 7 randomly sampled records from variant_samples_small.fasta (no replacement)")
  print("  - 3 random DNA records (A/C/G/T only) intended to fail alignment/QC")


if __name__ == "__main__":
  main()
