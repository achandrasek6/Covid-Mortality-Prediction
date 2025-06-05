#!/usr/bin/env python3
import csv

# Specify the input CSV file name (adjust as needed)
csv_file = "feature_matrix_train.csv"

# Open and read the CSV file.
with open(csv_file, "r", newline="") as f:
    reader = csv.reader(f)
    # The first row is assumed to be the header
    header = next(reader)
    # Columns 0, 1, and 2 are "SampleID", "Variant", and "Global CFR"
    feature_labels = header[3:]

    # Process each sample row
    for row in reader:
        sample_id = row[0]
        variant = row[1]
        # Convert feature values to integers (assumes they are "0" or "1")
        features = list(map(int, row[3:]))

        # Get the positions (column labels) where the value is 1.
        diff_positions = [label for label, val in zip(feature_labels, features) if val == 1]

        # Print out the sample and its differing positions
        print(f"Sample '{sample_id}' (Variant: {variant}) differs at positions:")
        if diff_positions:
            print(", ".join(diff_positions))
        else:
            print("No differences found.")
        print()  # blank line between samples
