import os
import pandas as pd

# Prepare IBM Debater test set for sentence level claim detection.
# The original file has no header and includes extra metadata columns.
# Here we only keep the sentence, gold label.

INPUT_PATH = "data/ibm_raw/test_set.csv"
OUTPUT_PATH = "data/ibm_claims/ibm_claim_sentence_test.csv"

# Column indices based on manual inspection
COL_SENTENCE = 3
COL_LABEL = 6


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"File not found: {INPUT_PATH}")

    # IBM test_set.csv has no header row
    df = pd.read_csv(INPUT_PATH, header=None)

    if df.shape[1] < 7:
        raise ValueError(
            f"Unexpected format. Expected >= 7 columns, got {df.shape[1]}"
        )

    clean_df = pd.DataFrame({
        "text": df[COL_SENTENCE].astype(str).str.strip(),
        "gold_claim": df[COL_LABEL].astype(int)
    })

    # Make sure labels are binary
    if not clean_df["gold_claim"].isin([0, 1]).all():
        raise ValueError("gold_claim must be 0 or 1")

    print("Label distribution:")
    print(clean_df["gold_claim"].value_counts())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    clean_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()