import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from src.pipeline import ClaimPipeline

# Collect gold mismatches for manual inspection.

DATA_PATH = "data/raw/labels_v5.csv"
OUTPUT_PATH = "analysis/error_samples.csv"

pipeline = ClaimPipeline()

df = pd.read_csv(DATA_PATH)

error_records = []

for _, row in df.iterrows():
    text = row["text"]
    gold_label = int(row["claim"])

    prediction = pipeline(text)
    pred_label = int(prediction.get("claim", False))

    if gold_label != pred_label:
        error_records.append({
            "text": text,
            "gold_claim": gold_label,
            "predicted_claim": pred_label,
            "claim_confidence": prediction.get("claim_confidence", 0.0),
            "predicted_span": prediction.get("span", "")
        })

error_df = pd.DataFrame(error_records)
error_df = error_df.sort_values(
    by="claim_confidence",
    ascending=False
)
error_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(error_df)} error cases to {OUTPUT_PATH}")