import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from collections import defaultdict

from src.pipeline import ClaimPipeline
from analysis.hedging import count_hedges, has_hedge

# Compare hedging behavior in predicted spans
# for explicit vs implicit claims.

DATA_PATH = "data/raw/labels_v5.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    df = df[df["claim"] == 1].reset_index(drop=True)

    pipeline = ClaimPipeline()

    stats = defaultdict(list)

    for _, row in df.iterrows():
        text = row["text"]
        gold_type = "explicit" if row["explicit"] == 1 else "implicit"

        result = pipeline.predict_on_long_text(text)
        span = result.get("span")

        if not span or span == "NO_SPAN_PREDICTED":
            continue

        stats[gold_type].append({
            "hedge_count": count_hedges(span),
            "has_hedge": has_hedge(span)
        })

    for claim_type in ["explicit", "implicit"]:
        total = len(stats[claim_type])
        with_hedge = sum(x["has_hedge"] for x in stats[claim_type])

        avg_hedges = (
            sum(x["hedge_count"] for x in stats[claim_type]) / total
            if total > 0 else 0
        )

        print(f"\n{claim_type.upper()} CLAIMS")
        print(f"Total spans: {total}")

        if total > 0:
            print(f"With hedging: {with_hedge} ({with_hedge / total:.2%})")
            print(f"Avg hedges per span: {avg_hedges:.2f}")
        else:
            print("No spans found.")
            

if __name__ == "__main__":
    main()