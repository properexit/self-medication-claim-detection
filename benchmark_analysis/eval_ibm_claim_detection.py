import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score
)

from src.pipeline import ClaimPipeline

# Zero-shot evaluation of the Reddit trained claim detector
# on IBM Debater sentence level data.

DATA_PATH = "data/ibm_claims/ibm_claim_sentence_test.csv"
THRESHOLD = 0.5


def main():
    df = pd.read_csv(DATA_PATH)

    pipeline = ClaimPipeline()

    gold_labels = []
    pred_labels = []
    confidences = []
    query_types = []

    print(f"Running zero-shot evaluation on {len(df)} sentences...\n")

    # Inference loop 
    for _, row in df.iterrows():
        text = str(row["text"])
        gold = int(row["gold_claim"])

        pred, conf = pipeline.predict_claim(text)

        gold_labels.append(gold)
        pred_labels.append(int(pred))
        confidences.append(float(conf))

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_labels,
        pred_labels,
        average="binary",
        zero_division=0
    )
    accuracy = accuracy_score(gold_labels, pred_labels)

    print("IBM Claim Detection (Zero shot)")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"Accuracy: {accuracy:.3f}")

    # Confidence comparison
    correct_conf = [
        c for g, p, c in zip(gold_labels, pred_labels, confidences)
        if g == p
    ]
    wrong_conf = [
        c for g, p, c in zip(gold_labels, pred_labels, confidences)
        if g != p
    ]

    print("\nConfidence")
    print(f"Avg (correct): {np.mean(correct_conf):.3f}")
    print(f"Avg (wrong): {np.mean(wrong_conf):.3f}")


if __name__ == "__main__":
    main()