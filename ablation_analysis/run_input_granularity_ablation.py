import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.pipeline import ClaimPipeline, split_sentences_regex


DATA_PATH = "data/raw/labels_v5.csv"

# decision threshold for binary classifier
THRESHOLD = 0.5

# stricter settings for precision oriented variant
HIGH_CONF_THRESHOLD = 0.6
MIN_SPAN_WORDS = 2

def evaluate(gold, preds):
    p, r, f, _ = precision_recall_fscore_support(
        gold,
        preds,
        average="binary",
        zero_division=0
    )
    return round(p, 3), round(r, 3), round(f, 3)


def main():
    df = pd.read_csv(DATA_PATH)

    texts = df["text"].tolist()
    gold_labels = df["claim"].astype(int).tolist()

    pipeline = ClaimPipeline()

    preds_full_post = []
    preds_any_sentence = []
    preds_best_sentence = []

    for text in texts:

        # 1. Full post
        pred_full, _ = pipeline.predict_claim(text)
        preds_full_post.append(int(pred_full))

        # 2. Any sentence (recall-oriented)
        sentences = split_sentences_regex(text)

        max_prob = 0.0
        for sent in sentences:
            _, prob = pipeline.predict_claim(sent)
            if prob > max_prob:
                max_prob = prob

        preds_any_sentence.append(int(max_prob > THRESHOLD))

        # 3. Best sentence (precision-oriented)
        result = pipeline.predict_on_long_text(text)

        claim_conf = result.get("claim_confidence", 0.0)
        span = result.get("span")

        has_span = span not in (None, "NO_SPAN_PREDICTED")
        span_len_ok = (
            has_span and len(span.split()) >= MIN_SPAN_WORDS
        )

        high_conf = claim_conf > HIGH_CONF_THRESHOLD

        preds_best_sentence.append(
            int(high_conf and has_span and span_len_ok)
        )

    p1, r1, f1 = evaluate(gold_labels, preds_full_post)
    p2, r2, f2 = evaluate(gold_labels, preds_any_sentence)
    p3, r3, f3 = evaluate(gold_labels, preds_best_sentence)

    print("\nInput Granularity Ablation\n")
    print(f"{'Variant':<20} {'P':<8} {'R':<8} {'F1':<8}")
    print("-" * 45)

    print(f"{'Full Post':<20} {p1:<8} {r1:<8} {f1:<8}")
    print(f"{'Any Sentence':<20} {p2:<8} {r2:<8} {f2:<8}")
    print(f"{'Best Sentence':<20} {p3:<8} {r3:<8} {f3:<8}")


if __name__ == "__main__":
    main()