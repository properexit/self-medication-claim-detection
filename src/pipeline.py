import re
import torch
import numpy as np
from transformers import AutoTokenizer

from src.model import BertMiniClaimDetector
from src.model_span import BertMiniSpanTagger


MODEL_NAME = "prajjwal1/bert-mini"
MAX_LEN = 512

CLAIM_MODEL_PATH = "experiments/claim_detection/bert_mini_claim_best.pt"
CLAIM_TYPE_MODEL_PATH = "experiments/claim_type/bert_mini_claim_type_best.pt"
SPAN_MODEL_PATH = "experiments/span_detection/bert_mini_span_best.pt"


def split_sentences_regex(text):
    """
    Sentence splitting using punctuation and newlines.
    Good enough for Reddit-style text.
    """
    parts = re.split(r'(?<=[\.\?\!])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]


class ClaimPipeline:
    """
    Inference pipeline combining:
    - claim detection
    - claim type (explicit / implicit)
    - span extraction

    Each component is trained separately.
    """

    def __init__(self, device=None):
        self.device = (
            device
            or ("mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True
        )

        # Load models
        self.claim_model = BertMiniClaimDetector(MODEL_NAME).to(self.device)
        self.claim_model.load_state_dict(
            torch.load(CLAIM_MODEL_PATH, map_location=self.device)
        )
        self.claim_model.eval()

        self.claim_type_model = BertMiniClaimDetector(MODEL_NAME).to(self.device)
        self.claim_type_model.load_state_dict(
            torch.load(CLAIM_TYPE_MODEL_PATH, map_location=self.device)
        )
        self.claim_type_model.eval()

        self.span_model = BertMiniSpanTagger(MODEL_NAME).to(self.device)
        self.span_model.load_state_dict(
            torch.load(SPAN_MODEL_PATH, map_location=self.device)
        )
        self.span_model.eval()

    def predict_claim(self, text):
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logit = self.claim_model(
                enc["input_ids"],
                enc["attention_mask"]
            )

        prob = torch.sigmoid(logit).item()
        return prob > 0.5, prob

    def predict_claim_type(self, text):
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logit = self.claim_type_model(
                enc["input_ids"],
                enc["attention_mask"]
            )

        prob = torch.sigmoid(logit).item()
        label = "explicit" if prob > 0.5 else "implicit"
        return label, prob

    def predict_span(self, text):
        """
        Extract one claim span using BIO decoding.
        """
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        offsets = enc["offset_mapping"].squeeze(0).tolist()

        with torch.no_grad():
            logits = self.span_model(input_ids, attention_mask)

        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # 0 = O, 1 = I, 2 = B
        B_probs = probs[:, 2]
        I_probs = probs[:, 1]

        valid_indices = [
            i for i, (s, e) in enumerate(offsets)
            if not (s == 0 and e == 0)
        ]

        if not valid_indices:
            return None

        best_b = max(valid_indices, key=lambda i: B_probs[i])

        # If even the strongest B token is weak, skip
        if B_probs[best_b] < 0.2:
            return None

        start = best_b
        end = best_b

        i = best_b + 1
        while i in valid_indices and I_probs[i] > 0.2:
            end = i
            i += 1

        start_char = offsets[start][0]
        end_char = offsets[end][1]

        if start_char >= end_char:
            return None

        return text[start_char:end_char]

    def __call__(self, text):
        output = {}

        has_claim, claim_prob = self.predict_claim(text)
        output["claim"] = has_claim
        output["claim_confidence"] = round(claim_prob, 3)

        if not has_claim:
            return output

        claim_type, type_prob = self.predict_claim_type(text)
        output["claim_type"] = claim_type
        output["claim_type_confidence"] = round(type_prob, 3)

        span = self.predict_span(text)
        output["span"] = span if span is not None else "NO_SPAN_PREDICTED"

        return output

    def predict_on_long_text(self, text, top_k=3):
        """
        For longer posts, score sentences first,
        then re-rank based on span quality.
        """
        sentences = split_sentences_regex(text)

        scored = []
        for sent in sentences:
            _, prob = self.predict_claim(sent)
            scored.append((sent, prob))

        scored.sort(key=lambda x: x[1], reverse=True)
        candidates = scored[:top_k]

        chosen = None
        chosen_score = 0.0

        for sent, claim_prob in candidates:
            result = self(sent)

            if not result.get("claim"):
                continue

            if result.get("span") in (None, "NO_SPAN_PREDICTED"):
                continue

            span_len = len(result["span"].split())
            score = claim_prob * span_len

            if score > chosen_score:
                chosen_score = score
                chosen = result
                chosen["source_sentence"] = sent

        if chosen is not None:
            return chosen

        return {
            "claim": False,
            "reason": "no reliable claim sentence detected"
        }