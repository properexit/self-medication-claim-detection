import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from collections import Counter


def normalize(text):
    """
    Lowercase + remove punctuation for rough matching.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(a, b):
    """
    Token level F1 between two token lists.
    Used for fuzzy span matching.
    """
    a_cnt = Counter(a)
    b_cnt = Counter(b)

    common = sum((a_cnt & b_cnt).values())
    if common == 0:
        return 0.0

    precision = common / sum(a_cnt.values())
    recall = common / sum(b_cnt.values())
    return 2 * precision * recall / (precision + recall)


def find_best_span(text, span_text, min_f1=0.5):
    """
    Try to align annotated span to the document
    using sliding window fuzzy matching.
    """
    norm_text = normalize(text)
    norm_span = normalize(span_text)

    doc_tokens = norm_text.split()
    span_tokens = norm_span.split()

    best_score = 0.0
    best_window = None

    for i in range(len(doc_tokens)):
        max_j = min(i + len(span_tokens) + 6, len(doc_tokens))
        for j in range(i + 1, max_j):
            window = doc_tokens[i:j]
            score = token_f1(window, span_tokens)

            if score > best_score:
                best_score = score
                best_window = (i, j)

    if best_window is None or best_score < min_f1:
        return None

    # Map token indices back to character offsets
    char_offsets = []
    cursor = 0

    for tok in text.split():
        start = text.find(tok, cursor)
        if start == -1:
            continue
        end = start + len(tok)
        char_offsets.append((start, end))
        cursor = end

    start_tok, end_tok = best_window

    if end_tok - 1 >= len(char_offsets):
        return None

    start_char = char_offsets[start_tok][0]
    end_char = char_offsets[end_tok - 1][1]

    return start_char, end_char


class SpanDataset(Dataset):
    """
    BIO styled span tagging dataset.
    Uses explicit span if available, otherwise implicit.
    """

    def __init__(self, tokenizer_name, max_len=512,
                 csv_path=None, dataframe=None):

        assert csv_path is not None or dataframe is not None, \
            "Provide either csv_path or dataframe"

        if dataframe is not None:
            self.df = dataframe.reset_index(drop=True)
        else:
            self.df = pd.read_csv(csv_path)
            self.df = self.df[self.df["claim"] == 1].reset_index(drop=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True
        )
        self.max_len = max_len

        self.samples = []

        exact, fuzzy, dropped = 0, 0, 0

        for _, row in self.df.iterrows():
            text = row["text"]

            # Prefer explicit span
            if row["explicit_span"] != -1:
                span_text = str(row["explicit_span"])
            elif row["implicit_span"] != -1:
                span_text = str(row["implicit_span"])
            else:
                continue

            start_char = text.find(span_text)

            if start_char != -1:
                end_char = start_char + len(span_text)
                exact += 1
            else:
                match = find_best_span(text, span_text)
                if match is None:
                    dropped += 1
                    continue
                start_char, end_char = match
                fuzzy += 1

            self.samples.append({
                "text": text,
                "start_char": start_char,
                "end_char": end_char
            })

        print(
            f"Span alignment -> Exact: {exact}, "
            f"Fuzzy: {fuzzy}, Dropped: {dropped}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offsets = encoding["offset_mapping"].squeeze(0)
        labels = torch.zeros(len(offsets), dtype=torch.long)

        start_char = sample["start_char"]
        end_char = sample["end_char"]

        # Assign I labels where token fully falls inside span
        for i, (tok_start, tok_end) in enumerate(offsets.tolist()):
            if tok_start == tok_end == 0:
                continue
            if tok_start >= start_char and tok_end <= end_char:
                labels[i] = 1  # I

        # First inside token becomes B
        inside = (labels == 1).nonzero(as_tuple=True)[0]
        if len(inside) > 0:
            labels[inside[0]] = 2  # B

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels
        }