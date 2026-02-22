import re
import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer


def split_sentences(text):
    """
    Sentence splitter based on punctuation.
    This is sufficient for noisy Reddit text.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


class ClaimDetectionDataset(Dataset):
    """
    Binary claim detection at document level.
    Each example is a full Reddit post.
    """

    def __init__(self, csv_path, tokenizer_name, max_len=512):
        df = pd.read_csv(csv_path)

        self.texts = df["text"].tolist()
        self.labels = df["claim"].astype(int).tolist()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


class ClaimTypeDataset(Dataset):
    """
    Document level explicit vs implicit classification.
    Only includes posts that are labeled as claims.
    """

    def __init__(self, csv_path, tokenizer_name, max_len=512):
        df = pd.read_csv(csv_path)

        # keep only posts that are claims
        df = df[df["claim"] == 1].reset_index(drop=True)

        self.texts = df["text"].tolist()
        self.labels = df["explicit"].astype(int).tolist()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


class SentenceClaimTypeDataset(Dataset):
    """
    Sentence level explicit vs implicit classification.

    Each sentence inherits the document level label.
    This introduces noise but allows fine grained analysis.
    """

    def __init__(self, csv_path, tokenizer_name, max_len=256):
        df = pd.read_csv(csv_path)

        # only posts containing claims
        df = df[df["claim"] == 1].reset_index(drop=True)

        self.sentences = []
        self.labels = []
        self.doc_ids = []

        for doc_id, row in df.iterrows():
            label = int(row["explicit"])
            text = row["text"]

            for sent in split_sentences(text):
                self.sentences.append(sent)
                self.labels.append(label)
                self.doc_ids.append(doc_id)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=False
        )
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "doc_id": self.doc_ids[idx]
        }