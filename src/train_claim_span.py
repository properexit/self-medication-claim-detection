import os
import torch
import pandas as pd

from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.datasets_span import SpanDataset
from src.model_span import BertMiniSpanTagger
from src.utils import set_seed


MODEL_NAME = "prajjwal1/bert-mini"

BATCH_SIZE = 8
EPOCHS = 15
LR = 2e-5
SEED = 42


set_seed(SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "labels_v5.csv")
SAVE_DIR = os.path.join(PROJECT_ROOT, "experiments", "span_detection")
os.makedirs(SAVE_DIR, exist_ok=True)


def train():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    # Only train span model on posts that actually contain claims
    df = pd.read_csv(DATA_PATH)
    df = df[df["claim"] == 1].reset_index(drop=True)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED
    )

    train_dataset = SpanDataset(
        dataframe=train_df,
        tokenizer_name=MODEL_NAME
    )

    val_dataset = SpanDataset(
        dataframe=val_df,
        tokenizer_name=MODEL_NAME
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = BertMiniSpanTagger(MODEL_NAME).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    # Ignore label 0 ("O") when computing loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    best_val_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)

            loss = criterion(
                logits.view(-1, 3),
                labels.view(-1)
            )

            loss.backward()
            optimizer.step()

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)

                preds = logits.argmax(dim=-1).cpu().numpy().flatten()
                gold = labels.cpu().numpy().flatten()

                # Only evaluate on non-O tokens
                mask = gold != 0
                all_preds.extend(preds[mask])
                all_labels.extend(gold[mask])

        val_f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"Epoch {epoch + 1} | Val Token Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(),
                os.path.join(SAVE_DIR, "bert_mini_span_best.pt")
            )

    print(f"\nBest Validation Token Macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()