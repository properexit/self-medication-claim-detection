import os
import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.datasets import ClaimTypeDataset
from src.model import BertMiniClaimDetector
from src.utils import set_seed


MODEL_NAME = "prajjwal1/bert-mini"

BATCH_SIZE = 16
EPOCHS = 20
LR = 2e-5
SEED = 42
PATIENCE = 3


set_seed(SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "labels_v5.csv")
SAVE_DIR = os.path.join(PROJECT_ROOT, "experiments", "claim_type")
os.makedirs(SAVE_DIR, exist_ok=True)


def train():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    # Only posts that actually contain claims
    dataset = ClaimTypeDataset(
        csv_path=DATA_PATH,
        tokenizer_name=MODEL_NAME
    )

    labels = dataset.labels
    indices = list(range(len(dataset)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=SEED
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

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

    model = BertMiniClaimDetector(MODEL_NAME).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    # Handle imbalance between explicit and implicit
    labels = np.array(dataset.labels)

    num_explicit = (labels == 1).sum()
    num_implicit = (labels == 0).sum()

    pos_weight = torch.tensor(
        num_implicit / num_explicit,
        device=device,
        dtype=torch.float32
    )

    print(f"Explicit: {num_explicit} | Implicit: {num_implicit}")
    print(f"Using pos_weight = {pos_weight.item():.4f}")

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = 0.0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_preds, train_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels_batch)

            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels_batch.cpu().numpy())

        train_f1 = f1_score(train_labels, train_preds, average="macro")

        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_batch = batch["label"].to(device)

                logits = model(input_ids, attention_mask)
                preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()

                val_preds.extend(preds)
                val_labels.extend(labels_batch.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average="macro")

        print(
            f"Epoch {epoch + 1} | "
            f"Train Macro-F1: {train_f1:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0

            torch.save(
                model.state_dict(),
                os.path.join(SAVE_DIR, "bert_mini_claim_type_best.pt")
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"\nBest Validation Macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()