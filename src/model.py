import torch.nn as nn
from transformers import AutoModel


class BertMiniClaimDetector(nn.Module):
    """
    Simple BERT based binary classifier.

    We take the CLS token representation and pass it
    through a linear layer to get a single logit.
    """

    def __init__(self, model_name):
        super().__init__()

        # Load pretrained encoder 
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Final classification layer
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Used CLS token (first token) for sequence representation
        cls_embedding = outputs.last_hidden_state[:, 0]

        logits = self.classifier(cls_embedding)

        return logits.squeeze(-1)