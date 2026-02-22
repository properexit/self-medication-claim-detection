import torch.nn as nn
from transformers import AutoModel


class BertMiniSpanTagger(nn.Module):
    """
    Token level BIO span tagger built on BERT mini.
    Labels: 0 = O, 1 = I, 2 = B
    """

    def __init__(self, model_name, num_labels=3):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Linear layer for per token classification
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        token_embeddings = outputs.last_hidden_state
        logits = self.classifier(token_embeddings)

        return logits