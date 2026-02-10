import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

ARTIFACT_PATH = "artifacts/Bert_Training_files"


class BertForMultiLabel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(
            self.bert.config.hidden_size,
            num_labels
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.pooler_output
        return self.fc(self.dropout(pooled))



@torch.no_grad()
def load_model():
    checkpoint = torch.load(
        f"{ARTIFACT_PATH}/bert_model.pt",
        map_location="cpu"
    )

    model = BertForMultiLabel(
        model_name=checkpoint["model_name"],
        num_labels=checkpoint["num_labels"]
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tokenizer = BertTokenizerFast.from_pretrained(ARTIFACT_PATH)

    return model, tokenizer


@torch.no_grad()
def predict(text, model, tokenizer, threshold=0.5):
    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    logits = model(
        encoded["input_ids"],
        encoded["attention_mask"]
    )

    probs = torch.sigmoid(logits).squeeze().tolist()

    scores = dict(zip(LABELS, [round(p, 4) for p in probs]))
    flags = {k: v >= threshold for k, v in scores.items()}

    return scores, flags
