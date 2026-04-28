import torch
from transformers import BertForSequenceClassification, AutoConfig
from typing import Optional

class BertModel:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        config = AutoConfig.from_pretrained(model_path, num_labels=2)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path, config=config, torch_dtype=torch.float16
        )
        self.model.to(self.device)
        self.model.eval()

    def forward(self, inputs: dict) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(**inputs.to(self.device))
        return outputs.logits

    @torch.no_grad()
    def predict_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        return self.forward(inputs)