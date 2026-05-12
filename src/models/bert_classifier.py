import re
from typing import List, Dict, Any, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoTokenizer, AutoConfig,
    BertForSequenceClassification
)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

class Tokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^\w\sёа-я]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess_texts(self, texts: List[str]) -> List[str]:
        return [self.clean_text(t) for t in texts]

    def tokenize(self, texts: List[str], max_length: int = 512, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        cleaned = self.preprocess_texts(texts)
        tokenized = self.tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors=return_tensors
        )
        return tokenized

    def __call__(self, texts: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        return self.tokenize(texts, **kwargs)


class BertModel:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        config = AutoConfig.from_pretrained(model_path, num_labels=2)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path, config=config, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    @torch.no_grad()
    def predict_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids, attention_mask)


class ToxicityClassifier:
    def __init__(self, model_path: str, device: Optional[str] = None, threshold: float = 0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.tokenizer = Tokenizer(model_path)
        self.model = BertModel(model_path, device=self.device)

    @torch.no_grad()
    def predict_single(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer([text])
        logits = self.model.predict_logits(inputs["input_ids"], inputs["attention_mask"])
        toxic_prob = F.softmax(logits, dim=-1)[0, 1].cpu().item()
        return {
            "text": text,
            "cleaned_text": self.tokenizer.preprocess_texts([text])[0],
            "toxic_prob": toxic_prob,
            "label": "toxic" if toxic_prob > self.threshold else "non-toxic",
            "confidence": max(toxic_prob, 1 - toxic_prob)
        }

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        inputs = self.tokenizer(texts)
        logits = self.model.predict_logits(inputs["input_ids"], inputs["attention_mask"])
        toxic_probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        cleaned_texts = self.tokenizer.preprocess_texts(texts)
        results = []
        for text, cleaned, prob in zip(texts, cleaned_texts, toxic_probs):
            results.append({
                "text": text,
                "cleaned_text": cleaned,
                "toxic_prob": float(prob),
                "label": "toxic" if prob > self.threshold else "non-toxic",
                "confidence": float(max(prob, 1 - prob))
            })
        return results

    def predict(self, texts: Union[str, List[str]], threshold: Optional[float] = None) -> Union[Dict, List[Dict]]:
        thr = threshold or self.threshold
        old_thr = self.threshold
        if thr != old_thr:
            self.threshold = thr
            result = self.predict_single(texts) if isinstance(texts, str) else self.predict_batch(texts)
            self.threshold = old_thr
            return result
        return self.predict_single(texts) if isinstance(texts, str) else self.predict_batch(texts)

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def __call__(self, texts: Union[str, List[str]], **kwargs):
        return self.predict(texts, **kwargs.get("threshold"))