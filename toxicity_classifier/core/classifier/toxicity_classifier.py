import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Union
from toxicity_classifier.bert import BertModel
from toxicity_classifier.preprocessing.tokenizer import Tokenizer


class ToxicityClassifier:

    def __init__(self, model_path: str, device: Optional[str] = None, threshold: float = 0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.tokenizer = Tokenizer(model_path)
        self.model = BertModel(model_path, device=self.device)

    @torch.no_grad()
    def predict_single(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer([text])
        logits = self.model.predict_logits(
            inputs["input_ids"],
            inputs["attention_mask"]
        )
        toxic_prob = F.softmax(logits, dim=-1)[0, 1].cpu().item()  # [0,1]

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
        logits = self.model.predict_logits(
            inputs["input_ids"],
            inputs["attention_mask"]
        )
        toxic_probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # [N]

        cleaned_texts = self.tokenizer.preprocess_texts(texts)
        results = []
        for i, (text, cleaned, prob) in enumerate(zip(texts, cleaned_texts, toxic_probs)):
            results.append({
                "text": text,
                "cleaned_text": cleaned,
                "toxic_prob": float(prob),
                "label": "toxic" if prob > self.threshold else "non-toxic",
                "confidence": float(max(prob, 1 - prob))
            })
        return results

    def predict(self, texts: Union[str, List[str]], threshold: Optional[float] = None, **kwargs) -> Union[
        Dict, List[Dict]]:
        threshold = threshold or self.threshold

        if isinstance(texts, str):
            return self.predict_single(texts)
        return self.predict_batch(texts)

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def batch_size(self, texts: List[str]) -> int:
        avg_len = np.mean([len(t) for t in texts])
        return 32 if avg_len < 100 else 16

    def __call__(self, texts: Union[str, List[str]], **kwargs):
        return self.predict(texts, **kwargs)