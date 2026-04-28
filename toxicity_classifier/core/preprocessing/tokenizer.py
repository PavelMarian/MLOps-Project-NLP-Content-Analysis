import re
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer


class Tokenizer:

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Fix для BERT

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove mentions/hashtags
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove extra whitespace/punct
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