# utils/dataset.py
from torch.utils.data import Dataset
import torch

class DecontextDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len=512, max_output_len=128):
        """
        data: list of dict {'claim':..., 'article':..., 'decontext_claim':...}
        tokenizer: HuggingFace tokenizer
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"claim: {item['claim']} context: {item['article']}"
        target_text = item['decontext_claim']

        inputs = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_input_len,
            padding="max_length",
            return_tensors="pt"
        )

        labels = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_output_len,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze()
        }
