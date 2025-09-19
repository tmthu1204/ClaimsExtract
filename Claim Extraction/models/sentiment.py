import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer

class SentimentModel:
    def __init__(self, model_name="wonrax/phobert-base-vietnamese-sentiment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = RobertaForSequenceClassification.from_pretrained(model_name)

    def predict(self, sentence: str) -> str:
        try:
            # PhoBERT yêu cầu văn bản đã word-segmented, nhưng để đơn giản ta vẫn encode trực tiếp
            # Truncate sentence if too long (PhoBERT max length is 512, but we use 256 for safety)
            encoded = self.tokenizer.encode(sentence, max_length=256, truncation=True, padding=True)
            input_ids = torch.tensor([encoded])
            
            with torch.no_grad():
                out = self.model(input_ids)
                probs = out.logits.softmax(dim=-1).tolist()[0]
            labels = ["NEG", "POS", "NEU"]
            return labels[probs.index(max(probs))]
        except Exception as e:
            print(f"Error in sentiment prediction: {e}")
            # If prediction fails, return neutral (keep the sentence)
            return "NEU"
