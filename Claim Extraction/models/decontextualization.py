# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch

# class Decontextualizer:
#     def __init__(self, model_name="VietAI/vit5-large-vietnews-summarization", device=None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

#     def contextualize(self, claim: str, article: str, max_input_length=512, max_output_length=128):
#         input_text = f"claim: {claim} context: {article}"
#         inputs = self.tokenizer(
#             input_text, return_tensors="pt", truncation=True,
#             max_length=max_input_length, padding="max_length"
#         ).to(self.device)
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_length=max_output_length,
#                 num_beams=4,
#                 early_stopping=True
#             )
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
# models/decontextualization.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Decontextualizer:
    def __init__(self, model_name="VietAI/vit5-large-vietnews-summarization", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def contextualize(self, claim: str, article: str, max_input_length=512, max_output_length=128):
        """
        Input:
            claim: câu claim mơ hồ
            article: nội dung bài báo
        Output:
            decontextualized claim
        """
        input_text = f"claim: {claim} context: {article}"
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True,
            max_length=max_input_length, padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_output_length,
                num_beams=4,
                early_stopping=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
