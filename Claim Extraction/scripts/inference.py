from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "./vit5-contextualize-claim"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def contextualize_claim(claim, article):
    text = f"Contextualize: {claim} Article: {article}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
claim = "Bà ấy thường xuyên đi làm trễ"
article = "Bà Lan năm nay đã 50 tuổi. Bà ấy thường xuyên đi làm trễ."
print(contextualize_claim(claim, article))
