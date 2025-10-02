import os, sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==== [1] Load model đã train ====
#MODEL_DIR = os.path.join("..", "vit5-contextualize-claim")  # ../vit5-contextualize-claim
# MODEL_DIR = os.path.abspath("../vit5-contextualize-claim")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, BASE_DIR)

print("CURRENT_DIR:", CURRENT_DIR)
print("BASE_DIR:", BASE_DIR)
print("sys.path:")
for p in sys.path:
    print(" -", p)

print("Folders in ROOT_DIR:", os.listdir(BASE_DIR))

MODEL_DIR = os.path.join(BASE_DIR, "vit5-base-contextualize-claim")
print(f">> Loading model from {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ==== [2] Load input claims ====
INPUT_PATH = os.path.join(BASE_DIR, "Test", "data", "contextualized_claims.csv")
df = pd.read_csv(INPUT_PATH)

if "claim" not in df.columns or "article" not in df.columns:
    raise ValueError("CSV cần có 2 cột: 'claim' và 'article'")

print(f">> Loaded {len(df)} claims from {INPUT_PATH}")

# Tạo input text giống format lúc train
df["input_text"] = df.apply(lambda row: f"Contextualize: {row['claim']} Article: {row['article']}", axis=1)

# ==== [3] Generate predictions ====
predictions = []
for i, text in enumerate(df["input_text"].tolist()):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,
        length_penalty=1.0,
        early_stopping=True
    )
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(pred.strip())

# ==== [4] Save output ====
df["contextualized_prediction"] = predictions
OUTPUT_PATH = os.path.join("data", "contextualized_claims_pred.csv")
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f">> DONE! Predictions saved to {OUTPUT_PATH}")
