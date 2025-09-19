import os
import argparse
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import torch


def load_model(ckpt: str):
    if os.path.isdir(ckpt) and os.path.exists(os.path.join(ckpt, "config.json")):
        print(f"✅ Loading model from checkpoint: {ckpt}")
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
    else:
        print(f"❌ Không tìm thấy checkpoint hợp lệ ở {ckpt}. Dùng model mặc định: google/mt5-large")
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-large")
    return tokenizer, model


def preprocess_function(examples, tokenizer, max_input=256, max_target=128):
    inputs = [f"claim: {c} article: {a}" for c, a in zip(examples["claim"], examples["article"])]
    model_inputs = tokenizer(inputs, max_length=max_input, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["decontext_claim"], max_length=max_target,
                           truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(args):
    # 1. Load dataset từ CSV
    dataset = load_dataset("csv", data_files=args.data_path)["train"]

    # 2. Load model & tokenizer
    tokenizer, model = load_model(args.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"💻 Using device: {device}")

    # 3. Preprocess
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # 4. Generate predictions
    preds, labels = [], []
    for example in dataset:
        input_text = f"claim: {example['claim']} article: {example['article']}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
        output = model.generate(**inputs, max_length=128)
        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        preds.append(pred)
        labels.append(example["decontext_claim"])

    # 5. Evaluate bằng ROUGE
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=preds, references=labels)
    print("📊 Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # 6. Lưu file kết quả
    out_df = pd.DataFrame({"claim": dataset["claim"],
                           "article": dataset["article"],
                           "gold": labels,
                           "pred": preds})
    out_df.to_csv("results.csv", index=False, encoding="utf-8-sig")
    print("✅ Saved predictions to results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints")
    parser.add_argument("--data_path", type=str, default="./data/test.csv")
    args = parser.parse_args()
    main(args)
