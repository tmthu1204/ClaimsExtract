# import pandas as pd
# from datasets import Dataset
# import evaluate
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM,
#     DataCollatorForSeq2Seq,
#     Seq2SeqTrainingArguments,
#     Seq2SeqTrainer
# )
# from sklearn.model_selection import train_test_split
# import torch

# # Check if CUDA is available
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU: {torch.cuda.get_device_name(0)}")
#     print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# print(">> [1] Load & split dataset")
# # --- Sửa phần đọc CSV ---
# df = pd.read_csv("data/contextualized_claims.csv",
#                  header=None, skipinitialspace=True)
# # Xóa dấu " nếu có
# df = df.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)
# # Đặt tên cột
# df.columns = ["claim", "article", "contextualized_claim"]

# print("   Số dòng dữ liệu:", len(df))

# def build_input(row):
#     return f"Contextualize: {row['claim']} Article: {row['article']}"

# df["input_text"] = df.apply(build_input, axis=1)
# df["target_text"] = df["contextualized_claim"]

# train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
# valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
# print("   Train:", len(train_df), "| Valid:", len(valid_df), "| Test:", len(test_df))

# train_dataset = Dataset.from_pandas(train_df[["input_text", "target_text"]])
# valid_dataset = Dataset.from_pandas(valid_df[["input_text", "target_text"]])
# test_dataset  = Dataset.from_pandas(test_df[["input_text", "target_text"]])

# # Phần còn lại giữ nguyên
# print(">> [2] Load tokenizer")
# model_name = "VietAI/vit5-large-vietnews-summarization"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# def preprocess(batch):
#     # Đảm bảo tất cả target_text là string, None -> ""
#     target_texts = [str(x) if x is not None else "" for x in batch["target_text"]]
    
#     model_inputs = tokenizer(batch["input_text"], max_length=512, truncation=True)
#     labels = tokenizer(target_texts, max_length=128, truncation=True)
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs


# print("   Preprocessing train...")
# train_dataset = train_dataset.map(preprocess, batched=True)
# print("   Preprocessing valid...")
# valid_dataset = valid_dataset.map(preprocess, batched=True)
# print("   Preprocessing test...")
# test_dataset  = test_dataset.map(preprocess, batched=True)

# print(">> [3] Load model + Data Collator")
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# print(">> [4] Load metric (ROUGE)")
# rouge = evaluate.load("rouge")

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
#     result = {k: v.mid.fmeasure * 100 for k, v in result.items()}
#     return {k: round(v, 2) for k, v in result.items()}

# print(">> [5] Init Trainer")
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./vit5-contextualize-claim",
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=3e-5,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     num_train_epochs=2,
#     weight_decay=0.01,
#     save_total_limit=1,
#     logging_dir="./logs",
#     logging_steps=10,
#     report_to="none",
#     predict_with_generate=True,
#     fp16=True,
#     dataloader_pin_memory=False,
#     gradient_accumulation_steps=4,
#     max_grad_norm=1.0,
#     warmup_steps=50,
#     save_steps=100,
#     eval_steps=100
# )

# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=valid_dataset,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# print(">> [6] Start training...")
# trainer.train()

# print(">> [7] Evaluate on test set")
# metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
# print("   Test metrics:", metrics)

# print(">> [8] Save predictions")
# from datasets import Dataset

# # Dự đoán trên test dataset
# predictions, labels, _ = trainer.predict(test_dataset)
# decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
# labels_ids = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
# decoded_labels = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

# # Lưu ra DataFrame
# import pandas as pd
# output_df = pd.DataFrame({
#     "input_text": test_df["input_text"].tolist(),
#     "expected_result": decoded_labels,
#     "actual_result": decoded_preds
# })

# # Lưu ra CSV
# output_df.to_csv("./vit5-contextualize-claim/test_predictions.csv", index=False)
# print("   Predictions saved to './vit5-contextualize-claim/test_predictions.csv'")

# print(">> [9] Save model")
# trainer.save_model("./vit5-contextualize-claim")
# tokenizer.save_pretrained("./vit5-contextualize-claim")
# print(">> DONE")

import os
import pandas as pd
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from sklearn.metrics import f1_score

# ---- Tăng timeout tải model ----
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 phút

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(">> [1] Load & split dataset")
# --- Sửa phần đọc CSV ---
df = pd.read_csv("data/contextualized_claims_train.csv",
                 header=None, skipinitialspace=True)
# Xóa dấu " nếu có
df = df.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)
# Đặt tên cột
df.columns = ["claim", "article", "contextualized_claim"]

print("   Số dòng dữ liệu:", len(df))

df["input_text"] = df.apply(lambda row: f"Contextualize: {row['claim']} Article: {row['article']}", axis=1)
df["target_text"] = df["contextualized_claim"]

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
print("   Train:", len(train_df), "| Valid:", len(valid_df), "| Test:", len(test_df))

# --- Kiểm tra overlap giữa các tập ---
def check_overlap(set1, set2, col="claim"):
    overlap = set(set1[col]) & set(set2[col])
    return len(overlap), list(overlap)[:5]  # in thử 5 mẫu trùng

train_valid_overlap, train_valid_samples = check_overlap(train_df, valid_df)
train_test_overlap, train_test_samples = check_overlap(train_df, test_df)
valid_test_overlap, valid_test_samples = check_overlap(valid_df, test_df)

print(f">> Overlap Train-Valid: {train_valid_overlap} samples")
if train_valid_overlap:
    print("   Ví dụ:", train_valid_samples)

print(f">> Overlap Train-Test: {train_test_overlap} samples")
if train_test_overlap:
    print("   Ví dụ:", train_test_samples)

print(f">> Overlap Valid-Test: {valid_test_overlap} samples")
if valid_test_overlap:
    print("   Ví dụ:", valid_test_samples)


train_dataset = Dataset.from_pandas(train_df[["input_text", "target_text"]])
valid_dataset = Dataset.from_pandas(valid_df[["input_text", "target_text"]])
test_dataset  = Dataset.from_pandas(test_df[["input_text", "target_text"]])

print(">> [2] Load tokenizer & model")
model_name = "VietAI/vit5-base-vietnews-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess(batch):
    # Chuyển sang list[str] để tokenizer không lỗi
    inputs = [str(x) for x in batch["input_text"]]
    targets = [str(x) for x in batch["target_text"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("   Preprocessing train...")
train_dataset = train_dataset.map(preprocess, batched=True)
print("   Preprocessing valid...")
valid_dataset = valid_dataset.map(preprocess, batched=True)
print("   Preprocessing test...")
test_dataset  = test_dataset.map(preprocess, batched=True)

print(">> [3] Data collator")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

print(">> [4] Load metrics")
rouge = evaluate.load("rouge")
bleu = evaluate.load("sacrebleu")

def clean_and_decode(sequences, tokenizer, pad_token_id=None):
    """
    sequences: numpy array (batch, seq_len) hoặc list[list[int]]
    trả về list[str] đã decode (strip)
    """
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id

    # đảm bảo là list[list[int]]
    if isinstance(sequences, np.ndarray):
        seqs = sequences.tolist()
    else:
        seqs = sequences

    decoded = []
    for seq in seqs:
        # convert mọi phần tử sang int an toàn
        cleaned = []
        for x in seq:
            try:
                xi = int(x)
            except Exception:
                xi = pad_token_id
            # nếu out of vocab hoặc negative -> dùng pad
            if xi < 0 or xi >= getattr(tokenizer, "vocab_size", 30000):
                xi = pad_token_id
            cleaned.append(xi)

        # strip trailing pads (nhiều trường hợp generated sequences padded)
        while len(cleaned) > 0 and cleaned[-1] == pad_token_id:
            cleaned.pop()

        # decode an toàn (nếu list rỗng -> trả chuỗi rỗng)
        if len(cleaned) == 0:
            text = ""
        else:
            # tokenizer.decode chấp nhận list[int]
            text = tokenizer.decode(cleaned, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            text = text.strip()
        decoded.append(text)
    return decoded


def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # Nếu output của model là tuple (logits, ...) thì lấy phần đầu
    if isinstance(preds, tuple):
        preds = preds[0]

    # Nếu là logits thì lấy argmax
    if preds.ndim == 3:  # (batch, seq_len, vocab_size)
        preds = np.argmax(preds, axis=-1)

    # Replace -100 bằng pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # ✅ Dùng clean_and_decode để tránh OverflowError
    decoded_preds = clean_and_decode(preds, tokenizer)
    decoded_labels = clean_and_decode(labels, tokenizer)

    # ROUGE
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_result = {k: round(v * 100, 2) for k, v in rouge_result.items()}

    # BLEU
    bleu_result = bleu.compute(predictions=decoded_preds,
                               references=[[l] for l in decoded_labels])
    bleu_score = round(bleu_result["score"], 2)

    # F1 (theo token-level macro)
    y_true = [l.split() for l in decoded_labels]
    y_pred = [p.split() for p in decoded_preds]

    # Flatten lists để tính F1 word-level
    y_true_flat, y_pred_flat = [], []
    for t, p in zip(y_true, y_pred):
        max_len = max(len(t), len(p))
        for i in range(max_len):
            y_true_flat.append(t[i] if i < len(t) else "PAD")
            y_pred_flat.append(p[i] if i < len(p) else "PAD")

    f1 = round(f1_score(y_true_flat, y_pred_flat, average="macro", zero_division=0) * 100, 2)

    return {
        **rouge_result,
        "bleu": bleu_score,
        "f1": f1,
    }






print(">> [5] Training arguments")
training_args = Seq2SeqTrainingArguments(
    output_dir="./vit5-base-contextualize-claim",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    predict_with_generate=True,
    generation_max_length=256,   # default thường = 20, quá ngắn
    fp16=True,
    gradient_accumulation_steps=4,
    label_smoothing_factor=0.1,  # giảm overfitting, giữ chi tiết
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print(">> [6] Start training...")

# Clear CUDA cache
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(">> Cleared CUDA cache before training")

trainer.train()

print(">> [7] Evaluate on test set")
metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
print("   Test metrics:", metrics)

print(">> [8] Save predictions")
predictions, labels, _ = trainer.predict(test_dataset)

# Nếu là logits thì argmax để lấy token ids
if predictions.ndim == 3:
    pred_ids = np.argmax(predictions, axis=-1)
else:
    pred_ids = predictions

# Decode predictions & labels bằng clean_and_decode
decoded_preds = clean_and_decode(pred_ids, tokenizer)
labels_ids = np.where(labels != -100, labels, tokenizer.pad_token_id)
decoded_labels = clean_and_decode(labels_ids, tokenizer)

# Xuất ra CSV
output_df = pd.DataFrame({
    "input_text": test_df["input_text"].tolist(),
    "expected_result": decoded_labels,
    "actual_result": decoded_preds
})
output_path = "./vit5-base-contextualize-claim/test_predictions.csv"
output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"   Predictions saved to '{output_path}'")



print(">> [9] Save model & tokenizer")
trainer.save_model("./vit5-base-contextualize-claim")
tokenizer.save_pretrained("./vit5-base-contextualize-claim")
print(">> DONE")
