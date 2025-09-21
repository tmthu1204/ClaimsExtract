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
import os

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

print(">> [1] Load & split dataset")
df = pd.read_csv("data/contextualized_claims.csv")
print("   Số dòng dữ liệu:", len(df))

def build_input(row):
    return f"Contextualize: {row['claim']} Article: {row['article']}"

df["input_text"] = df.apply(build_input, axis=1)
df["target_text"] = df["contextualized_claim"]

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
print("   Train:", len(train_df), "| Valid:", len(valid_df), "| Test:", len(test_df))

train_dataset = Dataset.from_pandas(train_df[["input_text", "target_text"]])
valid_dataset = Dataset.from_pandas(valid_df[["input_text", "target_text"]])
test_dataset  = Dataset.from_pandas(test_df[["input_text", "target_text"]])

print(">> [2] Load tokenizer")
model_name = "VietAI/vit5-large-vietnews-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(batch):
    model_inputs = tokenizer(batch["input_text"], max_length=256, truncation=True)  # Reduced length
    labels = tokenizer(batch["target_text"], max_length=64, truncation=True)       # Reduced length
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("   Preprocessing train...")
train_dataset = train_dataset.map(preprocess, batched=True)
print("   Preprocessing valid...")
valid_dataset = valid_dataset.map(preprocess, batched=True)
print("   Preprocessing test...")
test_dataset  = test_dataset.map(preprocess, batched=True)

print(">> [3] Load model + Data Collator")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

print(">> [4] Load metric (ROUGE)")
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: v.mid.fmeasure * 100 for k, v in result.items()}
    return {k: round(v, 2) for k, v in result.items()}

print(">> [5] Init Trainer")
training_args = Seq2SeqTrainingArguments(
    output_dir="./vit5-contextualize-claim",
    eval_strategy="no",              # No evaluation during training (faster)
    save_strategy="epoch",
    learning_rate=5e-5,             # Slightly higher learning rate
    per_device_train_batch_size=1,   # Very small batch
    per_device_eval_batch_size=1,    # Very small batch
    num_train_epochs=1,              # Only 1 epoch for CPU
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    predict_with_generate=True,
    dataloader_pin_memory=False,     # Reduce memory usage
    gradient_accumulation_steps=8,   # Large accumulation
    max_grad_norm=1.0,              # Gradient clipping
    warmup_steps=10,                # Small warmup
    save_steps=50,                  # Save frequently
    remove_unused_columns=False,    # Keep all columns
    dataloader_num_workers=0        # No multiprocessing
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print(">> [6] Start training... (This will take a LONG time on CPU)")
print("   ⚠️  WARNING: Training on CPU will be very slow!")
print("   ⚠️  Consider using Google Colab or Kaggle instead")
trainer.train()

print(">> [7] Evaluate on test set")
metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
print("   Test metrics:", metrics)

print(">> [8] Save model")
trainer.save_model("./vit5-contextualize-claim")
tokenizer.save_pretrained("./vit5-contextualize-claim")
print(">> DONE")
