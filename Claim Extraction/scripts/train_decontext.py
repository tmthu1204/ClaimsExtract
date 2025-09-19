# # scripts/train_decontext.py
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
# from utils.dataset import DecontextDataset

# # Config
# MODEL_NAME = "VietAI/vit5-large-vietnews-summarization"
# TRAIN_CSV = "data/train.csv"
# VALID_CSV = "data/valid.csv"
# BATCH_SIZE = 2
# EPOCHS = 3

# # Load tokenizer & model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# # Load data
# train_df = pd.read_csv(TRAIN_CSV)
# valid_df = pd.read_csv(VALID_CSV)

# train_data = train_df.to_dict(orient="records")
# valid_data = valid_df.to_dict(orient="records")

# train_dataset = DecontextDataset(train_data, tokenizer)
# valid_dataset = DecontextDataset(valid_data, tokenizer)

# training_args = Seq2SeqTrainingArguments(
#     output_dir="./checkpoints",
#     do_eval=True,   # bật evaluation
#     learning_rate=5e-5,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     save_total_limit=2,
#     predict_with_generate=True,
#     logging_dir="./logs",
#     logging_steps=10,
#     eval_strategy="steps",  # nếu version >=4.30
#     save_strategy="steps",
#     use_cpu=True,         # thay thế nếu có
# )


# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=valid_dataset,
#     processing_class=tokenizer
# )

# print("Num train samples:", len(train_dataset))
# print("Num valid samples:", len(valid_dataset))

# trainer.train()
# trainer.save_model("checkpoints/vit5_decontext_final")

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    pipeline
)

import transformers

print("Transformers version:", transformers.__version__)
print("Seq2SeqTrainingArguments class:", transformers.Seq2SeqTrainingArguments)


# ======================
# 1. Load dataset
# ======================
data_path = "data/train.csv"  # <-- đổi path này cho đúng
df = pd.read_csv(data_path)

# Chuẩn bị input-output format
df["input_text"] = "claim: " + df["claim"] + " article: " + df["article"]
df["target_text"] = df["decontext_claim"]

dataset = Dataset.from_pandas(df[["input_text", "target_text"]])
dataset = dataset.train_test_split(test_size=0.1)

# ======================
# 2. Tokenizer
# ======================
model_name = "VietAI/vit5-base-vietnews-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(batch):
    model_inputs = tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    labels = tokenizer(
        batch["target_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# ======================
# 3. Load model
# ======================
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ======================
# 4. Training arguments
# ======================
training_args = Seq2SeqTrainingArguments(
    output_dir="./vit5-decontext",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    logging_dir='./logs',
    logging_steps=50,
    report_to="none",
    fp16=True,
    gradient_checkpointing=True,
)


# ======================
# 5. Trainer
# ======================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

# ======================
# 6. Train
# ======================
trainer.train()

# ======================
# 7. Save model
# ======================
trainer.save_model("./vit5-decontext")
tokenizer.save_pretrained("./vit5-decontext")

# ======================
# 8. Test inference
# ======================
pipe = pipeline("text2text-generation", model="./vit5-decontext", tokenizer=tokenizer)

test_text = "claim: Bà ấy thường xuyên đi làm trễ article: Bà Lan năm nay đã 50 tuổi. Bà ấy thường xuyên đi làm trễ."
result = pipe(test_text, max_length=128)[0]["generated_text"]

print("\n=== TEST INFERENCE ===")
print("Input:", test_text)
print("Output:", result)
