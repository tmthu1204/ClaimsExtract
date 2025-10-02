import pandas as pd
import sys, os

import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, BASE_DIR)

print("CURRENT_DIR:", CURRENT_DIR)
print("BASE_DIR:", BASE_DIR)
print("sys.path:")
for p in sys.path:
    print(" -", p)

print("Folders in BASE_DIR:", os.listdir(BASE_DIR))





from models.sentiment import SentimentModel
from utils.preprocessing import split_sentences, rule_based_filter


# Load sentiment model
sentiment_model = SentimentModel()

# Đọc dữ liệu
# import os
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Claim Extraction
input_csv = os.path.join(BASE_DIR, "Test", "data", "test_claims.csv")

df = pd.read_csv(input_csv)
print(f"Processing {len(df)} articles...")

claims = []

# Tạo folder 'data' nếu chưa có
os.makedirs("data", exist_ok=True)

for idx, row in df.iterrows():
    article_id = row["ID"]

    # Kiểm tra title NaN
    title = row.get("Tiêu đề")
    if pd.isna(title):
        title = ""
    else:
        title = str(title)

    content = str(row.get("Nội dung", ""))

    # Nếu title rỗng thì chỉ dùng content
    text = title + " " + content if title else content

    print(f"\n📰 Processing article {idx+1}/{len(df)} | ID: {article_id} | Title: {title}")

    sentences = split_sentences(text)

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        # Rule-based filter
        if not rule_based_filter(s):
            print(f"[Filtered by rule] {s}")
            continue

        # Sentiment filter: loại câu POS/NEG cảm xúc
        sentiment = sentiment_model.predict(s)
        if sentiment in ["POS", "NEG"]:
            print(f"[Filtered by sentiment {sentiment}] {s}")
            continue

        # Nếu qua tất cả filter -> giữ lại
        print(f"[Kept] {s}")
        claims.append({
            "ID": article_id,
            "claim": s,
            "article": text   # thêm article gốc để model contextualize
        })

# Xuất ra CSV: ID, claim, article
out_df = pd.DataFrame(claims)


out_csv   = os.path.join(BASE_DIR, "Test", "data", "contextualized_claims.csv")

out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\n✅ Done! Claims saved to {out_csv}")
print(f"Total claims extracted: {len(claims)}")
