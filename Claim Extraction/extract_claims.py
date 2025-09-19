import pandas as pd
import os
from models.sentiment import SentimentModel
from utils.preprocessing import split_sentences, rule_based_filter
from models.decontextualization import Decontextualizer

# Load sentiment model
sentiment_model = SentimentModel()

decontext_model = Decontextualizer()


# Đọc dữ liệu
input_csv = r"D:\StudySpace\Tài liệu\UIT Data Challenge\Claim Extraction\data\test_claims.csv"
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
        s = s.strip()  # đảm bảo không còn dấu trắng đầu/đuôi

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

        # Tạo phiên bản có ngữ cảnh nếu cần
        # Bạn có thể kiểm tra nếu claim mơ hồ (ví dụ length < 15 từ) hoặc luôn contextualize
        contextual_claim = decontext_model.contextualize(s, content)

        claims.append({
            "ID": article_id,
            "claim": s,
            "contextual_claim": contextual_claim
        })



out_df = pd.DataFrame(claims)
out_csv = r"D:\StudySpace\Tài liệu\UIT Data Challenge\Claim Extraction\data\claims_output_test.csv"
out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\n✅ Done! Claims saved to {out_csv}")
print(f"Total claims extracted: {len(claims)}")
