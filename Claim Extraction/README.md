# Claim Extraction for Vietnamese News

## Cấu trúc pipeline
1. Tách câu từ tiêu đề + nội dung (`underthesea.sent_tokenize`).
2. Rule-based loại bỏ câu chứa từ khóa dự đoán/nhận định.
3. PhoBERT sentiment: loại bỏ câu POS/NEG, giữ lại NEU.
4. Xuất file `claims_output.csv`.

## Cách chạy
```bash
pip install -r requirements.txt
python extract_claims.py