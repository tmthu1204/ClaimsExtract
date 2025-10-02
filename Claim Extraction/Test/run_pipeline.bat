@echo off
echo [1] Extract claims từ test_claims.csv
python Test\extract_claims.py

echo [2] Train contextualization model với contextualized_claims.csv
python Test\contextualize_claims.py

echo [3] Hoàn tất pipeline!
pause
