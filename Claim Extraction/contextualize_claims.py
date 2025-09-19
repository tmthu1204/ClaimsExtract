# test_decontextualization.py

from models.decontextualization import Decontextualizer

# Khởi tạo model decontextualization
decontext_model = Decontextualizer(device="cpu")

# Ví dụ câu claim mơ hồ
claim = "Bà ấy thường xuyên đi làm trễ"
# Nội dung bài báo để thêm ngữ cảnh
article_content = """
Bà Lan năm nay đã 50 tuổi. Bà ấy thường xuyên đi làm trễ.
"""

# Decontextualize
decontext_claim = decontext_model.contextualize(claim, article_content)

print("Original claim:")
print(claim)
print("\nDecontextualized claim:")
print(decontext_claim)
