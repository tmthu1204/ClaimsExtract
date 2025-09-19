# scripts/infer_decontext.py
from models.decontextualization import Decontextualizer

# Init model
decontext_model = Decontextualizer(
    model_name="checkpoints/vit5_decontext_final"
)

claim = "Bà ấy thường xuyên đi làm trễ"
article_content = "Bà Lan năm nay đã 50 tuổi. Bà ấy thường xuyên đi làm trễ."

decontext_claim = decontext_model.contextualize(claim, article_content)

print("Original claim:")
print(claim)
print("\nDecontextualized claim:")
print(decontext_claim)
