import os
import re
from underthesea import sent_tokenize
import py_vncorenlp

# Đường dẫn tới thư mục chứa VnCoreNLP-1.2.jar và models
VNCORENLP_DIR = r"C:\Users\LENOVO\Downloads\vncorenlp"

# Khởi tạo VnCoreNLP với annotators đầy đủ
model = py_vncorenlp.VnCoreNLP(
    annotators=["wseg", "pos", "ner", "parse"],
    save_dir=VNCORENLP_DIR,
    max_heap_size='-Xmx2g'
)

# Danh sách viết tắt phổ biến
ABBREVIATIONS = ["PGS.", "GS.", "TS.", "ThS.", "TP.", "Ô.", "Bà.", "Ông."]

def protect_abbreviations(text: str):
    for abbr in ABBREVIATIONS:
        text = text.replace(abbr, abbr.replace(".", "<dot>"))
    return text

def restore_abbreviations(text: str):
    return text.replace("<dot>", ".")


def remove_special_chars(text: str) -> str:
    """
    Loại bỏ các ký tự đặc biệt, icon, emoji,... chỉ giữ chữ, số, dấu câu cơ bản.
    """
    # Giữ: chữ, số, dấu câu cơ bản (.,!?;:) và khoảng trắng
    text = re.sub(r"[^a-zA-ZÀ-ỹ0-9\s.,!?;:]", "", text)
    # Loại nhiều khoảng trắng liên tiếp
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_sentences(text: str):
    """
    Tách câu bằng underthesea, loại bỏ viết tắt, khoảng trắng, ký tự đặc biệt.
    """
    text = remove_special_chars(text)
    text = protect_abbreviations(text)
    sentences = sent_tokenize(text)
    sentences = [restore_abbreviations(s).strip() for s in sentences if s.strip()]
    return sentences



# def has_subject_predicate(sentence: str) -> bool:
#     """
#     Kiểm tra câu có chủ ngữ + vị ngữ (dựa trên POS tag) - phiên bản py_vncorenlp mới
#     """
#     try:
#         annotation = model.annotate_text(sentence)
        
#         if not isinstance(annotation, list):
#             return False  # annotation lỗi, coi như không có chủ-vị
        
#         tokens = []
#         for sent in annotation:
#             if isinstance(sent, list):
#                 for word in sent:
#                     if isinstance(word, dict) and "pos" in word:
#                         tokens.append(word)
        
#         # Nếu không có token nào, coi như không hợp lệ
#         if not tokens:
#             return False
        
#         has_noun = any(word["pos"].startswith("N") or word["pos"] == "P" for word in tokens)
#         has_verb = any(word["pos"].startswith("V") for word in tokens)
#         return has_noun and has_verb
    
#     except Exception as e:
#         print(f"[Warning] Error in has_subject_predicate: {e}")
#         return False
def has_subject_predicate(sentence: str) -> bool:
    """
    Kiểm tra câu có chủ-vị (noun + verb) với py_vncorenlp 2.x trả về dict
    """
    try:
        annotation = model.annotate_text(sentence)

        if not isinstance(annotation, dict):
            print(f"[Warning] Unexpected annotate_text output: {type(annotation)}")
            return False

        # Lấy danh sách token cho câu đầu tiên (key=0)
        sent_tokens = annotation.get(0, [])
        if not sent_tokens:
            print(f"[Debug] No tokens for sentence: {sentence}")
            return False

        # Kiểm tra noun + verb
        has_noun = any(tok["posTag"].startswith("N") or tok["posTag"] == "P" for tok in sent_tokens)
        has_verb = any(tok["posTag"].startswith("V") for tok in sent_tokens)

        # Debug POS
        print(f"[Debug] Tokens POS: {[tok['posTag'] for tok in sent_tokens]} -> has_noun: {has_noun}, has_verb: {has_verb}")

        return has_noun and has_verb

    except Exception as e:
        print(f"[Warning] Error in has_subject_predicate: {e}")
        return False







def rule_based_filter(sentence: str) -> bool:
    """
    Trả về True nếu câu nên giữ lại (claim)
    """
    s = sentence.strip()
    
    if not s or len(s.split()) < 4:  # quá ngắn
        return False
    if s.endswith("?") or s.endswith("!"):  # loại câu hỏi / cảm thán
        return False
    if any(s.lower().startswith(k) for k in ["nếu", "giả sử", "ước gì", "phải chi"]):
        return False
    if any(k in s.lower() for k in ["dự đoán", "nhận định", "cho rằng", "nghĩ rằng", "có thể", "khả năng", "hy vọng", "lo ngại", "ước tính", "dự báo", "ước lượng", "kỳ vọng", "tin rằng", "theo chuyên gia", "cho biết rằng"]):
        return False
    
    # Kiểm tra chủ-vị
    if not has_subject_predicate(s):
        return False

    return True



AMBIGUOUS_PRONOUNS = [
    "bà ấy", "ông ấy", "cô ấy", "anh ấy", "họ", "chị ấy", "ông", "bà"
]

def clean_text(text: str) -> str:
    """Loại bỏ khoảng trắng thừa, ký tự lạ"""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def detect_ambiguous_pronouns(sentence: str):
    """Trả về danh sách các pronouns chưa rõ ràng"""
    detected = []
    for pronoun in AMBIGUOUS_PRONOUNS:
        if pronoun in sentence:
            detected.append(pronoun)
    return detected
