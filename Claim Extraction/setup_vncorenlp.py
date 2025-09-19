import py_vncorenlp

VNCORENLP_DIR = r"C:\Users\LENOVO\Downloads\vncorenlp"

# nếu chưa có thì tải về (nó sẽ tự download jar + models vào VNCORENLP_DIR)
# py_vncorenlp.download_model(save_dir=VNCORENLP_DIR)

# chỉ chạy word segmentation
rdrsegmenter = py_vncorenlp.VnCoreNLP(
    save_dir=VNCORENLP_DIR,
    annotators=["wseg"],
    max_heap_size='-Xmx2g'
)

text = "Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
output = rdrsegmenter.word_segment(text)
print(output)
