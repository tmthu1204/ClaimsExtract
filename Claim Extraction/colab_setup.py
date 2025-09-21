# Google Colab Setup Script
# Chạy cell này đầu tiên trong Colab

# 1. Mount Google Drive (nếu cần)
from google.colab import drive
drive.mount('/content/drive')

# 2. Install dependencies
!pip install rouge_score absl-py

# 3. Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 4. Create directories
import os
os.makedirs('/content/data', exist_ok=True)
os.makedirs('/content/logs', exist_ok=True)

print("Setup completed!")
