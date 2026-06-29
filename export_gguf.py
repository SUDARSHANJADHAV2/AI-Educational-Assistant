import os
import sys

# MANUALLY ADD COMMON INSTALL PATHS TO ENVIRONMENT PATH
# This fixes the "Failed to install Kitware.CMake" error on Windows
cmake_path = r"C:\Program Files\CMake\bin"
openssl_path = r"C:\Program Files\OpenSSL-Win64\bin"

if cmake_path not in os.environ["PATH"]:
    os.environ["PATH"] = cmake_path + os.pathsep + os.environ["PATH"]
if openssl_path not in os.environ["PATH"]:
    os.environ["PATH"] = openssl_path + os.pathsep + os.environ["PATH"]

# Now we can import unsloth
from unsloth import FastLanguageModel

print("--- DEBUG PATH INFO ---")
print(f"CMake in Path? {any('cmake.exe' in os.listdir(p) for p in os.environ['PATH'].split(os.pathsep) if os.path.exists(p))}")
print("-----------------------")

print("Loading your trained weights from the 'my_custom_model' folder...")

# Load the already merged and saved model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "my_custom_model",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

print("✅ Model loaded successfully!")
print("Starting conversion to GGUF format... (This might take 10-20 minutes)...")

try:
    # Save as a Q4_K_M GGUF format
    model.save_pretrained_gguf("lm_studio_model", tokenizer, quantization_method="q4_k_m")
    print("🎉 DONE! Look for 'lm_studio_model.Q4_K_M.gguf' in your folder.")
except Exception as e:
    print(f"\n❌ GGUF conversion failed again: {e}")
    print("\nIf you see a 'cmake' error above, please try running this in a new terminal:")
    print('set PATH="%PATH%;C:\\Program Files\\CMake\\bin;C:\\Program Files\\OpenSSL-Win64\\bin"')
    print('python export_gguf.py')
