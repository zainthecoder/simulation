import torch
import sys

def check_cuda():
    print("\n=== CUDA Availability Check ===")
    print(f"CUDA is available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"\nGPU Information:")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
        print(f"\nCUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    else:
        print("\nCUDA is not available. Please check your GPU and CUDA installation.")
        print("Common issues:")
        print("1. No NVIDIA GPU installed")
        print("2. CUDA drivers not installed")
        print("3. PyTorch installed without CUDA support")
        print("\nTo install PyTorch with CUDA support, visit: https://pytorch.org/get-started/locally/")

if __name__ == "__main__":
    check_cuda() 