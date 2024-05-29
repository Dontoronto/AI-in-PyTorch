import torch

def check_cuda():
    # Überprüfen, ob CUDA verfügbar ist
    is_cuda_available = torch.cuda.is_available()

    if is_cuda_available:
        # CUDA Version und Name der GPU abrufen
        cuda_version = torch.version.cuda
        device_name = torch.cuda.get_device_name(0)
        device_test = torch.cuda.device

        print(f"CUDA is available!")
        print(f"CUDA Version: {cuda_version}")
        print(f"Device Name: {device_name}")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    check_cuda()
