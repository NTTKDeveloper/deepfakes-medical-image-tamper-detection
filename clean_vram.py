import torch
import gc

def free_vram():
    # Xóa cache bộ nhớ GPU của PyTorch
    torch.cuda.empty_cache()
    # Kích hoạt garbage collector của Python để giải phóng bộ nhớ không cần thiết
    gc.collect()

if __name__ == '__main__':
    # Hiển thị thông tin bộ nhớ trước khi dọn dẹp (tùy chọn)
    print("Trước khi dọn dẹp:")
    print(torch.cuda.memory_summary(device=torch.cuda.current_device()))
    
    # Gọi hàm dọn dẹp bộ nhớ GPU
    free_vram()
    
    # Hiển thị thông tin bộ nhớ sau khi dọn dẹp (tùy chọn)
    print("Sau khi dọn dẹp:")
    print(torch.cuda.memory_summary(device=torch.cuda.current_device()))
