import torch
import torch.nn as nn
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

class CNN2D(nn.Module):
    def __init__(self, input_size=(512, 512), num_classes=2):
        super(CNN2D, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 512 -> 256

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256 -> 128

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 -> 64

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 64 -> 32
        )

        # Kích thước đầu vào cho Fully Connected layers
        fc_input_dim = 32 * 32 * 256  # (32x32x256)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)  # 2 lớp: Ảnh thật (1) & Giả (0)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ==============================
# 1. Load mô hình đã tạo
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN2D().to(device)  # Khởi tạo mô hình
model.eval()  # Đặt mô hình về chế độ đánh giá (không huấn luyện)

# ==============================
# 2. Load ảnh DICOM
# ==============================
dicom_path = "./Tampered Scans/Experiment 1 - Blind/1003/0.dcm"  # Đường dẫn file DICOM
dicom_data = pydicom.dcmread(dicom_path)
image_array = dicom_data.pixel_array  # Lấy mảng pixel từ DICOM

# Chuyển kiểu dữ liệu từ int16 -> float32 và chuẩn hóa về [0,1]
image_array = image_array.astype(np.float32)
image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + 1e-6)  # Tránh chia 0

# Chuẩn bị transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

image_tensor = transform(image_array).unsqueeze(0).to(device)  # Thêm batch dimension

# ==============================
# 3. Dự đoán nhãn
# ==============================
with torch.no_grad():  # Không cần tính toán gradient khi dự đoán
    output = model(image_tensor)
    predicted_label = torch.argmax(output, dim=1).item()  # Lấy nhãn có xác suất cao nhất

# ==============================
# 4. Hiển thị ảnh & nhãn dự đoán
# ==============================
plt.imshow(image_array, cmap="gray")  # Hiển thị ảnh gốc
plt.title(f"Dự đoán: {'Ảnh Thật' if predicted_label == 1 else 'Ảnh Giả'}")
plt.axis("off")
plt.show()
