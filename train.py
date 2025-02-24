import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pydicom
from torchvision import transforms

# 1. Định nghĩa Dataset để đọc file DICOM và nhãn từ file CSV
class CTDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        """
        Args:
            data_dir (str): Thư mục chứa file DICOM.
            csv_file (str): Đường dẫn tới file CSV chứa tên file và nhãn (0: real, 1: fake).
            transform (callable, optional): Các transform áp dụng cho ảnh.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        csv_path = os.path.join(data_dir, csv_file)
        with open(csv_path, "r") as f:
            # Giả sử mỗi dòng trong CSV có định dạng: "filename,label"
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    filename, label = parts
                    self.samples.append((filename, int(label)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        dicom_path = os.path.join(self.data_dir, filename)
        # Đọc file DICOM
        dicom_data = pydicom.dcmread(dicom_path)
        # Lấy mảng pixel và chuyển về kiểu float32
        image = dicom_data.pixel_array.astype(np.float32)
        # Chuẩn hóa ảnh về khoảng [0, 1]
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-6)
        # Thêm kênh (vì ảnh grayscale)
        image = np.expand_dims(image, axis=0)  # kết quả shape: (1, H, W)
        
        # Chuyển ảnh về dạng tensor (nếu chưa)
        image = torch.tensor(image, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# 2. Xây dựng mô hình CNN đơn giản (2D CNN)
class SimpleCNN(nn.Module):
    def __init__(self, image_size=512):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Input: 1 kênh, Output: 16 kênh
            nn.ReLU(),
            nn.MaxPool2d(2),  # Giảm kích thước xuống một nửa
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Tính kích thước feature map sau 3 lần pooling:
        # Nếu ảnh đầu vào có kích thước image_size x image_size, sau 3 lần pooling sẽ có kích thước (image_size // 8)
        fc_input_dim = 64 * (image_size // 8) * (image_size // 8)
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 2 lớp: real và fake
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# 3. Hàm huấn luyện mô hình
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()  # Chế độ huấn luyện
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()  # Reset gradient
            outputs = model(inputs)  # Dự đoán
            loss = criterion(outputs, labels)  # Tính loss
            loss.backward()  # Lan truyền ngược
            optimizer.step()  # Cập nhật tham số
            
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# 4. Hàm main để chạy quá trình huấn luyện
def main():
    # Cấu hình thiết bị: sử dụng GPU nếu có
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Đường dẫn đến dữ liệu
    data_dir = "path_to_your_dataset"  # Thay đổi đường dẫn đến thư mục dữ liệu của bạn
    csv_file = "labels.csv"  # Tên file CSV chứa thông tin nhãn
    
    # Các transform (nếu cần, ở đây đã chuyển về tensor trong Dataset)
    transform = None
    
    # Tạo dataset và DataLoader
    dataset = CTDataset(data_dir, csv_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    
    # Khởi tạo mô hình, loss function và optimizer
    model = SimpleCNN(image_size=512).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Huấn luyện mô hình
    num_epochs = 10  # Thay đổi số epoch theo nhu cầu
    train_model(model, dataloader, criterion, optimizer, device, num_epochs=num_epochs)
    
    # Lưu mô hình sau khi huấn luyện (nếu cần)
    torch.save(model.state_dict(), "simple_cnn_ct.pth")
    print("Huấn luyện hoàn tất và mô hình đã được lưu.")

if __name__ == "__main__":
    main()
