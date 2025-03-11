import os
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler  # Dùng để chuẩn hóa đặc trưng và nhãn

# ---------------------------
# 1. Cài đặt ResNet50 làm feature extractor
# ---------------------------
resnet50 = models.resnet50(pretrained=True)
# Loại bỏ lớp fully-connected cuối cùng để lấy đặc trưng sau global pooling (2048 chiều)
feature_extractor = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
feature_extractor.eval()

# Freeze các tham số của mô hình
for param in feature_extractor.parameters():
    param.requires_grad = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor.to(device)

# ---------------------------
# 2. Định nghĩa transform cho tập train và test
# ---------------------------
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ColorJitter(brightness=10, contrast=0.2, saturation=0.2, hue=0.1),  # Chỉ thay đổi pixel, không làm thay đổi tọa độ
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------
# 3. Hàm trích xuất đặc trưng từ file DICOM
# ---------------------------
def extract_features(dcm_path, transform):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array

    # Nếu ảnh grayscale, chuyển thành RGB
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)
    
    img = img.astype(np.uint8)
    tensor_img = transform(img)
    tensor_img = tensor_img.unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(tensor_img)
    features = features.view(features.size(0), -1)  # Flatten thành vector 2048 chiều
    return features.cpu().numpy().flatten()

# ---------------------------
# 4. Chuẩn bị dữ liệu từ file CSV và ảnh DICOM
# ---------------------------
csv_path = './Data_CT/labels.csv'
df = pd.read_csv(csv_path)

# Tách dữ liệu thành tập train và test (80% train, 20% test)
df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
print(f"Số mẫu train ban đầu: {len(df_train)}, Số mẫu test ban đầu: {len(df_test)}")

def process_data(df_subset, transform):
    X, y = [], []
    for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Processing Samples"):
        uuid = str(row['uuid'])
        slice_num = row['slice']
        dcm_path = os.path.join(f"./Data_CT/{uuid}", f"{slice_num}.dcm")
        if not os.path.exists(dcm_path):
            continue  # Loại bỏ mẫu không có file DICOM
        try:
            feat = extract_features(dcm_path, transform)
            X.append(feat)
            y.append([row['x'], row['y']])
        except Exception as e:
            continue
    return np.array(X), np.array(y)

# Xử lý dữ liệu cho tập train và test
X_train, y_train = process_data(df_train, train_transform)
X_test, y_test = process_data(df_test, test_transform)

print(f"Số mẫu train hợp lệ: {len(X_train)} / {len(df_train)}")
print(f"Số mẫu test hợp lệ: {len(X_test)} / {len(df_test)}")

# ---------------------------
# 5. Chuẩn hóa đặc trưng và nhãn đã trích xuất
# ---------------------------
# Chuẩn hóa đặc trưng
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Chuẩn hóa nhãn (y)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# ---------------------------
# 6. Huấn luyện mô hình GPR phi tuyến với MultiOutputRegressor
# ---------------------------
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
regressor = MultiOutputRegressor(gpr)
regressor.fit(X_train_scaled, y_train_scaled)

# ---------------------------
# 7. Đánh giá mô hình
# ---------------------------
y_train_pred_scaled = regressor.predict(X_train_scaled)
y_test_pred_scaled = regressor.predict(X_test_scaled)

# Đảo chuẩn hóa kết quả dự đoán về thang gốc
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"Train MSE: {mse_train:.2f}")
print(f"Test MSE: {mse_test:.2f}")

print("\nMột vài dự đoán trên tập test:")
for i in range(min(5, len(X_test))):
    print(f"Ground truth: {y_test[i]}, Predicted: {y_test_pred[i]}")
