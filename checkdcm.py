import pydicom
import pandas as pd
import os

def check_coordinates(csv_path):
    # Đọc file CSV
    df = pd.read_csv(csv_path)
    
    for index, row in df.iterrows():
        uuid = str(row['uuid'])
        slice_num = row['slice']
        slice_path = os.path.join(f'./Data_CT/{uuid}/{slice_num}.dcm')
        x, y = row['x'], row['y']
        
        if not os.path.exists(slice_path):
            print(f"File {slice_path} không tồn tại!")
            continue
        
        # Đọc file DICOM
        ds = pydicom.dcmread(slice_path)
        img = ds.pixel_array
        height, width = img.shape[:2]  # Lấy kích thước ảnh
        
        # Kiểm tra tọa độ
        if 0 <= x < width and 0 <= y < height:
            print(f"Dòng {index}: Hợp lệ (width={width}, height={height})")
        else:
            print(f"Dòng {index}: Không hợp lệ (x={x}, y={y}, width={width}, height={height})")

# Gọi hàm kiểm tra với file CSV (thay thế 'data.csv' bằng đường dẫn thực tế)
check_coordinates('./Data_CT/labels.csv')