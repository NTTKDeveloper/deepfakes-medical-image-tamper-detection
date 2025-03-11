import pandas as pd
import os

# Đọc file CSV
file_path = "./Data_CT/labels_exp1.csv"  # Thay bằng đường dẫn thực tế
df = pd.read_csv(file_path)

# Thư mục chứa các folder uuid
base_dir = "./Data_CT/"  # Thay bằng đường dẫn thực tế

# Hàm kiểm tra số lượng thư mục tồn tại và không tồn tại
def check_folder_existence(type_name):
    df_filtered = df[df['type'] == type_name]  # Lọc theo type
    uuids = df_filtered['uuid'].astype(str).unique()  # Lấy danh sách uuid duy nhất
    
    existing_folders = [uuid for uuid in uuids if os.path.isdir(os.path.join(base_dir, uuid))]
    non_existing_folders = [uuid for uuid in uuids if not os.path.isdir(os.path.join(base_dir, uuid))]
    
    return len(existing_folders), len(non_existing_folders), existing_folders, non_existing_folders

# Kiểm tra riêng biệt cho FB và FM
fb_exist, fb_not_exist, fb_folders, fb_missing = check_folder_existence('FB')
fm_exist, fm_not_exist, fm_folders, fm_missing = check_folder_existence('FM')

# Kết quả
print(f"🔹 FB:")
print(f"   - Thư mục tồn tại: {fb_exist}")
print(f"   - Thư mục không tồn tại: {fb_not_exist}")

print(f"\n🔹 FM:")
print(f"   - Thư mục tồn tại: {fm_exist}")
print(f"   - Thư mục không tồn tại: {fm_not_exist}")
