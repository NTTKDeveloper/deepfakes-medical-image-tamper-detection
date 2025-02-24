import pydicom

# Đọc file DICOM
dicom_data = pydicom.dcmread("./Tampered Scans/Experiment 1 - Blind/1003/0.dcm")

# Lấy kích thước ảnh
image_array = dicom_data.pixel_array
print("Kích thước ảnh:", image_array.shape)
