

## 📦 Pre-trained Model

Dự án sử dụng **mô hình ResNet* cho bài toán phân loại **102 loài hoa**.

👉 **Tải model tại đây:**
[https://drive.usercontent.google.com/download?id=1rasCzO_wqgdlkMwYPcHCSMSUEUjU2Ywo&export=download&authuser=0](https://drive.usercontent.google.com/download?id=1rasCzO_wqgdlkMwYPcHCSMSUEUjU2Ywo&export=download&authuser=0)

### 📁 Cách sử dụng model

* Sau khi tải về, **đặt file model cùng cấp với `app.py`**
* Không cần chỉnh sửa thêm đường dẫn, chỉ cần chạy là dùng được

Ví dụ cấu trúc thư mục:

```
project/
│── app.py
│── model_resnet.h5   # (hoặc tên file model bạn tải về)
│── requirements.txt
│── ...
```

### ⚙️ Cài đặt thư viện

Trước khi chạy chương trình, hãy đảm bảo bạn đã cài đầy đủ các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### ▶️ Chạy ứng dụng

Sau khi hoàn tất các bước trên, chạy:

```bash
python app.py
```

---
