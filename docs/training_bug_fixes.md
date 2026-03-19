# Tài Liệu Sửa Lỗi Quá Trình Huấn Luyện (Training Bug Fixes)

Tài liệu này ghi lại các lỗi đã gặp phải trong quá trình chạy script huấn luyện (`src/training/model_trainer.py`) và các giải pháp đã được áp dụng để khắc phục.

## 1. Lỗi TypeError với Learning Rate

**Mô tả lỗi:**
Trong quá trình khởi tạo `SFTTrainer`, chương trình bị dừng lại với lỗi:
```python
TypeError: '<=' not supported between instances of 'float' and 'str'
```

**Nguyên nhân:**
Giá trị `learning_rate` định dạng khoa học (`2e-4`) trong file `configs/config.yaml` khi được đọc lên bằng thư viện `yaml` đôi khi bị hiểu nhầm thành chuỗi (string) thay vì số thực (float), dẫn đến việc so sánh toán học bị lỗi bên trong thư viện Optimizer của PyTorch.

**Cách khắc phục:**
Ép kiểu (cast) giá trị cấu hình `learning_rate` sang `float` một cách rõ ràng khi load cấu hình.
*File thay đổi: `src/utils/config.py`*
```python
learning_rate=float(config.get("learning_rate", 2e-4)),
```

## 2. Lỗi Hết Bộ Nhớ GPU (CUDA Out Of Memory) khi Huấn Luyện

**Mô tả lỗi:**
Chương trình báo lỗi không đủ bộ nhớ card đồ họa (VRAM) khi bắt đầu chạy đào tạo (Training Loop).
```python
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 594.00 MiB.
```

**Nguyên nhân:**
Card RTX 3050 có 6GB VRAM, hoàn toàn thiếu để load trực tiếp mảng trọng số (weights) của một mô hình 3 Tỷ tham số ở độ chính xác mặc định (16-bit/32-bit floating point). 

**Cách khắc phục:**
Kích hoạt tính năng Quantization 4-bit (QLoRA) cho cấu hình model.
*File thay đổi: `configs/config.yaml`*
```yaml
model:
  use_4bit: true
```

## 3. Lỗi Hết Bộ Nhớ GPU ngay lúc Load Model (Mặc dù đã bật 4-bit)

**Mô tả lỗi:**
Mặc dù đã chuyển sang dùng config 4-bit, lỗi OOM (Out Of Memory) lại ngay lập tức xuất hiện trong quá trình thư viện Hugging Face đẩy (load) các layer vào GPU.

**Nguyên nhân 1: Quá trình Un-quantized trung gian**
Khi load từng layer, hệ thống đôi khi tự ngầm hiểu layer đó là float32 tốn rất nhiều RAM, dẫn đến tràn VRAM trước khi kịp lượng tử hóa (quantize) thành 4-bit.
*Cách khắc phục:* Chỉnh sửa hàm load model luôn thiết lập `torch_dtype=torch.bfloat16`.
*File thay đổi: `src/training/model_loader.py`*
```python
torch_dtype=torch.bfloat16,
```

**Nguyên nhân 2: Phân mảnh bộ nhớ (Memory Fragmentation)**
Cơ chế cấp phát bộ nhớ mặc định của PyTorch chia nhỏ không gian VRAM, gây lãng phí và thất thoát (Fragmentation). Trình cấp phát không tìm được một dải bộ nhớ liên tục đủ lớn để nạp model.
*Cách khắc phục:* Điều chỉnh cấu hình cấp phát động của PyTorch để loại bỏ hiện tượng phân mảnh bằng một biến môi trường hệ thống.
*File thay đổi: `src/training/model_trainer.py`* (Chèn ngay dòng đầu tiên của file, TRƯỚC khi import PyTorch)
```python
import os

# Ngăn chặn PyTorch phân mảnh memory trước khi khởi tạo thư viện
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

## Tổng Kết

Các đoạn code ở trên kết hợp lại đã giúp fix triệt để mọi trở ngại, cho phép tiến trình Load Dataset, Load Model 4-bit bằng QLoRA, và chạy Training SFT diễn ra mượt mà và tiêu tốn chưa đến 5.5GB / 6GB VRAM của card RTX 3050.
