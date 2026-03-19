# Model Card: Bug Detection (Qwen2.5-Coder-3B-Instruct + QLoRA)

## Tóm tắt
Mô hình được fine-tune cho nhiệm vụ phát hiện lỗi trong mã Python. Đầu vào là mã lỗi, đầu ra gồm loại lỗi, mô tả, và đoạn mã đã sửa.

## Mục tiêu
- Phát hiện loại lỗi (bug type) trong đoạn mã Python.
- Mô tả ngắn gọn lỗi ở mức khái quát.
- Đề xuất mã sửa (fault-free code).

## Mục đích sử dụng (Intended Use)
Hỗ trợ phân tích lỗi trong mã Python cho mục đích nghiên cứu, học tập hoặc trợ lý lập trình. Không thay thế review của con người trong các tình huống quan trọng.

## Không phù hợp (Out-of-Scope)
Không dùng để ra quyết định bảo mật hoặc tự động sửa lỗi trong production mà không có kiểm định bổ sung.

## Mô hình gốc và kỹ thuật huấn luyện
- Base model: `Qwen/Qwen2.5-Coder-3B-Instruct`
- Fine-tuning: QLoRA 4-bit với LoRA adapter
- Trainer: `trl.SFTTrainer`
- Prompt: chat template với system/user/assistant, trong đó system prompt mô tả nhiệm vụ bug detection.

## Dữ liệu huấn luyện
- Dataset gốc: `OSS-forge/PyResBugs` (Hugging Face)
- Dataset sau xử lý: lưu tại `artifacts/data_transformation/`
- Chi tiết dữ liệu: xem `docs/dataset_card.md`

## Đánh giá (Evaluation)
- Split: train/validation (tách trong bước data transformation)
- Theo dõi: eval loss theo `eval_steps` trong `TrainingArguments`
- Ghi log: MLflow (local tracking)
- Hiện chưa có đánh giá thủ công hoặc benchmark ngoài loss.

## Metrics
Hiện chưa có metric ngoài loss (ví dụ: accuracy, exact match, pass@k). Cần bổ sung benchmark tiêu chuẩn nếu dùng cho so sánh mô hình.

## Hạn chế
- Dữ liệu chỉ tập trung vào Python; kết quả cho ngôn ngữ khác không được đảm bảo.
- Có thể tạo ra mã sửa không chính xác hoặc không an toàn; cần review thủ công.
- Không phù hợp cho quyết định bảo mật hoặc production mà không có kiểm định.

## Thiên lệch & Rủi ro
Chưa có phân tích định lượng về bias. Kết quả có thể thiên lệch theo kiểu lỗi hoặc phong cách code trong dataset.

## Cách sử dụng
### Huấn luyện
```bash
uv run python train.py
```

### Chỉ định cấu hình
```bash
uv run python train.py --config configs/config.yaml
```

### Inference
Pipeline inference chưa được tích hợp sẵn trong repo. Bạn có thể sử dụng adapter/weights từ thư mục output và nạp bằng Hugging Face Transformers/PEFT theo nhu cầu.

## Cấu hình quan trọng
Các tham số về model và training nằm trong `configs/config.yaml`.
