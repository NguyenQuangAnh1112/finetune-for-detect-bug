# finetune-for-detect-bug

Dự án fine-tune LLM (Llama/Qwen) bằng QLoRA + SFTTrainer cho bài toán phát hiện lỗi (bug detection). Pipeline gồm data ingestion, data validation, data transformation, training, và evaluation.

**Mục tiêu**
- Chuẩn hóa pipeline huấn luyện LLM dùng QLoRA (4-bit).
- Dễ tái lập, dễ chạy, dễ mở rộng.

**Yêu cầu hệ thống**
- Python >= 3.11
- GPU NVIDIA có CUDA (khuyến nghị VRAM >= 6GB cho 3B model)
- Driver + CUDA phù hợp với PyTorch

**Cài đặt nhanh (uv)**
1. Tạo môi trường ảo bằng uv.
2. Cài dependencies.

Ví dụ:
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

**Chạy huấn luyện**
```bash
uv run python train.py
```

**Chỉ định file cấu hình**
```bash
uv run python train.py --config configs/config.yaml
```

**Cấu hình chính**
- `configs/config.yaml`: đường dẫn data, model name, QLoRA, batch size, learning rate, v.v.

**MLflow**
- Các chỉ số và tham số sẽ được log vào MLflow mặc định (local tracking).
- Xem log trong thư mục `mlruns/`.

**Troubleshooting**
- OOM GPU: giảm `per_device_train_batch_size`, tăng `gradient_accumulation_steps`, bật `use_4bit: true`.
- Lỗi dataset không tìm thấy: chạy data transformation trước, đảm bảo `dataset_path` đúng.

**Cấu trúc thư mục**
- `src/data/`: ingestion, validation, transformation
- `src/training/`: load model, LoRA, trainer runner
- `src/utils/`: logging, exceptions, config
- `tests/`: unit tests

**Tài liệu thêm**
- `docs/code_reading_guide.md`
- `docs/training_bug_fixes.md`
