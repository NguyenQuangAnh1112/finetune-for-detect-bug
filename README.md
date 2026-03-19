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

**DVC (quản lý pipeline và data)**
- Chạy toàn bộ pipeline theo `dvc.yaml`:
```bash
dvc repro
```
- Đồng bộ dữ liệu lên remote (nếu đã cấu hình):
```bash
dvc push
```
- Kéo dữ liệu từ remote:
```bash
dvc pull
```

**Docker (GPU + CUDA)**
- Yêu cầu: NVIDIA driver + NVIDIA Container Toolkit trên máy host.
- Build image:
```bash
docker build -t bug-detect .
```
- Chạy training với GPU:
```bash
docker run --gpus all --rm -it \
  -v "$PWD:/app" \
  bug-detect \
  uv run python train.py --config configs/config.yaml
```
- Ghi chú: nếu driver quá cũ so với CUDA runtime trong image, container sẽ không dùng được GPU.

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
- `docs/dataset_card.md`
- `docs/model_card.md`

**DVC (tùy chọn)**
- Khởi tạo DVC (nếu chưa): `dvc init`
- Chạy pipeline dữ liệu theo `dvc.yaml`: `dvc repro`
- Track dữ liệu/artifact bằng DVC:
  - `git add dvc.yaml dvc.lock`
  - `git add artifacts/data_ingestion/PyresBugs.tsv.dvc` (nếu bạn dùng `dvc add` thủ công)
- Nếu có remote storage:
  - `dvc remote add -d storage <remote-url>`
  - `dvc push`
