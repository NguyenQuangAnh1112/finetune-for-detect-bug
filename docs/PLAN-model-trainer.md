# PLAN - Model Trainer (Fine-Tuning)

## Bối cảnh Pipeline

```
Data Ingestion → Data Validation → Data Transformation → [Model Trainer] → Evaluation
```

**Input:** `artifacts/data_transformation/` (DatasetDict đã tokenize)
**Output:** `artifacts/training/` (`adapter_model.safetensors`, `adapter_config.json`)

---

## Các thay đổi sẽ thực hiện

### 1. `configs/config.yaml`
Thêm block `training:` (LoRA r/alpha, epochs, batch_size, learning_rate, output dir, logging_steps).

### 2. `src/entity/config_entity.py`
Thêm dataclass `TrainingConfig`.

### 3. `src/utils/config.py`
Thêm method `get_training_config()`.

### 4. `src/training/model_trainer.py` *(File chính)*

Class `ModelTrainer` với **toàn bộ method dùng `@he_raise` + `logger`:**

| Method | Log | Bắt lỗi |
|--------|-----|---------|
| `__init__` | "Initializing ModelTrainer..." / "Model loaded" | `ModelError` nếu load thất bại |
| `setup_peft_model` | "Applying LoRA..." / "Trainable params: X/Y" | `ValueError` nếu sai target_modules |
| `train` | Log kích thước split, loss mỗi N steps | `torch.cuda.OutOfMemoryError` → in hướng dẫn giảm batch_size → raise `ModelError` |
| `save` | "Adapter saved to artifacts/training/" | Bắt lỗi ghi disk |

**Pattern chuẩn theo codebase:**
```python
from src.utils.logger import logger
from src.utils.exception import he_raise, ModelError

@he_raise
def train(self, dataset_dict):
    logger.info(f"Train size: {len(dataset_dict['train'])}, Val size: {len(dataset_dict['validation'])}")
    try:
        # SFTTrainer training
        ...
    except torch.cuda.OutOfMemoryError as e:
        logger.error("CUDA OOM — try reducing batch_size or gradient_accumulation_steps.")
        raise ModelError("OOM during training.") from e
```

### 5. `tests/training/test_model_trainer.py`
Unit tests (mock SFTTrainer, mock LLMProvider) kiểm tra init, LoRA config params, `save()` path.

---

## Thư viện cần thiết
```
peft
trl
```
*(Kiểm tra `pyproject.toml` trước khi cài)*

---

## Thứ tự thực hiện

1. `config.yaml` → `config_entity.py` → `config.py`
2. `model_trainer.py` (với logger + error handling)
3. `test_model_trainer.py` → chạy `uv run pytest`

---

## Lưu ý

- **KHÔNG** load lại Tokenizer (đã tokenize ở Data Transformation).
- Chỉ lưu **Adapter**, không merge vào model gốc.
- Nhất quán với pattern `@he_raise` + `logger` của `data_transformation.py`.
