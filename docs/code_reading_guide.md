# Hướng dẫn Đọc Code Dự án (Code Reading Guide)

Dự án này là một 파이프라인 (Pipeline) hoàn chỉnh để Fine-tune mô hình Llama 3.1 8B / Qwen 2.5 với QLoRA cho tác vụ phát hiện lỗi (bug detection). Để nắm bắt nhanh nhất toàn bộ luồng hoạt động của hệ thống, bạn nên đọc code theo trình tự từ cấu hình, dữ liệu, cho tới huấn luyện và cuối cùng là các tiện ích hỗ trợ.

Dưới đây là trình tự đọc code được khuyến nghị:

## 1. Cấu hình & Các hằng số (Configuration & Constants)
Mọi dự án Machine Learning chuẩn hóa đều bắt đầu từ cấu hình. Việc hiểu các tham số giúp bạn biết hệ thống có thể tùy biến những gì.
- **`configs/config.yaml`**: Trái tim cấu hình của toàn bộ dự án (đường dẫn data, model name, batch size, learning rate...).
- **`src/entity/config_entity.py`**: Nơi biến các cấu hình từ file `.yaml` thành các object (dataclass) có type hint rõ ràng trong Python để dùng trong source code.
- **`src/constants/__init__.py`**: Chứa các hằng số dùng chung (đường dẫn mặc định tới file config...).

## 2. Tiện ích dùng chung (Utilities)
Đọc lướt qua các file này để hiểu cách dự án xử lý lỗi, log thông tin và nạp cấu hình.
- **`src/utils/config.py`**: Lớp `ConfigurationManager` chịu trách nhiệm đọc `config.yaml` và trả về các cấu hình cho từng giai đoạn (Data Ingestion, Training...).
- **`src/utils/logger.py`**: Cấu hình logging để ghi log ra console và file.
- **`src/utils/exception.py`**: Hệ thống xử lý lỗi tùy chỉnh (Custom Exceptions) và decorator `@he_raise` chuyên dùng để bắt lỗi chuẩn MLOps.

## 3. Data Pipeline (Đường ống dữ liệu)
Đây là nơi dữ liệu thô được tải về, kiểm tra và tiền xử lý thành định dạng cho mô hình LLM. Đọc theo thứ tự:
- **`src/data/data_ingestion.py`**: Tải dữ liệu từ HuggingFace (hoặc nguồn khác) và lưu xuống ổ cứng.
- **`src/data/data_validation.py`**: Kiểm tra toàn vẹn dữ liệu (kiểm tra schema, các cột có đủ không trước khi đem vào xử lý).
- **`src/data/data_transformation.py`**: Chuyển đổi dữ liệu thô sang cấu trúc cho LLM (Tokenizer, áp dụng Chat Template, chia train/test).

## 4. Huấn luyện Mô hình (Model Training)
Đây là phần lõi của dự án thực hiện load mô hình và train với LoRA. Gần đây phần này đã được module hóa để dễ bảo trì. Hãy đọc theo thứ tự sau:
- **`src/training/quantization_config.py`**: Cấu hình 4-bit (QLoRA) bằng BitsAndBytes để giảm VRAM.
- **`src/training/model_loader.py`**: Tải Base Model từ HuggingFace (ví dụ: Qwen/Llama) và thiết lập môi trường (gradient checkpointing...).
- **`src/training/peft_setup.py`**: Gắn adapter LoRA (như r, alpha, target_modules) vào Base Model để tạo mô hình PEFT sẵn sàng cho train.
- **`src/training/trainer_runner.py`**: Chứa cấu hình huấn luyện `TrainingArguments` (epochs, lr) và gọi thư viện `SFTTrainer` (hoặc `Trainer`) để bắt đầu chạy vòng lặp train.
- **`src/training/model_trainer.py`**: Lớp đóng vai trò **Nhạc trưởng (Orchestrator)**. Nó điều phối tất cả các file trong thư mục `src/training` ở trên để thành một luồng chạy hoàn chỉnh.

## 5. Đánh giá Mô hình (Evaluation) - *Tùy chọn*
- **`src/evaluation/`**: Chứa code để lấy mô hình sau khi train ra dự đoán trên tập test và tính toán các chỉ số (nếu có).

## 6. Điểm Neo Chạy Core (Entry Points)
Cuối cùng, hãy xem cách tất cả các thành phần trên được lắp ráp ráp thành một quy trình chạy thực tế (Pipeline):
- **`train_notebook.ipynb`**: File Jupyter Notebook dùng để chạy toàn bộ pipeline từng bước một, rất phù hợp cho quá trình R&D, debug và xem biểu đồ/log trực quan. Trình tự gọi các class cũng thể hiện rõ vòng đời của ứng dụng ở đây.
- *(Nếu có)* **`train.py`** hoặc các lệnh dùng `uv run` ở thư mục gốc: File script chạy thẳng từ terminal cho mục đích chạy tự động (CI/CD) hoặc deploy.

---
💡 **Mẹo:** Trong VS Code / Cursor, bạn hãy bắt đầu mở file `train_notebook.ipynb` (hoặc script chạy tổng) lên trước. Sau đó nhấn `Ctrl` + Click (hoặc `Cmd` + Click) vào các class tĩnh như `DataIngestion`, `ModelTrainer` để nhảy trực tiếp vào code bên trong. Trình tự như vậy giống hệt với thứ tự thực thi của code thật.
