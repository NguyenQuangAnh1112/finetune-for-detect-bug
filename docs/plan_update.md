Kế hoạch tổng thể

1. Củng cố tài liệu & entry point
    - Viết README đầy đủ: mục tiêu, yêu cầu phần cứng, cài đặt, quickstart, pipeline, MLflow, troubleshooting.
    - Hoàn thiện train.py hoặc tạo CLI chuẩn (python -m ... / console script).
    - Thêm ví dụ cấu hình tối thiểu trong configs/.

2. Chuẩn hóa môi trường phát triển
    - Bổ sung nhóm dev dependencies trong pyproject.toml cho pytest, ruff, mypy, pre-commit.
    - Thêm cấu hình ruff/mypy cơ bản.
    - Cập nhật .gitignore để loại bỏ mlflow.db/artifacts.

3. Kiểm thử & CI
    - Viết thêm unit tests cho data pipeline và trainer_runner.
    - Thêm 1 integration test nhỏ (dataset mini).
    - Tạo GitHub Actions chạy pytest, ruff, mypy.

4. Quản lý cấu hình & tái lập
    - Thêm validation cấu hình (schema/kiểm tra kiểu/giá trị).
    - Ghi lại seed, config snapshot, và commit hash trong training run.
    - Log thêm artifact cấu hình lên MLflow.

5. Dữ liệu & mô hình
    - Tạo dataset card (nguồn, license, preprocessing).
    - Tạo model card (mục tiêu, eval, hạn chế, cách sử dụng).
    - Nếu dùng DVC: thêm dvc.yaml + hướng dẫn track data.

6. Vận hành/triển khai
    - Tạo Dockerfile + hướng dẫn CUDA/driver.
    - Viết hướng dẫn inference cơ bản (nếu cần).
