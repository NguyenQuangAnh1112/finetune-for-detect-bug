# Dataset Card: PyResBugs (Processed)

## Tóm tắt
Dataset phục vụ bài toán phát hiện lỗi (bug detection) trên mã Python. Dữ liệu gốc được lấy từ Hugging Face (OSS-forge/PyResBugs), sau đó được chuẩn hóa thành định dạng prompt theo chat template để fine-tune LLM.

## Nguồn dữ liệu
- Hugging Face: `OSS-forge/PyResBugs`
- URL: https://huggingface.co/datasets/OSS-forge/PyResBugs
- Split sử dụng: `train` (toàn bộ dữ liệu gốc nằm ở split này)

## License
- `cc-by-sa-4.0` (theo dataset card trên Hugging Face)

## Mục đích sử dụng (Intended Use)
Dataset phục vụ huấn luyện và đánh giá mô hình phát hiện lỗi trong mã Python. Không dùng để suy luận bảo mật hoặc đưa ra quyết định tự động trong sản phẩm sản xuất mà không có kiểm định bổ sung.

## Không phù hợp (Out-of-Scope)
Không phù hợp cho ngôn ngữ lập trình khác Python hoặc các tác vụ tạo mã ngoài phạm vi phát hiện lỗi.

## Cấu trúc dữ liệu
### Trường gốc quan trọng
Pipeline hiện tại sử dụng các trường sau:
- `Faulty Code`
- `Fault Free Code`
- `Bug_Type`
- `High-Level Description`

Các trường khác của dataset gốc vẫn được giữ trong file TSV tải về, nhưng không dùng trong bước tạo prompt.

## Tiền xử lý
Các bước được thực hiện bởi pipeline trong `src/data/`:
1. **Tải dữ liệu** từ Hugging Face và lưu về file TSV tại `artifacts/data_ingestion/PyresBugs.tsv`.
2. **Validation**: kiểm tra file TSV tồn tại.
3. **Lọc dữ liệu**: loại bỏ các dòng thiếu một trong các trường bắt buộc (`Faulty Code`, `Fault Free Code`, `Bug_Type`, `High-Level Description`).
4. **Tạo prompt**: ghép thành hội thoại gồm system/user/assistant. System prompt cố định mô tả nhiệm vụ phát hiện lỗi.
5. **Tokenization**: dùng `AutoTokenizer` theo `tokenizer_name` trong `configs/config.yaml`.
6. **Cắt và padding**: `max_length` (mặc định 1024), `padding="max_length"`.
7. **Chia tập**: tách `train/validation` theo `test_size` (mặc định 0.1).
8. **Lưu dữ liệu**: `DatasetDict` được lưu tại `artifacts/data_transformation/`.

## Thiên lệch & Rủi ro
Chưa có phân tích định lượng về bias. Dữ liệu có thể thiên lệch theo kiểu lỗi, phong cách code, hoặc domain của dự án gốc. Kết quả mô hình cần được review thủ công.

## Quyền riêng tư
Dataset chứa mã nguồn công khai. Tuy nhiên, không đảm bảo loại bỏ mọi thông tin nhạy cảm; cần thận trọng khi dùng trong môi trường sản xuất.

## Bảo trì & Phiên bản
Phiên bản dữ liệu phụ thuộc vào nguồn Hugging Face. Nếu nguồn cập nhật, kết quả huấn luyện có thể thay đổi. Khuyến nghị pin phiên bản dataset qua DVC hoặc snapshot.

## Ghi chú cấu hình
Các tham số tiền xử lý và đường dẫn nằm trong `configs/config.yaml`.
