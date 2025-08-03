# Phân tích tập dữ liệu quảng cáo trên Internet

Dự án này phân tích "Tập dữ liệu quảng cáo trên Internet" từ Kho lưu trữ học máy UCI để xây dựng một mô hình phân loại nhằm dự đoán liệu một hình ảnh có phải là quảng cáo ("ad") hay không ("nonad").

## Cấu trúc thư mục

- `/code`: Chứa tất cả các tập lệnh R để phân tích dữ liệu, từ tải và làm sạch đến xây dựng và so sánh mô hình.
- `/graphics`: Chứa tất cả các biểu đồ và hình ảnh được tạo ra trong quá trình phân tích.
- `/chapters`: Chứa các tệp LaTeX cho các chương của báo cáo.
- `/refs`: Chứa tệp BibTeX cho các tài liệu tham khảo.
- `report.tex`: Tệp LaTeX chính để biên dịch báo cáo.

## Cách chạy

1.  **Cài đặt các gói cần thiết:** Chạy tập lệnh `code/install_packages.R` để cài đặt tất cả các thư viện R cần thiết.
2.  **Chạy quy trình phân tích:** Thực thi các tập lệnh R trong thư mục `code` theo thứ tự từ `01` đến `08`.
3.  **Biên dịch báo cáo:** Sử dụng một trình biên dịch LaTeX (ví dụ: pdflatex) để biên dịch `report.tex`.
