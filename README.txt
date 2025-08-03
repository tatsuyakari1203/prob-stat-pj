# Phân tích tập dữ liệu quảng cáo trên Internet

Dự án này phân tích "Tập dữ liệu quảng cáo trên Internet" từ Kho lưu trữ học máy UCI để xây dựng một mô hình phân loại nhằm dự đoán liệu một hình ảnh có phải là quảng cáo ("ad") hay không ("nonad").

## Mô tả dự án

### Về tập dữ liệu
- **Nhiệm vụ**: Dự đoán liệu một hình ảnh có phải là quảng cáo ("ad") hay không ("nonad")
- **Kích thước**: 1559 cột dữ liệu
- **Cấu trúc**: Mỗi hàng đại diện cho một hình ảnh được gắn nhãn "ad" hoặc "nonad" ở cột cuối
- **Thuộc tính**: Cột 0-1557 đại diện cho các thuộc tính số của hình ảnh
- **Nguồn**: UCI Machine Learning Repository

### Yêu cầu dự án
- Sử dụng các phương pháp thống kê chưa được học trong lớp để phân tích tập dữ liệu
- Mô tả và làm sạch dữ liệu
- Thống kê mô tả: histogram, boxplot, scatter plot, mean, median,...
- Xác định mục tiêu và phương pháp thống kê để đạt được
- Phân tích dữ liệu với mã R và giải thích kết quả

## Cấu trúc thư mục

- `/code`: Chứa tất cả các tập lệnh R để phân tích dữ liệu, từ tải và làm sạch đến xây dựng và so sánh mô hình
  - `01_data_loading.R`: Tải dữ liệu
  - `02_data_cleaning.R`: Làm sạch dữ liệu
  - `03_eda.R`: Phân tích khám phá dữ liệu
  - `04_knn_model.R`: Mô hình K-Nearest Neighbors
  - `05_decision_tree.R`: Mô hình cây quyết định
  - `06_random_forest.R`: Mô hình Random Forest
  - `07_model_comparison.R`: So sánh các mô hình
  - `08_final_summary.R`: Tổng kết cuối cùng
  - `09_plot_summary.R`: Tạo biểu đồ tổng kết
  - `*.csv`: Các file dữ liệu được tạo ra từ quá trình phân tích
  - `*_log.txt`: Log files cho từng bước phân tích
- `/graphics`: Chứa tất cả các biểu đồ và hình ảnh được tạo ra trong quá trình phân tích
- `/chapters`: Chứa các tệp LaTeX cho các chương của báo cáo
  - `introduction.tex`: Giới thiệu
  - `data_description.tex`: Mô tả dữ liệu
  - `descriptive_statistics.tex`: Thống kê mô tả
  - `objective.tex`: Mục tiêu và phương pháp
- `/refs`: Chứa tệp BibTeX cho các tài liệu tham khảo
- `report.tex`: Tệp LaTeX chính để biên dịch báo cáo
- `codespace.sty`: Style file cho LaTeX
- `hcmut-report.cls`: Class file cho báo cáo HCMUT

## Cài đặt môi trường

### Cài đặt R trên Linux/WSL

#### Ubuntu/Debian:
```bash
# Cập nhật package list
sudo apt update

# Cài đặt R
sudo apt install r-base r-base-dev

# Cài đặt các dependencies cho các package R
sudo apt install libcurl4-openssl-dev libssl-dev libxml2-dev
sudo apt install libcairo2-dev libxt-dev
```

#### CentOS/RHEL/Fedora:
```bash
# Fedora
sudo dnf install R R-devel

# CentOS/RHEL (cần EPEL repository)
sudo yum install epel-release
sudo yum install R R-devel
```

### Cài đặt TeXLive trên Linux/WSL

#### Ubuntu/Debian:
```bash
# Cài đặt TeXLive full
sudo apt install texlive-full

# Hoặc cài đặt minimal và thêm packages cần thiết
sudo apt install texlive-latex-base texlive-latex-recommended
sudo apt install texlive-latex-extra texlive-fonts-recommended
sudo apt install texlive-bibtex-extra biber
```

#### CentOS/RHEL/Fedora:
```bash
# Fedora
sudo dnf install texlive-scheme-full

# CentOS/RHEL
sudo yum install texlive texlive-latex
```

### Kiểm tra cài đặt
```bash
# Kiểm tra R
R --version

# Kiểm tra LaTeX
pdflatex --version
biber --version
```

## Cách chạy dự án

### 1. Cài đặt các gói R cần thiết
```bash
Rscript code/install_packages.R
```

### 2. Chạy quy trình phân tích
```bash
# Chạy tất cả scripts theo thứ tự
bash code/run_all_scripts.sh

# Hoặc chạy từng script riêng lẻ
Rscript code/01_data_loading.R
Rscript code/02_data_cleaning.R
Rscript code/03_eda.R
Rscript code/04_knn_model.R
Rscript code/05_decision_tree.R
Rscript code/06_random_forest.R
Rscript code/07_model_comparison.R
Rscript code/08_final_summary.R
Rscript code/09_plot_summary.R
```

### 3. Biên dịch báo cáo LaTeX
```bash
# Biên dịch báo cáo
pdflatex report.tex
biber report
pdflatex report.tex
pdflatex report.tex
```

## Kết quả

Sau khi chạy xong, bạn sẽ có:
- File `report.pdf`: Báo cáo hoàn chỉnh
- Các file CSV trong `/code`: Kết quả phân tích dữ liệu
- Các file log: Chi tiết quá trình thực thi
- Các biểu đồ trong `/graphics`: Hình ảnh minh họa

## Tài liệu tham khảo

Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

```bibtex
@misc{Lichman:2013,
author = "M. Lichman",
year = "2013",
title = "{UCI} Machine Learning Repository",
url = "http://archive.ics.uci.edu/ml",
institution = "University of California, Irvine, School of Information and Computer Sciences"
}
```
