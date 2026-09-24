# Bài Tập 05: Mạng Tích Chập (CNN) & Các Mô Hình Phát Triển (VGG, ResNet, CBAM Attention)

> **HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG (PTIT)**  
> **Khoa Công nghệ Thông tin 1**  
> **Học phần:** Phát triển các Hệ thống Thông minh  
> **Giảng viên hướng dẫn:** PGS.TS. Trần Đình Quế  
> **Sinh viên thực hiện:** Nguyễn Nam Hải  
> **Mã sinh viên:** B23DCCN277 — **Lớp:** D23CTPM01 — **Nhóm học phần:** CT01  
> **Kho lưu trữ chính thức:** [https://github.com/HandQ2212/intel-sys-assignment-05](https://github.com/HandQ2212/intel-sys-assignment-05)  
> **Báo cáo kỹ thuật hoàn chỉnh:** [`A05_CT_hainn_277.pdf`](A05_CT_hainn_277.pdf) (30 trang chuẩn học thuật)

---

## 📌 1. Nguồn Dữ Liệu Kaggle Chính Thức

Toàn bộ thực nghiệm trong bài tập được xây dựng trên 3 bộ dữ liệu chuẩn mực tải về từ Kaggle:

| # | Tên Tập Dữ Liệu | Kiểu Dữ Liệu & Quy Mô | Đường Dẫn Kaggle Trực Tuyến |
|---|---|---|---|
| 1 | **CIFAR-10** | 60,000 ảnh màu RGB $32 \times 32$, 10 lớp vật thể tự nhiên | [kaggle.com/datasets/ayush1220/cifar10](https://www.kaggle.com/datasets/ayush1220/cifar10) |
| 2 | **MNIST Numbers** | 400,000 ảnh mở rộng, $28 \times 28$ Grayscale, 10 chữ số | [kaggle.com/datasets/alexandrelemercier/400k-augmented-mnist-extended-handwritten-digits](https://www.kaggle.com/datasets/alexandrelemercier/400k-augmented-mnist-extended-handwritten-digits) |
| 3 | **Diabetes CDC** | 324,372 bản ghi, 21 thuộc tính lâm sàng (Phân loại nhị phân) | [kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) |

---

## 🚀 2. Bảng Tổng Hợp Kết Quả Thực Nghiệm Toàn Diện (Master Benchmark)

Dưới đây là kết quả thực nghiệm của 12 mô hình được huấn luyện thực tế trên Apple Silicon GPU Metal (`mps`):

| Tập Dữ Liệu | Kiến Trúc Mô Hình | Tham Số (Params) | Test Acc (%) | Test Loss | Test F1 (Macro/Bin) | Training Time (s) |
|---|---|---|---|---|---|---|
| **CIFAR-10** | M1: Basic CNN | 20,202 | 30.93% | 1.8841 | 0.3015 | 39.43s |
| *(Ảnh màu 10 lớp)* | M2: VGG-Style CNN | 1,248,394 | 28.67% | 1.9568 | 0.2809 | 74.07s |
| | M3: ResNet-Style CNN | 1,862,794 | 32.53% | 1.8596 | 0.3204 | 114.73s |
| | **M4: Attention CNN (CBAM)** | **98,170** | **35.20%** | **1.7828** | **0.3475** | **45.01s** |
|---|---|---|---|---|---|---|
| **MNIST** | M1: Basic CNN | 19,626 | 42.00% | 1.9837 | 0.3857 | 13.92s |
| *(Chữ số viết tay)* | M2: VGG-Style CNN | 1,247,242 | **95.25%** | 0.1558 | **0.9526** | 30.31s |
| | M3: ResNet-Style CNN | 1,861,642 | 94.13% | 0.1983 | 0.9413 | 43.14s |
| | **M4: Attention CNN (CBAM)** | **97,594** | **93.00%** | **0.2312** | **0.9298** | **17.15s** |
|---|---|---|---|---|---|---|
| **Diabetes CDC** | M1: Basic 1D CNN | 6,529 | 79.68% | 0.4430 | 0.6559 | 24.23s |
| *(Bảng 21 thuộc tính)*| M2: VGG-Style 1D CNN | 95,297 | 80.03% | 0.4398 | 0.6653 | 47.93s |
| | M3: ResNet-Style 1D CNN | 148,801 | **80.52%** | 0.4361 | 0.6720 | 66.85s |
| | **M4: Attention 1D CNN** | **9,058** | **80.22%** | **0.4385** | **0.6730** | **26.97s** |

---

## 📂 3. Cấu Trúc Mã Nguồn Dự Án (13 Notebooks)

Toàn bộ hệ thống mã nguồn được tổ chức độc lập theo từng notebook tự chứa (self-contained):

```
intel-sys-assignment-05/
├── A05_CT_hainn_277.tex        # Báo cáo kỹ thuật mã nguồn LaTeX
├── A05_CT_hainn_277.pdf        # Báo cáo kỹ thuật PDF (30 trang chuẩn PTIT)
├── README.md                   # Tài liệu hướng dẫn dự án
├── run_experiments.py          # Script tự động huấn luyện 12 mô hình trên MPS/CUDA
├── comparison.ipynb            # Notebook phân tích, đối chuẩn & vẽ biểu đồ tổng hợp
│
├── cifar10_basic_cnn.ipynb     # CIFAR-10: Basic CNN (20k params)
├── cifar10_vgg_cnn.ipynb       # CIFAR-10: VGG-style CNN (1.25M params)
├── cifar10_resnet_cnn.ipynb    # CIFAR-10: ResNet-style CNN (1.86M params)
├── cifar10_attention_cnn.ipynb # CIFAR-10: Attention CNN CBAM (98k params)
│
├── mnist_basic_cnn.ipynb       # MNIST: Basic CNN (19k params)
├── mnist_vgg_cnn.ipynb         # MNIST: VGG-style CNN (1.25M params)
├── mnist_resnet_cnn.ipynb      # MNIST: ResNet-style CNN (1.86M params)
├── mnist_attention_cnn.ipynb   # MNIST: Attention CNN CBAM (97k params)
│
├── diabetes_basic_cnn.ipynb    # Diabetes: Basic 1D CNN (6.5k params)
├── diabetes_vgg_cnn.ipynb      # Diabetes: VGG-style 1D CNN (95k params)
├── diabetes_resnet_cnn.ipynb   # Diabetes: ResNet-style 1D CNN (148k params)
├── diabetes_attention_cnn.ipynb# Diabetes: Attention 1D CNN (9k params)
│
├── dataset/                    # Liên kết tập dữ liệu (cifar10, mnist_number, diabets.csv)
├── figures/                    # 14 hình ảnh biểu đồ phân tích thực nghiệm
└── results/                    # 12 file kết quả JSON lưu chi tiết metrics theo epoch
```

---

## ⚡ 4. Hướng Dẫn Cài Đặt Nhanh & Tái Lập Kết Quả

### Bước 1: Clone kho mã nguồn từ GitHub
```bash
git clone https://github.com/HandQ2212/intel-sys-assignment-05.git
cd intel-sys-assignment-05
```

### Bước 2: Cài đặt thư viện phụ thuộc
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn
```

### Bước 3: Huấn luyện tự động toàn bộ 12 mô hình
```bash
python run_experiments.py
```
*Script sẽ tự động phát hiện thiết bị tốt nhất (`mps` trên Apple Silicon, `cuda` trên Nvidia GPU, hoặc `cpu`), thực thi huấn luyện và trích xuất toàn bộ biểu đồ vào `figures/` cùng các file kết quả vào `results/`.*

### Bước 4: Khảo sát đối chuẩn trên Notebook
Mở và chạy [`comparison.ipynb`](comparison.ipynb) trên VS Code hoặc JupyterLab để quan sát trực quan các bảng đối chuẩn, đường cong học tập, ma trận nhầm lẫn và đường biên tối ưu Pareto.
