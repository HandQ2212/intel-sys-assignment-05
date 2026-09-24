#!/usr/bin/env python3
"""Generate comparison notebook for Assignment 05."""
import json
import os

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def save_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0",
                              "codemirror_mode": {"name": "ipython", "version": 3},
                              "file_extension": ".py", "mimetype": "text/x-python"}
        },
        "nbformat": 4, "nbformat_minor": 4
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Created: {filepath}")

cells = []

# ═══════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════
cells.append(md_cell(
"# So sánh & Đánh giá 4 Mô hình CNN\n"
"## Bài tập 05 — Intelligent System Development\n"
"\n"
"Notebook này tổng hợp và so sánh kết quả từ 12 thí nghiệm:\n"
"- **3 tập dữ liệu**: CIFAR-10, MNIST, Diabetes\n"
"- **4 mô hình CNN**: Basic CNN, VGG-style, ResNet-style, Attention CNN (CBAM)\n"
"\n"
"### Tiêu chí đánh giá\n"
"| Tiêu chí | Mô tả |\n"
"|----------|--------|\n"
"| **Accuracy** | Tỷ lệ dự đoán đúng tổng thể |\n"
"| **F1-Score** | Trung bình điều hòa của Precision và Recall |\n"
"| **Training Loss** | Đường cong loss qua các epoch |\n"
"| **Parameters** | Tổng số tham số của mô hình |\n"
"| **Training Time** | Thời gian huấn luyện |"
))

# ═══════════════════════════════════════════════════════════
# Theory overview
# ═══════════════════════════════════════════════════════════
cells.append(md_cell(
"## Tổng quan các Mô hình\n"
"\n"
"### M1: Basic CNN (Conv + BN + ReLU + Pool)\n"
"$$\\hat{y} = f_{\\text{FC}} \\circ f_{\\text{pool}_2} \\circ f_{\\text{relu}_2} \\circ f_{\\text{conv}_2} \\circ f_{\\text{pool}_1} \\circ f_{\\text{relu}_1} \\circ f_{\\text{conv}_1}(X)$$\n"
"\n"
"### M2: VGG-style CNN (Deep 3×3 stacking)\n"
"Nhiều lớp Conv $3 \\times 3$ xếp chồng → tăng depth, giảm params so với kernel lớn.\n"
"\n"
"### M3: ResNet-style CNN (Residual Connections)\n"
"$$Y = F(X) + X$$\n"
"Skip connection cho phép gradient chảy trực tiếp → train được mạng sâu hơn.\n"
"\n"
"### M4: Attention CNN (CBAM)\n"
"$$f_{\\text{CBAM}}(X) = M_s(M_c(X) \\odot X) \\odot (M_c(X) \\odot X)$$\n"
"Channel Attention (\"WHAT?\") + Spatial Attention (\"WHERE?\")."
))

# ═══════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════
cells.append(code_cell(
"import json\n"
"import os\n"
"import glob\n"
"import numpy as np\n"
"import pandas as pd\n"
"import matplotlib.pyplot as plt\n"
"import seaborn as sns\n"
"import warnings\n"
"warnings.filterwarnings('ignore')\n"
"\n"
"plt.style.use('seaborn-v0_8-whitegrid')\n"
"plt.rcParams['figure.figsize'] = (14, 6)\n"
"plt.rcParams['font.size'] = 12"
))

# ═══════════════════════════════════════════════════════════
# Load results
# ═══════════════════════════════════════════════════════════
cells.append(md_cell(
"## 1. Tải Kết quả\n"
"\n"
"Kết quả được lưu từ mỗi notebook training vào thư mục `results/`."
))

cells.append(code_cell(
"# Load all result files\n"
"result_files = sorted(glob.glob('results/*.json'))\n"
"\n"
"if not result_files:\n"
"    print('⚠️  Chưa có kết quả! Hãy chạy các notebook training trước.')\n"
"    print('Các file cần chạy:')\n"
"    notebooks = [\n"
"        'cifar10_basic_cnn.ipynb', 'cifar10_vgg_cnn.ipynb',\n"
"        'cifar10_resnet_cnn.ipynb', 'cifar10_attention_cnn.ipynb',\n"
"        'mnist_basic_cnn.ipynb', 'mnist_vgg_cnn.ipynb',\n"
"        'mnist_resnet_cnn.ipynb', 'mnist_attention_cnn.ipynb',\n"
"        'diabetes_basic_cnn.ipynb', 'diabetes_vgg_cnn.ipynb',\n"
"        'diabetes_resnet_cnn.ipynb', 'diabetes_attention_cnn.ipynb',\n"
"    ]\n"
"    for nb in notebooks:\n"
"        print(f'  - {nb}')\n"
"else:\n"
"    all_results = []\n"
"    for f in result_files:\n"
"        with open(f, 'r') as fp:\n"
"            data = json.load(fp)\n"
"        all_results.append(data)\n"
"        print(f'✓ Loaded: {os.path.basename(f)}')\n"
"    print(f'\\nTổng cộng: {len(all_results)} kết quả')"
))

# ═══════════════════════════════════════════════════════════
# Build comparison table
# ═══════════════════════════════════════════════════════════
cells.append(md_cell(
"## 2. Bảng So sánh Tổng hợp"
))

cells.append(code_cell(
"if result_files:\n"
"    # Build comparison DataFrame\n"
"    rows = []\n"
"    for r in all_results:\n"
"        rows.append({\n"
"            'Dataset': r.get('dataset', 'N/A'),\n"
"            'Model': r.get('model', 'N/A'),\n"
"            'Test Accuracy': r.get('test_accs', [0])[-1] if r.get('test_accs') else 0,\n"
"            'Test Loss': r.get('test_losses', [0])[-1] if r.get('test_losses') else 0,\n"
"            'Parameters': r.get('num_params', 0),\n"
"            'Epochs': r.get('epochs', 0),\n"
"        })\n"
"    \n"
"    df_results = pd.DataFrame(rows)\n"
"    \n"
"    # Format for display\n"
"    df_display = df_results.copy()\n"
"    df_display['Test Accuracy'] = df_display['Test Accuracy'].apply(lambda x: f'{x:.4f}')\n"
"    df_display['Test Loss'] = df_display['Test Loss'].apply(lambda x: f'{x:.4f}')\n"
"    df_display['Parameters'] = df_display['Parameters'].apply(lambda x: f'{x:,}')\n"
"    \n"
"    print('═' * 80)\n"
"    print('BẢNG SO SÁNH TỔNG HỢP CÁC MÔ HÌNH CNN')\n"
"    print('═' * 80)\n"
"    display(df_display)\n"
"else:\n"
"    print('Chưa có kết quả để so sánh.')"
))

# ═══════════════════════════════════════════════════════════
# Accuracy comparison chart
# ═══════════════════════════════════════════════════════════
cells.append(md_cell(
"## 3. Biểu đồ So sánh Accuracy"
))

cells.append(code_cell(
"if result_files:\n"
"    datasets = df_results['Dataset'].unique()\n"
"    models = df_results['Model'].unique()\n"
"    \n"
"    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5))\n"
"    if len(datasets) == 1:\n"
"        axes = [axes]\n"
"    \n"
"    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']\n"
"    \n"
"    for idx, ds in enumerate(datasets):\n"
"        subset = df_results[df_results['Dataset'] == ds]\n"
"        bars = axes[idx].bar(subset['Model'], subset['Test Accuracy'].astype(float),\n"
"                            color=colors[:len(subset)])\n"
"        axes[idx].set_title(f'{ds}', fontsize=14, fontweight='bold')\n"
"        axes[idx].set_ylabel('Accuracy')\n"
"        axes[idx].set_ylim(0, 1.05)\n"
"        axes[idx].tick_params(axis='x', rotation=30)\n"
"        \n"
"        # Add value labels on bars\n"
"        for bar, val in zip(bars, subset['Test Accuracy'].astype(float)):\n"
"            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,\n"
"                          f'{val:.3f}', ha='center', va='bottom', fontsize=10)\n"
"    \n"
"    plt.suptitle('So sánh Accuracy giữa các Mô hình CNN', fontsize=16, fontweight='bold')\n"
"    plt.tight_layout()\n"
"    plt.show()"
))

# ═══════════════════════════════════════════════════════════
# Training curves comparison
# ═══════════════════════════════════════════════════════════
cells.append(md_cell(
"## 4. So sánh Đường cong Training"
))

cells.append(code_cell(
"if result_files:\n"
"    datasets = df_results['Dataset'].unique()\n"
"    \n"
"    fig, axes = plt.subplots(len(datasets), 2, figsize=(14, 5*len(datasets)))\n"
"    if len(datasets) == 1:\n"
"        axes = axes.reshape(1, -1)\n"
"    \n"
"    colors = {'Basic CNN': '#2196F3', 'VGG-style CNN': '#4CAF50',\n"
"              'ResNet-style CNN': '#FF9800', 'Attention CNN (CBAM)': '#E91E63',\n"
"              'Basic CNN 1D': '#2196F3', 'VGG-style CNN 1D': '#4CAF50',\n"
"              'ResNet-style CNN 1D': '#FF9800', 'Attention CNN 1D (CBAM)': '#E91E63'}\n"
"    \n"
"    for ds_idx, ds in enumerate(datasets):\n"
"        ds_results = [r for r in all_results if r.get('dataset') == ds]\n"
"        \n"
"        for r in ds_results:\n"
"            model_name = r.get('model', 'N/A')\n"
"            color = colors.get(model_name, '#333333')\n"
"            \n"
"            if r.get('train_losses'):\n"
"                axes[ds_idx, 0].plot(r['train_losses'], label=f'{model_name} (train)',\n"
"                                     color=color, linestyle='-')\n"
"            if r.get('test_losses'):\n"
"                axes[ds_idx, 0].plot(r['test_losses'], label=f'{model_name} (test)',\n"
"                                     color=color, linestyle='--')\n"
"            \n"
"            if r.get('train_accs'):\n"
"                axes[ds_idx, 1].plot(r['train_accs'], label=f'{model_name} (train)',\n"
"                                     color=color, linestyle='-')\n"
"            if r.get('test_accs'):\n"
"                axes[ds_idx, 1].plot(r['test_accs'], label=f'{model_name} (test)',\n"
"                                     color=color, linestyle='--')\n"
"        \n"
"        axes[ds_idx, 0].set_title(f'{ds} — Loss', fontsize=13)\n"
"        axes[ds_idx, 0].set_xlabel('Epoch')\n"
"        axes[ds_idx, 0].set_ylabel('Loss')\n"
"        axes[ds_idx, 0].legend(fontsize=8)\n"
"        axes[ds_idx, 0].grid(True)\n"
"        \n"
"        axes[ds_idx, 1].set_title(f'{ds} — Accuracy', fontsize=13)\n"
"        axes[ds_idx, 1].set_xlabel('Epoch')\n"
"        axes[ds_idx, 1].set_ylabel('Accuracy')\n"
"        axes[ds_idx, 1].legend(fontsize=8)\n"
"        axes[ds_idx, 1].grid(True)\n"
"    \n"
"    plt.suptitle('So sánh Đường cong Training', fontsize=16, fontweight='bold')\n"
"    plt.tight_layout()\n"
"    plt.show()"
))

# ═══════════════════════════════════════════════════════════
# Parameter comparison
# ═══════════════════════════════════════════════════════════
cells.append(md_cell(
"## 5. So sánh Số lượng Tham số"
))

cells.append(code_cell(
"if result_files:\n"
"    fig, ax = plt.subplots(figsize=(12, 5))\n"
"    \n"
"    # Group by model\n"
"    labels = []\n"
"    param_counts = []\n"
"    bar_colors = []\n"
"    color_map = {'Basic': '#2196F3', 'VGG': '#4CAF50', 'ResNet': '#FF9800', 'Attention': '#E91E63'}\n"
"    \n"
"    for r in all_results:\n"
"        label = f\"{r.get('dataset', '?')}\\n{r.get('model', '?')}\"\n"
"        labels.append(label)\n"
"        param_counts.append(r.get('num_params', 0))\n"
"        \n"
"        model_name = r.get('model', '')\n"
"        if 'Basic' in model_name:\n"
"            bar_colors.append(color_map['Basic'])\n"
"        elif 'VGG' in model_name:\n"
"            bar_colors.append(color_map['VGG'])\n"
"        elif 'ResNet' in model_name:\n"
"            bar_colors.append(color_map['ResNet'])\n"
"        elif 'Attention' in model_name:\n"
"            bar_colors.append(color_map['Attention'])\n"
"        else:\n"
"            bar_colors.append('#999999')\n"
"    \n"
"    bars = ax.bar(range(len(labels)), param_counts, color=bar_colors)\n"
"    ax.set_xticks(range(len(labels)))\n"
"    ax.set_xticklabels(labels, fontsize=8, rotation=0)\n"
"    ax.set_ylabel('Number of Parameters')\n"
"    ax.set_title('So sánh Số lượng Tham số', fontsize=14, fontweight='bold')\n"
"    \n"
"    for bar, count in zip(bars, param_counts):\n"
"        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),\n"
"               f'{count:,}', ha='center', va='bottom', fontsize=7)\n"
"    \n"
"    plt.tight_layout()\n"
"    plt.show()"
))

# ═══════════════════════════════════════════════════════════
# Heatmap comparison
# ═══════════════════════════════════════════════════════════
cells.append(md_cell(
"## 6. Heatmap So sánh"
))

cells.append(code_cell(
"if result_files:\n"
"    # Create pivot table: Dataset × Model → Accuracy\n"
"    pivot = df_results.pivot_table(\n"
"        values='Test Accuracy', \n"
"        index='Dataset', \n"
"        columns='Model',\n"
"        aggfunc='first'\n"
"    ).astype(float)\n"
"    \n"
"    plt.figure(figsize=(10, 4))\n"
"    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', \n"
"                linewidths=1, linecolor='white',\n"
"                cbar_kws={'label': 'Accuracy'})\n"
"    plt.title('Accuracy Heatmap: Dataset × Model', fontsize=14, fontweight='bold')\n"
"    plt.ylabel('Dataset')\n"
"    plt.xlabel('Model')\n"
"    plt.tight_layout()\n"
"    plt.show()"
))

# ═══════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════
cells.append(md_cell(
"## 7. Phân tích & Nhận xét"
))

cells.append(code_cell(
"if result_files:\n"
"    print('═' * 70)\n"
"    print('PHÂN TÍCH KẾT QUẢ')\n"
"    print('═' * 70)\n"
"    \n"
"    datasets = df_results['Dataset'].unique()\n"
"    \n"
"    for ds in datasets:\n"
"        subset = df_results[df_results['Dataset'] == ds].copy()\n"
"        subset['Test Accuracy'] = subset['Test Accuracy'].astype(float)\n"
"        \n"
"        best_idx = subset['Test Accuracy'].idxmax()\n"
"        best_model = subset.loc[best_idx, 'Model']\n"
"        best_acc = subset.loc[best_idx, 'Test Accuracy']\n"
"        \n"
"        print(f'\\n📊 Dataset: {ds}')\n"
"        print(f'   Mô hình tốt nhất: {best_model} (Accuracy = {best_acc:.4f})')\n"
"        print(f'   Ranking:')\n"
"        ranked = subset.sort_values('Test Accuracy', ascending=False)\n"
"        for rank, (_, row) in enumerate(ranked.iterrows(), 1):\n"
"            print(f'      {rank}. {row[\"Model\"]}: {row[\"Test Accuracy\"]:.4f} '\n"
"                  f'({row[\"Parameters\"]:,.0f} params)')\n"
"    \n"
"    print(f'\\n{\"═\" * 70}')\n"
"    print('TỔNG KẾT')\n"
"    print(f'{\"═\" * 70}')\n"
"    print()\n"
"    print('1. Basic CNN: Mô hình đơn giản nhất, ít tham số, phù hợp làm baseline.')\n"
"    print('2. VGG-style: Tăng depth bằng 3×3 stacking → cải thiện accuracy nhưng nhiều params hơn.')\n"
"    print('3. ResNet-style: Skip connections giúp train sâu hơn, thường cho kết quả tốt.')\n"
"    print('4. Attention CNN: CBAM giúp model tập trung vào features quan trọng.')\n"
"    print()\n"
"    print('💡 Nhận xét chung:')\n"
"    print('   - Mô hình phức tạp hơn KHÔNG luôn cho accuracy cao hơn.')\n"
"    print('   - Cần cân bằng giữa accuracy và computational cost (params).')\n"
"    print('   - Hiệu quả phụ thuộc vào đặc điểm dữ liệu:')\n"
"    print('     + Image data: CNN 2D phù hợp tự nhiên.')\n"
"    print('     + Tabular data: CNN 1D là \"teaching experiment\", không tối ưu.')"
))

# ═══════════════════════════════════════════════════════════
# Conclusion
# ═══════════════════════════════════════════════════════════
cells.append(md_cell(
"## 8. Kết luận\n"
"\n"
"### Các bài học chính\n"
"\n"
"1. **CNN là hợp hàm**: Mọi kiến trúc CNN đều có thể biểu diễn dưới dạng $\\hat{y} = f_L \\circ f_{L-1} \\circ \\cdots \\circ f_1(X)$\n"
"\n"
"2. **Tiến hóa kiến trúc = Giải quyết hạn chế**:\n"
"   - Basic CNN → VGG: Tăng depth bằng small kernels\n"
"   - VGG → ResNet: Skip connections giải quyết degradation\n"
"   - ResNet → CBAM: Attention chọn lọc features quan trọng\n"
"\n"
"3. **Không có mô hình \"tốt nhất\" cho mọi bài toán**: Hiệu quả phụ thuộc vào data, task, và resources.\n"
"\n"
"4. **Tabular data + CNN = Teaching experiment**: CNN được thiết kế cho dữ liệu có cấu trúc không gian (ảnh, audio). Dữ liệu tabular không có spatial locality tự nhiên.\n"
"\n"
"### Công thức kiến trúc tổng quát\n"
"$$\\text{new\\_architecture} = \\text{old\\_architecture} + \\text{mechanism\\_addressing\\_a\\_limitation}$$"
))

# Save
save_notebook(cells, "comparison.ipynb")
