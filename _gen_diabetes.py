import os
import json

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def save_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0", "codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python"}
        },
        "nbformat": 4, "nbformat_minor": 4
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Created: {filepath}")

models = [
    {
        "key": "basic",
        "name": "Basic CNN 1D",
        "filename": "diabetes_basic_cnn.ipynb",
        "theory": r"""## 2. Cơ sở lý thuyết: Basic CNN 1D cho dữ liệu bảng
Việc sử dụng mạng tích chập (CNN) trên dữ liệu bảng (tabular data) có thể xem là một bài toán thực nghiệm. Thông thường, CNN dùng cho ảnh vì có tính không gian địa phương (spatial locality). Đối với dữ liệu bảng, ta coi 21 đặc trưng như một chuỗi 1D chiều dài 21.

Mạng Basic CNN 1D sử dụng tích chập 1 chiều (Conv1d).
Công thức cho phép tích chập 1D:
$$ y[i] = \sum_{k=0}^{K-1} w[k] x[i - k] + b $$
Cấu trúc mạng:
$$ \hat{y} = FC(Pool(ReLU(BN(Conv1d_2(Pool(ReLU(BN(Conv1d_1(X))))))))) $$
""",
        "code": """class BasicCNN1D(nn.Module):
    \"\"\"
    M1: Basic 1D CNN for tabular data
    ŷ = FC(Pool(ReLU(BN(Conv1d₂(Pool(ReLU(BN(Conv1d₁(X)))))))))
    \"\"\"
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = BasicCNN1D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
print(model)
"""
    },
    {
        "key": "vgg",
        "name": "VGG-style CNN 1D",
        "filename": "diabetes_vgg_cnn.ipynb",
        "theory": r"""## 2. Cơ sở lý thuyết: VGG-style CNN 1D
VGG (Visual Geometry Group) nổi tiếng với việc xếp chồng nhiều lớp tích chập có kích thước kernel nhỏ (3x3) thay vì sử dụng kernel lớn. 
Đối với dữ liệu bảng 1D, ta xếp chồng các lớp `Conv1d(kernel_size=3)`.

Kiến trúc xếp chồng:
$$ Block = Conv1D \rightarrow BN \rightarrow ReLU \rightarrow Conv1D \rightarrow BN \rightarrow ReLU \rightarrow MaxPool1D $$

Tương tự mạng cơ bản, việc áp dụng CNN lên tabular data mang tính thực nghiệm, không gian dữ liệu không có sẵn tính chất local dependency như ảnh.
""",
        "code": """class VGGStyleCNN1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = VGGStyleCNN1D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
print(model)
"""
    },
    {
        "key": "resnet",
        "name": "ResNet-style CNN 1D",
        "filename": "diabetes_resnet_cnn.ipynb",
        "theory": r"""## 2. Cơ sở lý thuyết: ResNet-style CNN 1D
ResNet giải quyết vấn đề mất mát gradient (vanishing gradient) khi mô hình trở nên quá sâu bằng cách giới thiệu các kết nối tắt (residual/skip connections).

Hàm số tại một residual block trong 1D:
$$ y = \mathcal{F}(x, \{W_i\}) + x $$
Trong đó $\mathcal{F}(x, \{W_i\})$ biểu diễn phép chập qua các lớp Conv1d.

Dữ liệu bảng không có tính chất cục bộ tự nhiên, do vậy đây là một hướng tiếp cận theo kiểu "thử nghiệm".
""",
        "code": """class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.block(x) + x)

class ResNetStyleCNN1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        self.res1 = ResidualBlock1D(64)
        self.down1 = nn.Sequential(
            nn.Conv1d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        self.res2 = ResidualBlock1D(128)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.initial(x)
        x = self.down1(self.res1(x))
        x = self.res2(x)
        return self.classifier(x)

model = ResNetStyleCNN1D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
print(model)
"""
    },
    {
        "key": "attention",
        "name": "Attention CNN 1D (CBAM)",
        "filename": "diabetes_attention_cnn.ipynb",
        "theory": r"""## 2. Cơ sở lý thuyết: Attention CNN 1D (CBAM)
Cơ chế Attention (như CBAM - Convolutional Block Attention Module) giúp mô hình tập trung vào những đặc trưng quan trọng nhất.
Với 1D, CBAM bao gồm 2 phần:
1. **Channel Attention**: Nhấn mạnh các channel quan trọng (các bộ lọc tạo ra phản ứng mạnh).
   $$ M_c(F) = \sigma(MLP(AvgPool1D(F)) + MLP(MaxPool1D(F))) $$
2. **Temporal Attention** (thay cho Spatial trong 2D): Nhấn mạnh vị trí quan trọng trong chuỗi đặc trưng 1D.
   $$ M_s(F) = \sigma(Conv1D([AvgPool1D(F); MaxPool1D(F)])) $$
   
Ứng dụng trên dữ liệu tabular là để quan sát xem liệu việc kết hợp Attention có bù đắp được sự thiếu vắng về cấu trúc không gian tự nhiên của dữ liệu không.
""",
        "code": """class ChannelAttention1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg_out + max_out).view(b, c, 1)

class TemporalAttention1D(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=4, temporal_kernel=3):
        super().__init__()
        self.ca = ChannelAttention1D(channels, reduction)
        self.ta = TemporalAttention1D(temporal_kernel)
    def forward(self, x):
        return self.ta(self.ca(x))

class AttentionCNN1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv1d(in_channels, 32, 3, padding=1, bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True))
        self.cbam1 = CBAM1D(32, reduction=4)
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.cbam2 = CBAM1D(64, reduction=4)
        self.pool2 = nn.MaxPool1d(2)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, num_classes))
    def forward(self, x):
        x = self.pool1(self.cbam1(self.block1(x)))
        x = self.pool2(self.cbam2(self.block2(x)))
        return self.classifier(x)

model = AttentionCNN1D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
print(model)
"""
    }
]

for m in models:
    cells = []
    
    # 1. Title
    cells.append(md_cell(f"# Mô hình: {m['name']}\n**Dataset**: Diabetes\n**Bài tập**: Thực nghiệm CNN 1D trên dữ liệu Tabular\n"))
    
    # 2. Theory
    cells.append(md_cell(m['theory']))
    
    # 3. Imports
    cells.append(code_cell("""import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
"""))

    # 4. Config
    cells.append(code_cell(f"""EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3
IN_CHANNELS = 1
NUM_CLASSES = 1
DATASET_NAME = 'Diabetes'
MODEL_NAME = '{m['name']}'
MODEL_KEY = '{m['key']}'
"""))

    # 5. Data Loading & Preprocessing
    cells.append(md_cell("## 3. Data Loading & Preprocessing\n"))
    cells.append(code_cell("""df = pd.read_csv('../intel-sys-assignment-04/dataset/diabets.csv')
# Lấy mẫu 50000 dòng để tính toán nhanh hơn
df = df.sample(n=50000, random_state=42)

X = df.drop(columns=['Diabetes_binary']).values
y = df['Diabetes_binary'].values

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Chuyển thành tensor và thêm chiều channel (batch, channel, features) -> (batch, 1, 21)
X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
y_test_tensor = torch.FloatTensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
"""))

    # 6. Data Exploration
    cells.append(md_cell("## 4. Data Exploration\n"))
    cells.append(code_cell("""print("Head of dataset:")
display(df.head())
print("\\nDataset description:")
display(df.describe())
"""))
    cells.append(code_cell("""plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Diabetes_binary')
plt.title("Class Distribution")
plt.show()
"""))
    cells.append(code_cell("""plt.figure(figsize=(15, 12))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()
"""))

    # 7. Model Architecture
    cells.append(md_cell("## 5. Model Architecture\n"))
    cells.append(code_cell(m['code']))

    # 8. Training Functions
    cells.append(md_cell("## 6. Training Functions\n"))
    cells.append(code_cell("""criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(-1)  # (batch, 1) -> (batch,)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        predicted = (torch.sigmoid(outputs) >= 0.5).float()
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).squeeze(-1)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * inputs.size(0)
        probs = torch.sigmoid(outputs)
        predicted = (probs >= 0.5).float()
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
    return total_loss / total, correct / total, all_preds, all_labels, all_probs
"""))

    # 9. Training Loop
    cells.append(md_cell("## 7. Training Loop\n"))
    cells.append(code_cell("""train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc, _, _, _ = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
"""))

    # 10. Visualization
    cells.append(md_cell("## 8. Visualization\n"))
    cells.append(code_cell("""plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train')
plt.plot(test_accs, label='Test')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()
"""))

    # 11. Final Evaluation
    cells.append(md_cell("## 9. Final Evaluation\n"))
    cells.append(code_cell("""test_loss, test_acc, all_preds, all_labels, all_probs = evaluate(model, test_loader, criterion, device)

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['No Diabetes', 'Diabetes']))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
"""))

    # 12. Save Results
    cells.append(md_cell("## 10. Save Results\n"))
    cells.append(code_cell("""os.makedirs('results', exist_ok=True)
results = {
    'model_name': MODEL_NAME,
    'dataset': DATASET_NAME,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LR,
    'final_train_acc': train_accs[-1],
    'final_test_acc': test_accs[-1],
    'final_train_loss': train_losses[-1],
    'final_test_loss': test_losses[-1],
    'roc_auc': roc_auc
}

res_path = f"results/{DATASET_NAME.lower()}_{MODEL_KEY}.json"
with open(res_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"Saved results to {res_path}")
"""))

    # 13. Conclusion
    cells.append(md_cell(f"## 11. Conclusion\nMô hình {m['name']} đã được huấn luyện và đánh giá trên tập dữ liệu Diabetes (dữ liệu bảng coi như chuỗi 1D).\n"))

    save_notebook(cells, m['filename'])
