import json
import os

def code_cell(src):
    src_lines = [line + '\n' for line in src.split('\n')]
    src_lines[-1] = src_lines[-1].rstrip('\n')
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src_lines}

def md_cell(src):
    src_lines = [line + '\n' for line in src.split('\n')]
    src_lines[-1] = src_lines[-1].rstrip('\n')
    return {"cell_type": "markdown", "metadata": {}, "source": src_lines}

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

IMPORTS = """import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')
"""

def generate_config(model_name, dataset_name, model_key):
    return f"""EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
IN_CHANNELS = 3
NUM_CLASSES = 10
MODEL_NAME = '{model_name}'
DATASET_NAME = '{dataset_name}'
MODEL_KEY = '{model_key}'
"""

DATA_LOADING = """# Data transforms
train_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

test_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Load dataset
train_dir = '../intel-sys-assignment-04/dataset/cifar10/train/'
test_dir = '../intel-sys-assignment-04/dataset/cifar10/test/'

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class_names = train_dataset.classes
print(f'Classes: {class_names}')
print(f'Train size: {len(train_dataset)}, Test size: {len(test_dataset)}')
"""

DATA_EXPLORATION = """# Plot class distribution
train_counts = [0] * NUM_CLASSES
for _, label in train_dataset:
    train_counts[label] += 1

plt.figure(figsize=(10, 5))
plt.bar(class_names, train_counts)
plt.title('Class Distribution in Training Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Show a grid of sample images (one per class)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()
found_classes = set()

# To get unnormalized images for visualization
inv_normalize = transforms.Normalize(
    mean=[-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616],
    std=[1/0.2470, 1/0.2435, 1/0.2616]
)

for img, label in train_dataset:
    if label not in found_classes:
        found_classes.add(label)
        img = inv_normalize(img)
        img = img.numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        axes[label].imshow(img)
        axes[label].set_title(class_names[label])
        axes[label].axis('off')
    if len(found_classes) == NUM_CLASSES:
        break
plt.tight_layout()
plt.show()
"""

TRAIN_FNS = """def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels
"""

TRAIN_LOOP = """# Instantiate model, loss, optimizer
model = model_class(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print(f'Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
"""

VISUALIZATION = """plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()
"""

EVALUATION = """test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)

print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
"""

SAVE_RESULTS = """os.makedirs('results', exist_ok=True)
results = {
    'model': MODEL_NAME,
    'dataset': DATASET_NAME,
    'test_loss': test_loss,
    'test_acc': test_acc,
    'train_losses': train_losses,
    'train_accs': train_accs,
    'test_losses': test_losses,
    'test_accs': test_accs
}
with open(f'results/cifar10_{MODEL_KEY}.json', 'w') as f:
    json.dump(results, f, indent=4)
print(f"Results saved to results/cifar10_{MODEL_KEY}.json")
"""

M1_THEORY = """## 2. Cơ sở lý thuyết

**Mạng Convolutional Neural Network (CNN) cơ bản**

- **Phép toán Tích chập (Convolution):**
  Công thức tích chập 2D:
  $$Y(i,j) = \sum_{u} \sum_{v} K(u,v) \cdot X(i+u,j+v) + b$$
  Trong đó $X$ là ảnh đầu vào hoặc feature map, $K$ là kernel, $b$ là bias.
- **Tính chất:**
  - *Kết nối cục bộ (Local connectivity):* Mỗi neuron chỉ kết nối với một vùng nhỏ của đầu vào.
  - *Chia sẻ trọng số (Weight sharing):* Kernel được sử dụng chung trên toàn bộ ảnh, giúp giảm số lượng tham số.
- **Các thành phần khác:**
  - *Hàm kích hoạt ReLU:* $$f(x) = \max(0, x)$$
  - *Batch Normalization (BN):* Chuẩn hóa đầu ra của một lớp để quá trình huấn luyện ổn định hơn.
  - *Pooling:* Giảm kích thước không gian (spatial size) của feature map.
- **Hợp thành hàm (Function composition):**
  Một mạng CNN có thể coi như hợp thành của nhiều hàm:
  $$\hat{y} = f_{FC} \circ f_{pool2} \circ f_{relu2} \circ f_{conv2} \circ f_{pool1} \circ f_{relu1} \circ f_{conv1}(X)$$
"""

M1_CODE = """class BasicCNN(nn.Module):
    \"\"\"
    M1: Basic CNN
    ŷ = FC(Pool(ReLU(BN(Conv₂(Pool(ReLU(BN(Conv₁(X)))))))))
    \"\"\"
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model_class = BasicCNN
"""

M2_THEORY = """## 2. Cơ sở lý thuyết

**Mạng VGG-style CNN**

- **Kiến trúc xếp chồng sâu (Deep stacking):**
  Thay vì dùng các kernel kích thước lớn (như $5 \times 5$ hay $7 \times 7$), VGG sử dụng nhiều lớp tích chập với kernel nhỏ $3 \times 3$ xếp chồng lên nhau.
- **Receptive Field Equivalence:**
  Hai lớp chập $3 \times 3$ xếp chồng có Receptive Field tương đương với một lớp $5 \times 5$. Ba lớp $3 \times 3$ tương đương một lớp $7 \times 7$. Tuy nhiên, việc sử dụng nhiều lớp nhỏ có hai ưu điểm chính:
  - Giảm số lượng tham số (ví dụ: $3 \times 3^2 = 27$ nhỏ hơn $7^2 = 49$).
  - Tăng tính phi tuyến do có nhiều hàm ReLU xen kẽ.
- **Hợp thành hàm với khối (Blocks):**
  Mạng được chia thành các khối (blocks), mỗi khối gồm các tích chập và theo sau là Pooling.
  $$\hat{y} = f_{FC} \circ f_{block3} \circ f_{block2} \circ f_{block1}(X)$$
"""

M2_CODE = """class VGGStyleCNN(nn.Module):
    \"\"\"
    M2: VGG-style CNN — Deep stacking of 3×3 convolutions
    \"\"\"
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model_class = VGGStyleCNN
"""

M3_THEORY = """## 2. Cơ sở lý thuyết

**Mạng ResNet-style CNN**

- **Vấn đề suy thoái (Degradation problem):**
  Khi mạng trở nên quá sâu, độ chính xác có thể bão hòa và sau đó suy giảm nhanh chóng. Nguyên nhân không hẳn do overfitting mà là do rất khó để các lớp mạng tối ưu một ánh xạ đồng nhất (identity mapping).
- **Học thặng dư (Residual Learning):**
  ResNet giải quyết bằng cách đưa ra kết nối tắt (skip connection/shortcut connection). Mạng sẽ học hàm thặng dư:
  $$Y = \mathcal{F}(X) + X$$
  Thay vì bắt lớp phải học toàn bộ ánh xạ $H(X)$, nó chỉ cần học phần khác biệt $\mathcal{F}(X) = H(X) - X$.
- **Lợi ích của Identity Mapping:**
  Việc cho phép tín hiệu truyền trực tiếp qua lớp mạng (dưới dạng phép cộng) giúp duy trì gradient trong quá trình lan truyền ngược (backpropagation), giải quyết hiện tượng vanishing gradient, làm cho việc huấn luyện các mạng cực sâu trở nên khả thi.
"""

M3_CODE = """class ResidualBlock(nn.Module):
    \"\"\"Residual Block: Y = F(X) + X\"\"\"
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.block(x) + x)  # skip connection

class ResNetStyleCNN(nn.Module):
    \"\"\"
    M3: ResNet-style CNN with Residual Connections
    \"\"\"
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.res1 = ResidualBlock(64)
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.res2 = ResidualBlock(128)
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.res3 = ResidualBlock(256)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.initial(x)
        x = self.down1(self.res1(x))
        x = self.down2(self.res2(x))
        x = self.res3(x)
        return self.classifier(x)

model_class = ResNetStyleCNN
"""

M4_THEORY = """## 2. Cơ sở lý thuyết

**Mạng Attention CNN (CBAM)**

CBAM (Convolutional Block Attention Module) là một module chú ý nhẹ giúp trích xuất các đặc trưng quan trọng theo cả hai khía cạnh: kênh (channel) và không gian (spatial).
- **Chú ý kênh (Channel Attention - "WHAT is important?"):**
  Học cách đánh trọng số cho từng kênh đặc trưng. Kết hợp Max Pooling và Average Pooling, đi qua MLP chung.
  $$M_c(F) = \sigma(MLP(AvgPool(F)) + MLP(MaxPool(F)))$$
- **Chú ý không gian (Spatial Attention - "WHERE is important?"):**
  Tập trung vào phần nào trong mặt phẳng không gian mang nhiều thông tin. Gom kênh bằng pooling và áp dụng tích chập.
  $$M_s(F) = \sigma(f^{7 \\times 7}([AvgPool(F); MaxPool(F)]))$$
- **Hợp thành của CBAM:**
  Đầu ra đặc trưng sau khi qua chú ý kênh sẽ tiếp tục qua chú ý không gian:
  $$F' = M_c(F) \otimes F$$
  $$F'' = M_s(F') \otimes F'$$
"""

M4_CODE = """class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg_out + max_out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)
    def forward(self, x):
        return self.sa(self.ca(x))

class AttentionCNN(nn.Module):
    \"\"\"
    M4: CNN with CBAM Attention Modules
    \"\"\"
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.cbam1 = CBAM(32, reduction=4)
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.cbam2 = CBAM(64, reduction=8)
        self.pool2 = nn.MaxPool2d(2)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.cbam3 = CBAM(128, reduction=16)
        self.pool3 = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.pool1(self.cbam1(self.block1(x)))
        x = self.pool2(self.cbam2(self.block2(x)))
        x = self.pool3(self.cbam3(self.block3(x)))
        return self.classifier(x)

model_class = AttentionCNN
"""

models = [
    ('Basic CNN', 'basic', M1_THEORY, M1_CODE),
    ('VGG-style CNN', 'vgg', M2_THEORY, M2_CODE),
    ('ResNet-style CNN', 'resnet', M3_THEORY, M3_CODE),
    ('Attention CNN (CBAM)', 'attention', M4_THEORY, M4_CODE)
]

for model_name, model_key, theory, code in models:
    cells = [
        md_cell(f"# Model: {model_name}\\n**Dataset:** CIFAR-10\\n**Assignment Context:** Training CNNs from scratch on CIFAR-10."),
        md_cell(theory),
        md_cell("## 3. Imports and Setup"),
        code_cell(IMPORTS),
        md_cell("## 4. Configuration"),
        code_cell(generate_config(model_name, 'CIFAR-10', model_key)),
        md_cell("## 5. Data Loading"),
        code_cell(DATA_LOADING),
        md_cell("## 6. Data Exploration"),
        code_cell(DATA_EXPLORATION),
        md_cell("## 7. Model Architecture"),
        code_cell(code),
        md_cell("## 8. Training Functions"),
        code_cell(TRAIN_FNS),
        md_cell("## 9. Training Loop"),
        code_cell(TRAIN_LOOP),
        md_cell("## 10. Visualization"),
        code_cell(VISUALIZATION),
        md_cell("## 11. Final Evaluation"),
        code_cell(EVALUATION),
        md_cell("## 12. Save Results"),
        code_cell(SAVE_RESULTS),
        md_cell("## 13. Conclusion\\nModel trained successfully.")
    ]
    filename = f"cifar10_{model_key}_cnn.ipynb"
    save_notebook(cells, filename)
