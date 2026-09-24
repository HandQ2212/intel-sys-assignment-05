import json
import os

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(True)}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}

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

imports_code = """import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')
"""

data_code = """DATA_DIR = '../intel-sys-assignment-04/dataset/mnist_number/'
TRAIN_DIR = os.path.join(DATA_DIR, 'Augmented MNIST Training Set (400k)')
TEST_DIR = os.path.join(DATA_DIR, 'MNIST Validation Set (4k)')

train_transforms = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(28),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transforms = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)

torch.manual_seed(42)
indices = torch.randperm(len(full_train_dataset))[:20000]
train_dataset = Subset(full_train_dataset, indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class_names = [str(i) for i in range(10)]
print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {class_names}")
"""

explore_code = """# Data Exploration
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

class_found = {i: False for i in range(10)}
images_to_show = {}

for imgs, labels in train_loader:
    for img, label in zip(imgs, labels):
        lbl = label.item()
        if not class_found[lbl]:
            images_to_show[lbl] = img
            class_found[lbl] = True
        if all(class_found.values()):
            break
    if all(class_found.values()):
        break

for i in range(10):
    img = images_to_show[i].squeeze().numpy()
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'Class: {i}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

from collections import Counter
train_labels = [full_train_dataset.targets[idx] for idx in indices.tolist()]
label_counts = Counter(train_labels)

plt.figure(figsize=(10, 5))
sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()))
plt.title('Class Distribution in Train Subset (20000)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()
"""

train_func_code = """def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, correct / total

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return running_loss / total, correct / total, all_preds, all_labels
"""

train_loop_code = """criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'time': 0}
start_time = time.time()

print(f"Starting training {MODEL_NAME} for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

history['time'] = time.time() - start_time
print(f"Training completed in {history['time']:.2f} seconds.")
"""

viz_code = """plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['test_loss'], label='Test')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['test_acc'], label='Test')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
"""

eval_code = """_, _, y_pred, y_true = evaluate(model, test_loader, criterion)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
"""

save_code = """os.makedirs('results', exist_ok=True)
result_path = f"results/{DATASET_NAME.lower()}_{MODEL_KEY}.json"

results_dict = {
    "model": MODEL_NAME,
    "dataset": DATASET_NAME,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "train_acc": history['train_acc'][-1],
    "test_acc": history['test_acc'][-1],
    "time": history['time']
}

with open(result_path, 'w') as f:
    json.dump(results_dict, f, indent=4)

print(f"Results saved to {result_path}")
"""

models = [
    {
        "file": "mnist_basic_cnn.ipynb",
        "key": "basic",
        "name": "Basic CNN",
        "theory": "# Lý thuyết: Basic CNN\n\nMạng CNN cơ bản được xây dựng dựa trên sự xếp chồng các tầng tích chập (Convolutional Layers), các tầng chuẩn hóa (Batch Normalization), hàm kích hoạt (ReLU) và các tầng pooling.\n\n**Toán học của tích chập:**\n$$ Y_{i,j} = \\sum_{m} \\sum_{n} X_{i+m, j+n} \\cdot K_{m,n} + b $$\n\n**Hàm kích hoạt ReLU:**\n$$ f(x) = \\max(0, x) $$\n\n**Cấu trúc mạng (Function Composition):**\n$$ f(x) = f_{classifier}(f_{features}(x)) $$\nTrong đó $f_{features}$ thực hiện trích xuất đặc trưng không gian thông qua các phép tích chập và gộp, còn $f_{classifier}$ đưa ra dự đoán phân loại dựa trên các đặc trưng đó.\n",
        "code": "class BasicCNN(nn.Module):\n    def __init__(self, in_channels=1, num_classes=10):\n        super().__init__()\n        self.features = nn.Sequential(\n            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),\n            nn.BatchNorm2d(32),\n            nn.ReLU(inplace=True),\n            nn.MaxPool2d(2),\n            nn.Conv2d(32, 64, kernel_size=3, padding=1),\n            nn.BatchNorm2d(64),\n            nn.ReLU(inplace=True),\n            nn.MaxPool2d(2),\n        )\n        self.classifier = nn.Sequential(\n            nn.AdaptiveAvgPool2d(1),\n            nn.Flatten(),\n            nn.Linear(64, num_classes),\n        )\n    def forward(self, x):\n        return self.classifier(self.features(x))\n\nmodel = BasicCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)\nprint(f\"Parameters: {sum(p.numel() for p in model.parameters())}\")\n"
    },
    {
        "file": "mnist_vgg_cnn.ipynb",
        "key": "vgg",
        "name": "VGG-style CNN",
        "theory": "# Lý thuyết: VGG-style CNN\n\nKiến trúc VGG nổi tiếng với việc sử dụng liên tiếp nhiều filter kích thước nhỏ $3 \\times 3$ thay vì các filter lớn. Điều này giúp tăng độ sâu của mạng đồng thời giảm số lượng tham số.\n\n**Receptive Field Equivalence:**\nHai tầng tích chập $3 \\times 3$ liên tiếp có vùng cảm nhận (receptive field) tương đương với một tầng $5 \\times 5$, và ba tầng $3 \\times 3$ tương đương với một tầng $7 \\times 7$. Việc sử dụng nhiều tầng $3 \\times 3$ kèm theo các hàm kích hoạt phi tuyến (ReLU) giữa chúng làm tăng tính biểu diễn của mạng.\n",
        "code": "class VGGStyleCNN(nn.Module):\n    def __init__(self, in_channels=1, num_classes=10):\n        super().__init__()\n        self.features = nn.Sequential(\n            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),\n            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),\n            nn.MaxPool2d(2),\n            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),\n            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),\n            nn.MaxPool2d(2),\n            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),\n            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),\n            nn.MaxPool2d(2),\n        )\n        self.classifier = nn.Sequential(\n            nn.AdaptiveAvgPool2d(1), nn.Flatten(),\n            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),\n            nn.Linear(256, num_classes),\n        )\n    def forward(self, x):\n        return self.classifier(self.features(x))\n\nmodel = VGGStyleCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)\nprint(f\"Parameters: {sum(p.numel() for p in model.parameters())}\")\n"
    },
    {
        "file": "mnist_resnet_cnn.ipynb",
        "key": "resnet",
        "name": "ResNet-style CNN",
        "theory": "# Lý thuyết: ResNet-style CNN\n\nVấn đề thoái hóa (Degradation Problem): Khi mạng nơ-ron trở nên quá sâu, sai số huấn luyện có xu hướng tăng lên.\n\n**Residual Learning (Học phần dư):**\nThay vì cố gắng học trực tiếp một hàm biểu diễn tiềm ẩn $H(x)$, ResNet học một hàm phần dư $F(x) = H(x) - x$. Do đó, ánh xạ cuối cùng là:\n$$ Y = F(x) + x $$\n\n**Skip Connections (Kết nối tắt):**\nPhép cộng được thực hiện thông qua các \"skip connections\", giúp đạo hàm truyền ngược trực tiếp qua các tầng, giảm hiện tượng vanishing gradient.\n",
        "code": "class ResidualBlock(nn.Module):\n    def __init__(self, channels):\n        super().__init__()\n        self.block = nn.Sequential(\n            nn.Conv2d(channels, channels, 3, padding=1, bias=False),\n            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),\n            nn.Conv2d(channels, channels, 3, padding=1, bias=False),\n            nn.BatchNorm2d(channels),\n        )\n        self.relu = nn.ReLU(inplace=True)\n    def forward(self, x):\n        return self.relu(self.block(x) + x)\n\nclass ResNetStyleCNN(nn.Module):\n    def __init__(self, in_channels=1, num_classes=10):\n        super().__init__()\n        self.initial = nn.Sequential(\n            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),\n            nn.BatchNorm2d(64), nn.ReLU(inplace=True),\n        )\n        self.res1 = ResidualBlock(64)\n        self.down1 = nn.Sequential(\n            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),\n            nn.BatchNorm2d(128), nn.ReLU(inplace=True),\n        )\n        self.res2 = ResidualBlock(128)\n        self.down2 = nn.Sequential(\n            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),\n            nn.BatchNorm2d(256), nn.ReLU(inplace=True),\n        )\n        self.res3 = ResidualBlock(256)\n        self.classifier = nn.Sequential(\n            nn.AdaptiveAvgPool2d(1), nn.Flatten(),\n            nn.Linear(256, num_classes),\n        )\n    def forward(self, x):\n        x = self.initial(x)\n        x = self.down1(self.res1(x))\n        x = self.down2(self.res2(x))\n        return self.classifier(self.res3(x))\n\nmodel = ResNetStyleCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)\nprint(f\"Parameters: {sum(p.numel() for p in model.parameters())}\")\n"
    },
    {
        "file": "mnist_attention_cnn.ipynb",
        "key": "attention",
        "name": "Attention CNN (CBAM)",
        "theory": "# Lý thuyết: Attention CNN (CBAM)\n\nCơ chế Attention giúp mạng tập trung vào các đặc trưng quan trọng nhất. CBAM kết hợp hai loại attention:\n\n**1. Channel Attention (WHAT):**\nTập trung vào \"kênh nào\" là quan trọng:\n$$ M_c(F) = \\sigma(MLP(AvgPool(F)) + MLP(MaxPool(F))) $$\n\n**2. Spatial Attention (WHERE):**\nTập trung vào \"vị trí nào\" mang thông tin quan trọng:\n$$ M_s(F) = \\sigma(f^{7 \\times 7}([AvgPool(F); MaxPool(F)])) $$\n",
        "code": "class ChannelAttention(nn.Module):\n    def __init__(self, channels, reduction=16):\n        super().__init__()\n        hidden = max(channels // reduction, 4)\n        self.avg_pool = nn.AdaptiveAvgPool2d(1)\n        self.max_pool = nn.AdaptiveMaxPool2d(1)\n        self.fc = nn.Sequential(\n            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),\n            nn.Linear(hidden, channels, bias=False),\n        )\n        self.sigmoid = nn.Sigmoid()\n    def forward(self, x):\n        b, c, _, _ = x.shape\n        avg_out = self.fc(self.avg_pool(x).view(b, c))\n        max_out = self.fc(self.max_pool(x).view(b, c))\n        return x * self.sigmoid(avg_out + max_out).view(b, c, 1, 1)\n\nclass SpatialAttention(nn.Module):\n    def __init__(self, kernel_size=7):\n        super().__init__()\n        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)\n        self.sigmoid = nn.Sigmoid()\n    def forward(self, x):\n        avg_out = torch.mean(x, dim=1, keepdim=True)\n        max_out, _ = torch.max(x, dim=1, keepdim=True)\n        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))\n\nclass CBAM(nn.Module):\n    def __init__(self, channels, reduction=16, spatial_kernel=7):\n        super().__init__()\n        self.ca = ChannelAttention(channels, reduction)\n        self.sa = SpatialAttention(spatial_kernel)\n    def forward(self, x):\n        return self.sa(self.ca(x))\n\nclass AttentionCNN(nn.Module):\n    def __init__(self, in_channels=1, num_classes=10):\n        super().__init__()\n        self.block1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))\n        self.cbam1 = CBAM(32, reduction=4)\n        self.pool1 = nn.MaxPool2d(2)\n        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))\n        self.cbam2 = CBAM(64, reduction=8)\n        self.pool2 = nn.MaxPool2d(2)\n        self.block3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))\n        self.cbam3 = CBAM(128, reduction=16)\n        self.pool3 = nn.MaxPool2d(2)\n        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes))\n    def forward(self, x):\n        x = self.pool1(self.cbam1(self.block1(x)))\n        x = self.pool2(self.cbam2(self.block2(x)))\n        x = self.pool3(self.cbam3(self.block3(x)))\n        return self.classifier(x)\n\nmodel = AttentionCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)\nprint(f\"Parameters: {sum(p.numel() for p in model.parameters())}\")\n"
    }
]

for m in models:
    cells = []
    
    # 1. Title & Overview
    cells.append(md_cell(f"# Model: {m['name']}\\n**Dataset:** MNIST\\n**Task:** Multiclass Image Classification (10 classes)"))
    
    # 2. Theory
    cells.append(md_cell(m['theory']))
    
    # 3. Imports
    cells.append(code_cell(imports_code))
    
    # 4. Config
    config_code = f"EPOCHS = 20\nBATCH_SIZE = 64\nLR = 1e-3\nIN_CHANNELS = 1\nNUM_CLASSES = 10\nMODEL_NAME = '{m['name']}'\nMODEL_KEY = '{m['key']}'\nDATASET_NAME = 'MNIST'\n"
    cells.append(code_cell(config_code))
    
    # 5. Data Loading & 6. Data Exploration
    cells.append(md_cell("## Data Loading"))
    cells.append(code_cell(data_code))
    cells.append(md_cell("## Data Exploration"))
    cells.append(code_cell(explore_code))
    
    # 7. Model Architecture
    cells.append(md_cell("## Model Architecture"))
    cells.append(code_cell(m['code']))
    
    # 8. Training Functions
    cells.append(md_cell("## Training Functions"))
    cells.append(code_cell(train_func_code))
    
    # 9. Training Loop
    cells.append(md_cell("## Training Loop"))
    cells.append(code_cell(train_loop_code))
    
    # 10. Visualization
    cells.append(md_cell("## Visualization"))
    cells.append(code_cell(viz_code))
    
    # 11. Final Evaluation
    cells.append(md_cell("## Final Evaluation"))
    cells.append(code_cell(eval_code))
    
    # 12. Save Results
    cells.append(md_cell("## Save Results"))
    cells.append(code_cell(save_code))
    
    # 13. Conclusion
    cells.append(md_cell("## Conclusion\\nTraining and evaluation completed successfully."))
    
    save_notebook(cells, m['file'])
