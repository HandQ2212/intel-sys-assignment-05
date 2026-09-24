#!/usr/bin/env python3
"""
Comprehensive Training & Evaluation Script for Assignment 05
Trains 4 CNN architectures (Basic, VGG-style, ResNet-style, Attention CBAM)
across 3 datasets (CIFAR-10, MNIST, Diabetes).
Saves metrics into results/ and publication-quality figures into figures/.
"""

import os
import sys
import time
import json
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"==> Using device: {device}", flush=True)

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# ==============================================================================
# MODEL ARCHITECTURES (2D FOR CIFAR-10 & MNIST)
# ==============================================================================

class BasicCNN2D(nn.Module):
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


class VGGStyleCNN2D(nn.Module):
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


class ResidualBlock2D(nn.Module):
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
        return self.relu(self.block(x) + x)


class ResNetStyleCNN2D(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.res1 = ResidualBlock2D(64)
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.res2 = ResidualBlock2D(128)
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.res3 = ResidualBlock2D(256)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down1(self.res1(x))
        x = self.down2(self.res2(x))
        return self.classifier(self.res3(x))


class ChannelAttention2D(nn.Module):
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


class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM2D(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention2D(channels, reduction)
        self.sa = SpatialAttention2D(spatial_kernel)

    def forward(self, x):
        return self.sa(self.ca(x))


class AttentionCNN2D(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.cbam1 = CBAM2D(32, reduction=4)
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.cbam2 = CBAM2D(64, reduction=8)
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.cbam3 = CBAM2D(128, reduction=16)
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


# ==============================================================================
# MODEL ARCHITECTURES (1D FOR DIABETES TABULAR DATA)
# ==============================================================================

class BasicCNN1D(nn.Module):
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


class VGGStyleCNN1D(nn.Module):
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


class ResidualBlock1D(nn.Module):
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


class ChannelAttention1D(nn.Module):
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
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.cbam1 = CBAM1D(32, reduction=4)
        self.pool1 = nn.MaxPool1d(2)

        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        self.cbam2 = CBAM1D(64, reduction=4)
        self.pool2 = nn.MaxPool1d(2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.pool1(self.cbam1(self.block1(x)))
        x = self.pool2(self.cbam2(self.block2(x)))
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# TRAINING & EVALUATION FUNCTIONS
# ==============================================================================
def train_epoch_multiclass(model, loader, criterion, optimizer, device):
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
def eval_multiclass(model, loader, criterion, device):
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
        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    return total_loss / total, correct / total, all_preds, all_labels

def train_epoch_binary(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        predicted = (torch.sigmoid(outputs) >= 0.5).float()
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / total, correct / total

@torch.no_grad()
def eval_binary(model, loader, criterion, device):
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
        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    return total_loss / total, correct / total, all_preds, all_labels, all_probs


# ==============================================================================
# PIPELINE 1: CIFAR-10 EXPERIMENT
# ==============================================================================
def run_cifar10():
    print("\n" + "="*80, flush=True)
    print(">>> RUNNING EXPERIMENTS ON CIFAR-10 DATASET", flush=True)
    print("="*80, flush=True)

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

    train_dir = 'dataset/cifar10/train/'
    test_dir = 'dataset/cifar10/test/'

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    test_ds = datasets.ImageFolder(test_dir, transform=test_transform)

    # Subsample 12,000 train and 2,400 test for fast convergence
    train_indices = torch.randperm(len(train_ds))[:12000]
    train_subset = Subset(train_ds, train_indices)
    test_indices = torch.randperm(len(test_ds))[:2400]
    test_subset = Subset(test_ds, test_indices)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    class_names = train_ds.classes

    # Visualize 1 sample per class quickly using pre-known class indices
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    class_indices = {c: i * 5000 for i, c in enumerate(range(10))}
    for lbl, idx in class_indices.items():
        r, c = lbl // 5, lbl % 5
        img, _ = train_ds[idx]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.2470, 0.2435, 0.2616]) + np.array([0.4914, 0.4822, 0.4465])
        img_np = np.clip(img_np, 0, 1)
        axes[r, c].imshow(img_np)
        axes[r, c].set_title(class_names[lbl], fontsize=10, fontweight='bold')
        axes[r, c].axis('off')
    plt.suptitle("Mẫu Ảnh Đại Diện 10 Lớp Tập Dữ Liệu CIFAR-10", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/cifar10_samples.png', dpi=300)
    plt.close()
    print("✓ Saved figures/cifar10_samples.png", flush=True)

    models_dict = {
        'basic': ('Basic CNN', BasicCNN2D(in_channels=3, num_classes=10)),
        'vgg': ('VGG-style CNN', VGGStyleCNN2D(in_channels=3, num_classes=10)),
        'resnet': ('ResNet-style CNN', ResNetStyleCNN2D(in_channels=3, num_classes=10)),
        'attention': ('Attention CNN (CBAM)', AttentionCNN2D(in_channels=3, num_classes=10)),
    }

    EPOCHS = 5
    results_cifar = {}

    for key, (name, model) in models_dict.items():
        print(f"\n--- Training {name} on CIFAR-10 ({count_parameters(model):,} params) ---", flush=True)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        train_losses, test_losses = [], []
        train_accs, test_accs = [], []
        t0 = time.time()

        for ep in range(1, EPOCHS + 1):
            tr_l, tr_a = train_epoch_multiclass(model, train_loader, criterion, optimizer, device)
            te_l, te_a, preds, targets = eval_multiclass(model, test_loader, criterion, device)
            train_losses.append(tr_l)
            test_losses.append(te_l)
            train_accs.append(tr_a)
            test_accs.append(te_a)
            print(f"Epoch {ep:2d}/{EPOCHS}: Train Loss={tr_l:.4f}, Train Acc={tr_a:.4f} | Test Loss={te_l:.4f}, Test Acc={te_a:.4f}", flush=True)

        elapsed = time.time() - t0
        te_l, te_a, preds, targets = eval_multiclass(model, test_loader, criterion, device)
        cm = confusion_matrix(targets, preds).tolist()
        cr = classification_report(targets, preds, target_names=class_names, output_dict=True)

        res = {
            'dataset': 'CIFAR-10',
            'model': name,
            'model_key': key,
            'num_params': count_parameters(model),
            'training_time': round(elapsed, 2),
            'epochs': EPOCHS,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'final_test_accuracy': te_a,
            'final_test_loss': te_l,
            'macro_f1': cr['macro avg']['f1-score'],
            'confusion_matrix': cm,
            'class_names': class_names
        }
        with open(f'results/cifar10_{key}.json', 'w') as fp:
            json.dump(res, fp, indent=2)
        results_cifar[key] = res

    # Plot CIFAR-10 training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'basic': '#1f77b4', 'vgg': '#2ca02c', 'resnet': '#ff7f0e', 'attention': '#d62728'}
    for key, res in results_cifar.items():
        c = colors[key]
        axes[0].plot(range(1, EPOCHS+1), res['train_losses'], label=f"{res['model']} (train)", color=c, linestyle=':')
        axes[0].plot(range(1, EPOCHS+1), res['test_losses'], label=f"{res['model']} (test)", color=c, linewidth=2)
        axes[1].plot(range(1, EPOCHS+1), res['train_accs'], label=f"{res['model']} (train)", color=c, linestyle=':')
        axes[1].plot(range(1, EPOCHS+1), res['test_accs'], label=f"{res['model']} (test)", color=c, linewidth=2)

    axes[0].set_title("Hàm Mất Mát (Cross-Entropy Loss) Theo Epoch - CIFAR-10", fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Độ Chính Xác (Accuracy) Theo Epoch - CIFAR-10", fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/cifar10_training_curves.png', dpi=300)
    plt.close()

    # Plot CIFAR-10 confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, (key, res) in enumerate(results_cifar.items()):
        ax = axes[idx // 2, idx % 2]
        cm_arr = np.array(res['confusion_matrix'])
        sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f"Ma Trận Nhầm Lẫn: {res['model']} (Acc: {res['final_test_accuracy']*100:.2f}%)", fontweight='bold')
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.tick_params(axis='x', rotation=45)
    plt.suptitle("Ma Trận Nhầm Lẫn Đối Chuẩn 4 Kiến Trúc CNN Trên CIFAR-10", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/cifar10_confusion_matrices.png', dpi=300)
    plt.close()
    print("✓ Completed CIFAR-10 pipeline", flush=True)


# ==============================================================================
# PIPELINE 2: MNIST EXPERIMENT
# ==============================================================================
def run_mnist():
    print("\n" + "="*80, flush=True)
    print(">>> RUNNING EXPERIMENTS ON MNIST DATASET", flush=True)
    print("="*80, flush=True)

    train_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(28),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dir = 'dataset/mnist_number/Augmented MNIST Training Set (400k)'
    test_dir = 'dataset/mnist_number/MNIST Validation Set (4k)'

    full_train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    test_ds = datasets.ImageFolder(test_dir, transform=test_transform)

    # Subsample 15,000 samples for swift training
    indices = torch.randperm(len(full_train_ds))[:15000]
    train_subset = Subset(full_train_ds, indices)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Quick sample plot: sample from each label folder
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for lbl in range(10):
        r, c = lbl // 5, lbl % 5
        # Each label has ~40k images, index lbl*40000 has that digit
        img, _ = full_train_ds[lbl * 40000]
        img_np = img.squeeze().numpy() * 0.3081 + 0.1307
        axes[r, c].imshow(img_np, cmap='gray')
        axes[r, c].set_title(f"Chữ số {lbl}", fontsize=10, fontweight='bold')
        axes[r, c].axis('off')
    plt.suptitle("Mẫu Ảnh Chữ Số Viết Tay (0–9) Tập Dữ Liệu MNIST", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/mnist_samples.png', dpi=300)
    plt.close()
    print("✓ Saved figures/mnist_samples.png", flush=True)

    models_dict = {
        'basic': ('Basic CNN', BasicCNN2D(in_channels=1, num_classes=10)),
        'vgg': ('VGG-style CNN', VGGStyleCNN2D(in_channels=1, num_classes=10)),
        'resnet': ('ResNet-style CNN', ResNetStyleCNN2D(in_channels=1, num_classes=10)),
        'attention': ('Attention CNN (CBAM)', AttentionCNN2D(in_channels=1, num_classes=10)),
    }

    EPOCHS = 5
    results_mnist = {}

    for key, (name, model) in models_dict.items():
        print(f"\n--- Training {name} on MNIST ({count_parameters(model):,} params) ---", flush=True)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        train_losses, test_losses = [], []
        train_accs, test_accs = [], []
        t0 = time.time()

        for ep in range(1, EPOCHS + 1):
            tr_l, tr_a = train_epoch_multiclass(model, train_loader, criterion, optimizer, device)
            te_l, te_a, preds, targets = eval_multiclass(model, test_loader, criterion, device)
            train_losses.append(tr_l)
            test_losses.append(te_l)
            train_accs.append(tr_a)
            test_accs.append(te_a)
            print(f"Epoch {ep:2d}/{EPOCHS}: Train Loss={tr_l:.4f}, Train Acc={tr_a:.4f} | Test Loss={te_l:.4f}, Test Acc={te_a:.4f}", flush=True)

        elapsed = time.time() - t0
        te_l, te_a, preds, targets = eval_multiclass(model, test_loader, criterion, device)
        cm = confusion_matrix(targets, preds).tolist()
        cr = classification_report(targets, preds, target_names=[str(i) for i in range(10)], output_dict=True)

        res = {
            'dataset': 'MNIST',
            'model': name,
            'model_key': key,
            'num_params': count_parameters(model),
            'training_time': round(elapsed, 2),
            'epochs': EPOCHS,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'final_test_accuracy': te_a,
            'final_test_loss': te_l,
            'macro_f1': cr['macro avg']['f1-score'],
            'confusion_matrix': cm,
            'class_names': [str(i) for i in range(10)]
        }
        with open(f'results/mnist_{key}.json', 'w') as fp:
            json.dump(res, fp, indent=2)
        results_mnist[key] = res

    # Plot MNIST training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'basic': '#1f77b4', 'vgg': '#2ca02c', 'resnet': '#ff7f0e', 'attention': '#d62728'}
    for key, res in results_mnist.items():
        c = colors[key]
        axes[0].plot(range(1, EPOCHS+1), res['train_losses'], label=f"{res['model']} (train)", color=c, linestyle=':')
        axes[0].plot(range(1, EPOCHS+1), res['test_losses'], label=f"{res['model']} (test)", color=c, linewidth=2)
        axes[1].plot(range(1, EPOCHS+1), res['train_accs'], label=f"{res['model']} (train)", color=c, linestyle=':')
        axes[1].plot(range(1, EPOCHS+1), res['test_accs'], label=f"{res['model']} (test)", color=c, linewidth=2)

    axes[0].set_title("Hàm Mất Mát (Cross-Entropy Loss) Theo Epoch - MNIST", fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Độ Chính Xác (Accuracy) Theo Epoch - MNIST", fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/mnist_training_curves.png', dpi=300)
    plt.close()

    # Plot MNIST confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, (key, res) in enumerate(results_mnist.items()):
        ax = axes[idx // 2, idx % 2]
        cm_arr = np.array(res['confusion_matrix'])
        sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False)
        ax.set_title(f"Ma Trận Nhầm Lẫn: {res['model']} (Acc: {res['final_test_accuracy']*100:.2f}%)", fontweight='bold')
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
    plt.suptitle("Ma Trận Nhầm Lẫn Đối Chuẩn 4 Kiến Trúc CNN Trên MNIST", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/mnist_confusion_matrices.png', dpi=300)
    plt.close()
    print("✓ Completed MNIST pipeline", flush=True)


# ==============================================================================
# PIPELINE 3: DIABETES (1D CNN TABULAR DATA)
# ==============================================================================
def run_diabetes():
    print("\n" + "="*80, flush=True)
    print(">>> RUNNING EXPERIMENTS ON DIABETES CDC TABULAR DATASET (1D CNN)", flush=True)
    print("="*80, flush=True)

    df = pd.read_csv('dataset/diabets.csv')
    df = df.sample(n=40000, random_state=SEED)

    target_col = 'Diabetes_binary'
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values
    y = df[target_col].values

    # EDA figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = df[target_col].value_counts()
    axes[0].bar(['Không tiểu đường (0)', 'Tiểu đường (1)'], counts.values, color=['#2b5c8f', '#d95f02'], width=0.5)
    axes[0].set_title("Phân Phối Nhãn Nhị Phân (Diabetes_binary)", fontweight='bold')
    axes[0].set_ylabel("Số lượng mẫu")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f"{v:,} ({v/len(df)*100:.1f}%)", ha='center', fontweight='bold')

    corr = df[['Diabetes_binary', 'HighBP', 'HighChol', 'BMI', 'GenHlth', 'Age', 'DiffWalk', 'HeartDiseaseorAttack']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1], cbar=True)
    axes[1].set_title("Ma Trận Tương Quan Các Đặc Trưng Lâm Sàng Quan Trọng", fontweight='bold')
    axes[1].tick_params(axis='x', rotation=35)

    plt.tight_layout()
    plt.savefig('figures/diabetes_eda.png', dpi=300)
    plt.close()
    print("✓ Saved figures/diabetes_eda.png", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape for 1D CNN: (Batch, Channels=1, Length=21)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=256, shuffle=False)

    models_dict = {
        'basic': ('Basic CNN 1D', BasicCNN1D(in_channels=1, num_classes=1)),
        'vgg': ('VGG-style CNN 1D', VGGStyleCNN1D(in_channels=1, num_classes=1)),
        'resnet': ('ResNet-style CNN 1D', ResNetStyleCNN1D(in_channels=1, num_classes=1)),
        'attention': ('Attention CNN 1D (CBAM)', AttentionCNN1D(in_channels=1, num_classes=1)),
    }

    EPOCHS = 8
    results_diabetes = {}

    for key, (name, model) in models_dict.items():
        print(f"\n--- Training {name} on Diabetes 1D ({count_parameters(model):,} params) ---", flush=True)
        model = model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        train_losses, test_losses = [], []
        train_accs, test_accs = [], []
        t0 = time.time()

        for ep in range(1, EPOCHS + 1):
            tr_l, tr_a = train_epoch_binary(model, train_loader, criterion, optimizer, device)
            te_l, te_a, preds, targets, probs = eval_binary(model, test_loader, criterion, device)
            train_losses.append(tr_l)
            test_losses.append(te_l)
            train_accs.append(tr_a)
            test_accs.append(te_a)
            print(f"Epoch {ep:2d}/{EPOCHS}: Train Loss={tr_l:.4f}, Train Acc={tr_a:.4f} | Test Loss={te_l:.4f}, Test Acc={te_a:.4f}", flush=True)

        elapsed = time.time() - t0
        te_l, te_a, preds, targets, probs = eval_binary(model, test_loader, criterion, device)
        cm = confusion_matrix(targets, preds).tolist()
        cr = classification_report(targets, preds, target_names=['No Diabetes', 'Diabetes'], output_dict=True)
        auc = roc_auc_score(targets, probs)
        fpr, tpr, _ = roc_curve(targets, probs)

        res = {
            'dataset': 'Diabetes CDC',
            'model': name,
            'model_key': key,
            'num_params': count_parameters(model),
            'training_time': round(elapsed, 2),
            'epochs': EPOCHS,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'final_test_accuracy': te_a,
            'final_test_loss': te_l,
            'macro_f1': cr['macro avg']['f1-score'],
            'auc_score': auc,
            'confusion_matrix': cm,
            'fpr': fpr.tolist()[:100],
            'tpr': tpr.tolist()[:100]
        }
        with open(f'results/diabetes_{key}.json', 'w') as fp:
            json.dump(res, fp, indent=2)
        results_diabetes[key] = res

    # Plot Diabetes training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'basic': '#1f77b4', 'vgg': '#2ca02c', 'resnet': '#ff7f0e', 'attention': '#d62728'}
    for key, res in results_diabetes.items():
        c = colors[key]
        axes[0].plot(range(1, EPOCHS+1), res['train_losses'], label=f"{res['model']} (train)", color=c, linestyle=':')
        axes[0].plot(range(1, EPOCHS+1), res['test_losses'], label=f"{res['model']} (test)", color=c, linewidth=2)
        axes[1].plot(range(1, EPOCHS+1), res['train_accs'], label=f"{res['model']} (train)", color=c, linestyle=':')
        axes[1].plot(range(1, EPOCHS+1), res['test_accs'], label=f"{res['model']} (test)", color=c, linewidth=2)

    axes[0].set_title("Hàm Mất Mát (BCE Loss) Theo Epoch - Diabetes 1D", fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Độ Chính Xác (Accuracy) Theo Epoch - Diabetes 1D", fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/diabetes_training_curves.png', dpi=300)
    plt.close()

    # Plot ROC curves for Diabetes
    plt.figure(figsize=(8, 6))
    for key, res in results_diabetes.items():
        plt.plot(res['fpr'], res['tpr'], label=f"{res['model']} (AUC = {res['auc_score']:.3f})", color=colors[key], linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Guess (AUC = 0.5)')
    plt.title("Đường Cong ROC Đối Chuẩn 4 Mô Hình 1D CNN Trên Diabetes", fontweight='bold', fontsize=13)
    plt.xlabel("Tỷ lệ Dương tính Giả (FPR)")
    plt.ylabel("Tỷ lệ Dương tính Thật (TPR)")
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/diabetes_roc_curves.png', dpi=300)
    plt.close()
    print("✓ Completed Diabetes pipeline", flush=True)


# ==============================================================================
# PIPELINE 4: OVERALL COMPARISON FIGURES
# ==============================================================================
def generate_overall_figures():
    print("\n" + "="*80, flush=True)
    print(">>> GENERATING OVERALL COMPARISON FIGURES", flush=True)
    print("="*80, flush=True)

    import glob
    files = sorted(glob.glob('results/*.json'))
    all_data = []
    for f in files:
        with open(f) as fp:
            all_data.append(json.load(fp))

    df_res = pd.DataFrame(all_data)
    print(df_res[['dataset', 'model', 'final_test_accuracy', 'macro_f1', 'num_params', 'training_time']], flush=True)

    # 1. Overall Accuracy Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_res, x='dataset', y='final_test_accuracy', hue='model', palette='tab10')
    plt.title("So Sánh Độ Chính Xác (Test Accuracy) Của 4 Kiến Trúc CNN Trên 3 Tập Dữ Liệu", fontweight='bold', fontsize=14)
    plt.xlabel("Tập Dữ Liệu", fontweight='bold')
    plt.ylabel("Test Accuracy", fontweight='bold')
    plt.ylim(0.6, 1.02)
    plt.legend(title="Kiến Trúc", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/overall_accuracy_comparison.png', dpi=300)
    plt.close()

    # 2. Number of Parameters Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_res, x='dataset', y='num_params', hue='model', palette='viridis')
    plt.title("So Sánh Quy Mô Tham Số (Trainable Parameters) Giữa Các Mô Hình", fontweight='bold', fontsize=14)
    plt.xlabel("Tập Dữ Liệu", fontweight='bold')
    plt.ylabel("Số Lượng Tham Số (Scale Log)", fontweight='bold')
    plt.yscale('log')
    plt.legend(title="Kiến Trúc", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/overall_parameters_comparison.png', dpi=300)
    plt.close()

    # 3. Accuracy vs Parameters (Pareto Frontier Analysis)
    plt.figure(figsize=(10, 6))
    markers = {'CIFAR-10': 'o', 'MNIST': 's', 'Diabetes CDC': '^'}
    for _, row in df_res.iterrows():
        plt.scatter(row['num_params'], row['final_test_accuracy'],
                    s=140, marker=markers.get(row['dataset'], 'o'), alpha=0.85,
                    label=f"{row['dataset']} - {row['model']}")
        plt.annotate(f"{row['model'].split()[0]} ({row['dataset'][:4]})",
                     (row['num_params'], row['final_test_accuracy']),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.xscale('log')
    plt.title("Đường Cong Đánh Đổi Pareto: Năng Lực Biểu Diễn (Accuracy) vs. Chi Phí Tham Số", fontweight='bold', fontsize=13)
    plt.xlabel("Số Lượng Tham Số Mô Hình (Log Scale)", fontweight='bold')
    plt.ylabel("Độ Chính Xác Trên Tập Kiểm Thử (Test Accuracy)", fontweight='bold')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('figures/accuracy_parameters_pareto.png', dpi=300)
    plt.close()

    # 4. Overall Heatmap
    pivot = df_res.pivot_table(index='dataset', columns='model', values='final_test_accuracy')
    plt.figure(figsize=(11, 4))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu', cbar_kws={'label': 'Accuracy'}, linewidths=1.2)
    plt.title("Ma Trận Hiệu Năng Test Accuracy: Dataset × Model Architecture", fontweight='bold', fontsize=13)
    plt.xlabel("Kiến Trúc Mô Hình", fontweight='bold')
    plt.ylabel("Tập Dữ Liệu", fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/overall_heatmap.png', dpi=300)
    plt.close()

    print("==> All figures generated successfully in figures/", flush=True)


if __name__ == '__main__':
    t_start = time.time()
    run_cifar10()
    run_mnist()
    run_diabetes()
    generate_overall_figures()
    print(f"\n==================================================================", flush=True)
    print(f"ALL 12 EXPERIMENTS COMPLETED IN {time.time() - t_start:.2f} SECONDS", flush=True)
    print(f"==================================================================", flush=True)
