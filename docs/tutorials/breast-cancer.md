---
title: Breast Cancer Detection
layout: default
parent: Tutorials
nav_order: 2
---

# Breast Cancer Detection Tutorial

This tutorial demonstrates how to build a breast cancer detection pipeline using synthetic data augmentation with DiffuAug and distribution similarity evaluation with EvalDistSim.

## Overview

**Estimated time**: 60 minutes  
**Difficulty**: Advanced  
**Prerequisites**: Python 3.9+, Familiarity with deep learning, Access to medical imaging data

## What You'll Learn

- Generate synthetic mammography data using DiffuAug
- Evaluate synthetic data quality with EvalDistSim
- Train a detection model with augmented dataset
- Compare performance against baseline

## Dataset

This tutorial uses a sample mammography dataset. Due to data privacy concerns, we provide synthetic data for demonstration:

```bash
# Download sample dataset
wget https://example.com/sstdv-docs/sample-mammo.zip
unzip sample-mammo.zip -d ./data/
```

## Step 1: Setup Environment

```bash
# Clone the repositories
git clone https://github.com/SSTDV-Project/DiffuAug.git
git clone https://github.com/SSTDV-Project/EvalDistSim.git

# Install dependencies
cd DiffuAug
pip install -e .
cd ../EvalDistSim
pip install -e .
```

## Step 2: Generate Synthetic Data with DiffuAug

```python
import diffaug
import numpy as np

# Load original dataset
original_data = np.load('./data/mammo_samples.npy')
labels = np.load('./data/mammo_labels.npy')

# Initialize DiffuAug
augmenter = diffaug.DiffuAug(
    model_type='ddpm',
    checkpoint='diffaug_pretrained.pth'
)

# Generate synthetic samples
n_synthetic = 500
synthetic_data = augmenter.generate(
    condition=labels,
    n_samples=n_synthetic,
    diffusion_steps=100
)

print(f"Generated {n_synthetic} synthetic mammograms")
print(f"Original shape: {original_data.shape}")
print(f"Synthetic shape: {synthetic_data.shape}")
```

## Step 3: Evaluate Distribution Similarity

```python
import evaldistsim

# Initialize evaluator
evaluator = evaldistsim.EvalDistSim()

# Compute distribution metrics
metrics = evaluator.compute_metrics(
    original=original_data,
    synthetic=synthetic_data,
    methods=['ks_test', 'wasserstein', 'fid']
)

print("Distribution Similarity Metrics:")
print(f"  Kolmogorov-Smirnov: {metrics['ks_test']:.4f}")
print(f"  Wasserstein Distance: {metrics['wasserstein']:.4f}")
print(f"  Fréchet Inception Distance: {metrics['fid']:.4f}")
```

## Step 4: Train Detection Model

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Combine original and synthetic data
X_train = np.concatenate([original_data, synthetic_data], axis=0)
y_train = np.concatenate([labels, labels[:n_synthetic]], axis=0)

# Create dataloaders
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.LongTensor(y_train)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define simple detection model
class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(64, 2)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Training loop
model = DetectionModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")
```

## Step 5: Evaluate Performance

```python
# Evaluate on test set
model.eval()
with torch.no_grad():
    test_outputs = model(torch.FloatTensor(X_test))
    predictions = test_outputs.argmax(dim=1)
    accuracy = (predictions == y_test).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")
```

## Expected Results

After completing this tutorial, you should have:

- 500 synthetic mammograms that pass distribution similarity tests
- A trained detection model with potential accuracy improvement from augmentation
- Quantitative metrics comparing original vs. synthetic data quality

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch size or use gradient checkpointing |
| Low FID score | Adjust diffusion steps or model architecture |
| Training instability | Reduce learning rate or increase warmup |

## Next Steps

- [Brain Imaging Tutorial](brain-imaging.html) - Apply similar techniques to brain imaging
- [EvalDistSim Reference](../documentations/vcm.html) - Deep dive into distribution evaluation
- [DiffuAug GitHub](https://github.com/SSTDV-Project/DiffuAug) - Latest updates and models

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{sstdv2024breast,
  title={Synthetic Data Augmentation for Breast Cancer Detection},
  author={SSTDV-Project},
  year={2024}
}
```
