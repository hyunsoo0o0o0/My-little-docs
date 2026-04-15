---
title: DiffAM
layout: default
parent: Software References
nav_order: 2
---

# DiffAM: Differentiable Appearance Modeling

A framework for learning, manipulating, and generating image appearances using differentiable models and deep learning.

## Overview

DiffAM enables researchers to extract interpretable appearance embeddings from images, generate synthetic variations, and manipulate appearance attributes in a differentiable manner. Built on PyTorch, it provides pretrained models and easy-to-use APIs for medical imaging, computer graphics, and computer vision applications.

**Repository**: [GitHub](https://github.com/SSTDV-Project/DiffAM)  
**Live Demo**: [sstdv-project.github.io/DiffAM](https://sstdv-project.github.io/DiffAM)  
**License**: MIT

## Installation

```bash
pip install diffam
```

Or from source:

```bash
git clone https://github.com/SSTDV-Project/DiffAM.git
cd DiffAM
pip install -e .
```

## Quick Start

```python
import diffam
import torch

# Load pretrained model
model = diffam.DiffAM(pretrained='medical_imaging')

# Extract appearance embeddings from image
image = torch.randn(1, 1, 256, 256)  # Batch of 1, 1 channel, 256x256
embeddings = model.extract_embeddings(image)

# Generate synthetic variation
synthetic = model.generate(embeddings, n_samples=5)

# Manipulate appearance
edited = model.edit_attributes(embeddings, attribute='contrast', strength=1.5)
```

## Core Concepts

### Appearance Embeddings

DiffAM represents images as compositions of:

1. **Content features** (C): What the image depicts (anatomy, objects)
2. **Appearance features** (A): How it looks (lighting, contrast, texture)

```
z = [c; a]  # Joint embedding
```

### Differentiable Operations

All operations support gradient flow, enabling:

- End-to-end training
- Appearance optimization
- Style transfer via gradient descent

## API Reference

### DiffAM Class

```python
from diffam import DiffAM

model = DiffAM(
    pretrained=None,           # 'medical_imaging', 'face', or None
    backbone='resnet50',        # Feature extractor architecture
    embedding_dim=512,          # Size of appearance embedding
    device='cuda'              # 'cuda' or 'cpu'
)
```

#### Methods

**extract_embeddings(image)**

Extract joint content-appearance embeddings.

```python
embeddings = model.extract_embeddings(
    image,                      # torch.Tensor [B, C, H, W]
    return_separate=True        # Return (content, appearance) separately
)
# Returns: torch.Tensor [B, embedding_dim]
```

**generate(condition, n_samples)**

Generate samples conditioned on embeddings.

```python
synthetic = model.generate(
    condition=embeddings,       # Content embeddings
    n_samples=5,               # Number to generate
    appearance_variation=0.1    # Randomness in appearance
)
# Returns: torch.Tensor [n_samples, C, H, W]
```

**edit_attributes(embeddings, attribute, strength)**

Manipulate specific appearance attributes.

```python
edited = model.edit_attributes(
    embeddings,                # Original embeddings
    attribute='contrast',      # 'contrast', 'brightness', 'texture', 'style'
    strength=1.5               # Manipulation strength (-2 to 2)
)
```

### Pretrained Models

```python
from diffam.pretrained import list_models, load_model

# List available models
models = list_models()
# ['medical_imaging_v1', 'face_v1', 'face_v2', 'general_v1']

# Load specific model
model = load_model('medical_imaging_v1')
```

| Model | Dataset | Best For |
|-------|---------|----------|
| `medical_imaging_v1` | Combined medical imaging | MRI, CT, X-ray |
| `face_v1` | FFHQ | Face images |
| `face_v2` | FFHQ + augmentation | Low-quality faces |
| `general_v1` | ImageNet | Natural images |

## Tutorial: Appearance Interpolation

```python
import diffam
import matplotlib.pyplot as plt

model = diffam.DiffAM(pretrained='medical_imaging')

# Extract embeddings from two images
emb1 = model.extract_embeddings(image1)
emb2 = model.extract_embeddings(image2)

# Interpolate appearances
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
interpolated = []

for alpha in alphas:
    mix_emb = (1 - alpha) * emb1 + alpha * emb2
    interp = model.generate(condition=mix_emb[:1], n_samples=1)
    interpolated.append(interp[0])

# Visualize
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for ax, img, alpha in zip(axes, interpolated, alphas):
    ax.imshow(img[0].cpu().detach(), cmap='gray')
    ax.set_title(f'α = {alpha}')
    ax.axis('off')

plt.savefig('appearance_interpolation.png')
```

## Tutorial: Style Transfer

```python
# Extract content and appearance separately
content = model.extract_embeddings(content_image, return_separate=True)[0]
_, appearance = model.extract_embeddings(style_image, return_separate=True)

# Combine
styled = model.generate(condition=content, appearance_reference=appearance)

# Save result
diffam.utils.save_image(styled[0], 'styled_output.png')
```

## Advanced Usage

### Custom Training

```python
from diffam.training import DiffAMTrainer

trainer = DiffAMTrainer(
    model=model,
    lr=1e-4,
    content_loss='perceptual',
    appearance_loss='contrastive'
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        losses = trainer.step(batch)
        trainer.backward(losses)
```

### Fine-tuning for New Domain

```python
# Freeze content encoder, fine-tune appearance modules
model.freeze_encoders()

# Add new appearance attributes
model.add_attribute('pathology_level', dim=64)

# Fine-tune on new data
trainer = DiffAMTrainer(model, lr=1e-5)
trainer.train(new_dataloader, epochs=10)
```

## Architecture

DiffAM consists of:

```
Content Encoder (E_c)     → Content embedding c
Appearance Encoder (E_a)  → Appearance embedding a
Generator (G)            → Image synthesis from (c, a)
Discriminator (D)        → Adversarial training (optional)
```

## Performance

| Operation | GPU | CPU |
|-----------|-----|-----|
| Embedding extraction (256×256) | ~10ms | ~80ms |
| Generation (256×256) | ~15ms | ~120ms |
| Full interpolation | ~50ms | ~400ms |

## Dependencies

- torch >= 2.0
- torchvision >= 0.15
- einops >= 0.6
- kornia >= 0.6
- matplotlib >= 3.5

## Citation

If DiffAM contributes to your research, please cite:

```bibtex
@article{diffam2024,
  title={DiffAM: Differentiable Appearance Modeling for Medical Imaging},
  author={SSTDV-Project},
  year={2024},
  url={https://github.com/SSTDV-Project/DiffAM}
}
```

## See Also

- [BrainShapeToolKit](brainshapetoolkit.html) - Combine with shape analysis
- [Brain Imaging Tutorial](../tutorials/brain-imaging.html) - Complete workflow
- [DiffuAug](vcm.html) - Related augmentation framework
