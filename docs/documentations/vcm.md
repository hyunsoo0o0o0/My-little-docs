---
title: VCM
layout: default
parent: Software References
nav_order: 3
---

# VCM: Variable Coverage Model

A framework for generating synthetic data with controllable coverage properties, ensuring generated samples span the full distribution of interest.

## Overview

VCM addresses a common problem in synthetic data generation: ensuring that rare cases, edge cases, and corner cases are adequately represented in the generated dataset. Unlike standard generative models that optimize for overall fidelity, VCM explicitly models and controls the coverage of the data distribution.

**Repository**: [GitHub](https://github.com/SSTDV-Project/VCM)  
**License**: MIT

## Installation

```bash
pip install vcm-model
```

Or from source:

```bash
git clone https://github.com/SSTDV-Project/VCM.git
cd VCM
pip install -e .
```

## Quick Start

```python
import vcm
import numpy as np

# Initialize VCM with coverage constraints
model = vcm.VCM(
    coverage_target=0.95,      # Cover 95% of real distribution
    min_group_size=50,         # Minimum samples per mode
    diversity_weight=0.3       # Balance coverage vs fidelity
)

# Train on real data
model.fit(real_data, epochs=100)

# Generate with guaranteed coverage
synthetic_data = model.generate(n_samples=1000, coverage='guaranteed')

# Check coverage
coverage = vcm.metrics.coverage(real_data, synthetic_data)
print(f"Actual coverage: {coverage:.2%}")
```

## Core Concepts

### Coverage Definition

VCM defines coverage as the proportion of the real data distribution for which generated samples exist within a specified distance:

```
coverage(D_real, D_gen) = |{x ∈ D_real : ∃ y ∈ D_gen with dist(x,y) < ε}| / |D_real|
```

### Coverage Modes

| Mode | Behavior |
|------|----------|
| `guaranteed` | Ensures minimum coverage, may sacrifice fidelity |
| `balanced` | Equal weight to coverage and fidelity |
| `fidelity` | Optimizes for quality, best-effort coverage |

## API Reference

### VCM Class

```python
from vcm import VCM

model = VCM(
    coverage_target=0.95,      # Target coverage (0 to 1)
    min_group_size=100,        # Minimum samples per mode
    diversity_weight=0.5,     # Weight for diversity loss
    latent_dim=128,            # Latent space dimension
    hidden_dims=[256, 512],    # Generator architecture
    device='cuda'              # 'cuda' or 'cpu'
)
```

#### Methods

**fit(data, epochs=100, batch_size=64)**

Train the model on real data.

```python
model.fit(
    data,                     # numpy array or torch.Tensor
    epochs=100,
    batch_size=64,
    val_split=0.2,            # Fraction for validation
    early_stopping=True        # Stop if validation loss plateaus
)
```

**generate(n_samples, coverage='balanced')**

Generate synthetic samples.

```python
synthetic = model.generate(
    n_samples=1000,
    coverage='guaranteed',    # 'guaranteed', 'balanced', or 'fidelity'
    temperature=1.0           # Sampling temperature
)
```

**coverage_analysis(data_gen)**

Analyze coverage of generated data against training distribution.

```python
analysis = model.coverage_analysis(synthetic_data)

print(f"Overall coverage: {analysis['overall']:.2%}")
print(f"Per-mode coverage: {analysis['per_mode']}")
print(f"Uncovered regions: {analysis['gaps']}")
```

### CoverageMetrics

```python
from vcm.metrics import CoverageMetrics

metrics = CoverageMetrics()

# Compute various coverage metrics
results = metrics.compute(
    real=real_data,
    synthetic=synthetic_data,
    methods=['coverage', 'kNN', 'precision', 'recall']
)

# methods available: 'coverage', 'kNN', 'precision', 'recall', 'fid', 'density'
```

| Metric | Description | Range |
|--------|-------------|-------|
| `coverage` | Proportion of real data covered | 0-1 |
| `kNN` | k-nearest neighbor overlap | 0-1 |
| `precision` | Quality of generated samples | 0-1 |
| `recall` | Ability to capture rare cases | 0-1 |
| `FID` | Fréchet Inception Distance | 0-∞ |
| `density` | Local density ratio | 0-∞ |

## Examples

### Example 1: Guaranteed Coverage for Rare Events

```python
import vcm
import numpy as np

# Simulated medical data with rare conditions
data = np.load('medical_data.npy')
print(f"Rare events in data: {np.sum(data[:, -1] > 0) / len(data):.2%}")

# Initialize with high coverage target
model = vcm.VCM(coverage_target=0.99, min_group_size=100)

# Train
model.fit(data, epochs=200)

# Generate
synthetic = model.generate(n_samples=5000, coverage='guaranteed')

# Verify rare events are captured
rare_in_real = np.sum(data[:, -1] > 0) / len(data)
rare_in_synthetic = np.sum(synthetic[:, -1] > 0) / len(synthetic)
print(f"Rare events - Real: {rare_in_real:.2%}, Synthetic: {rare_in_synthetic:.2%}")
```

### Example 2: Balancing Coverage and Quality

```python
# Compare different coverage modes
modes = ['guaranteed', 'balanced', 'fidelity']
results = {}

for mode in modes:
    synthetic = model.generate(n_samples=1000, coverage=mode)
    results[mode] = {
        'coverage': metrics.coverage(real_data, synthetic),
        'fid': metrics.fid(real_data, synthetic),
        'precision': metrics.precision(real_data, synthetic)
    }

print("\nMode Comparison:")
print(f"{'Mode':<12} {'Coverage':>10} {'FID':>10} {'Precision':>10}")
print("-" * 44)
for mode, vals in results.items():
    print(f"{mode:<12} {vals['coverage']:>10.2%} {vals['fid']:>10.2f} {vals['precision']:>10.2%}")
```

### Example 3: Targeted Generation for Undersampled Regions

```python
# Identify undersampled regions in real data
coverage_map = metrics.region_analysis(real_data, synthetic_data)

# Generate more samples for gap regions
gap_regions = coverage_map[coverage_map['coverage'] < 0.5]
for region in gap_regions:
    samples = model.generate(
        n_samples=200,
        coverage='guaranteed',
        region_constraint=region['bounds']  # Generate in specific region
    )
    synthetic = np.vstack([synthetic, samples])
```

## Coverage Constraints

VCM supports hard constraints on coverage:

```python
model = vcm.VCM(
    coverage_target=0.95,
    constraints={
        'class_balance': True,     # Maintain class proportions
        'range_bounds': [-3, 3],   # Stay within data range
        'min_per_class': 100       # Minimum per class
    }
)
```

## Integration with Other Tools

VCM works well with:

- **DiffAM**: Generate appearance variations while maintaining coverage
- **DiffuAug**: Augmentation with coverage guarantees
- **EvalDistSim**: Verify generated data quality

```python
# Complete pipeline
import vcm, diffam, evaldistsim

# Generate with VCM
synthetic = vcm.generate(n_samples=1000, coverage='guaranteed')

# Verify with EvalDistSim
metrics = evaldistsim.compute(real, synthetic)

# Apply DiffAM variations
augmented = diffam.apply_variations(synthetic)
```

## Performance

| Operation | Time (10k samples) |
|-----------|-------------------|
| Training (per epoch) | ~30s |
| Generation | ~2s |
| Coverage analysis | ~5s |

## Dependencies

- torch >= 2.0
- numpy >= 1.21
- scipy >= 1.7
- scikit-learn >= 1.0
- matplotlib >= 3.5

## Citation

If VCM contributes to your research, please cite:

```bibtex
@article{vcm2024,
  title={VCM: Variable Coverage Model for Synthetic Data Generation},
  author={SSTDV-Project},
  year={2024},
  url={https://github.com/SSTDV-Project/VCM}
}
```

## See Also

- [DiffAM](diffam.html) - Appearance modeling
- [DiffuAug](vcm.html) - Diffusion augmentation (same page - integration example)
- [Breast Cancer Tutorial](../tutorials/breast-cancer.html) - Complete pipeline example
