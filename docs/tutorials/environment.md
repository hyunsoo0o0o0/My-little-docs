---
title: Environmental Data Analysis
layout: default
parent: Tutorials
nav_order: 3
---

# Environmental Data Analysis Tutorial

This tutorial demonstrates sea surface temperature (SST) modeling and extreme value analysis using KIOST-SST-EVL for environmental research applications.

## Overview

**Estimated time**: 30 minutes  
**Difficulty**: Beginner-Intermediate  
**Prerequisites**: Python 3.9+, Basic NumPy/Pandas knowledge

## What You'll Learn

- Load and preprocess SST satellite data
- Apply extreme value theory for tail analysis
- Model rare events using loss functions
- Visualize temporal and spatial patterns

## Dataset

This tutorial uses sample SST data from NOAA OI SST V2. Download the dataset:

```bash
# Download sample SST dataset
wget https://example.com/sstdv-docs/sample-sst-data.nc
```

## Step 1: Setup Environment

```bash
# Clone the repository
git clone https://github.com/SSTDV-Project/KIOST-SST-EVL.git
cd KIOST-SST-EVL
pip install -e .

# Verify installation
python -c "import kiost_sst_evl; print('KIOST-SST-EVL installed successfully')"
```

## Step 2: Load and Explore SST Data

```python
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import kiost_sst_evl as sst

# Load SST data
data = xr.open_dataset('./sample-sst-data.nc')
sst_data = data['sst']

print(f"Dataset shape: {sst_data.shape}")
print(f"Time range: {data.time.min().values} to {data.time.max().values}")
print(f"Spatial extent: {data.lat.min().values:.1f}°N to {data.lat.max().values:.1f}°N")

# Basic statistics
print(f"\nSST Statistics:")
print(f"  Mean: {sst_data.mean().values:.2f}°C")
print(f"  Std: {sst_data.std().values:.2f}°C")
print(f"  Min: {sst_data.min().values:.2f}°C")
print(f"  Max: {sst_data.max().values:.2f}°C")
```

## Step 3: Extreme Value Analysis

```python
from kiost_sst_evl.extremes import ExtremeValueAnalyzer

# Initialize analyzer
analyzer = ExtremeValueAnalyzer(threshold_percentile=95)

# Fit GPD to exceedances
result = analyzer.fit(sst_data.values.flatten())

print("Extreme Value Analysis Results:")
print(f"  Threshold (95th percentile): {result.threshold:.2f}°C")
print(f"  Number of exceedances: {result.n_exceedances}")
print(f"  Shape parameter (ξ): {result.xi:.4f}")
print(f"  Scale parameter (σ): {result.sigma:.4f}")
print(f"  Return level (100-year): {result.return_level(100):.2f}°C")
```

## Step 4: Train EVL Model

```python
from kiost_sst_evl.models import ExtremeValueLossModel

# Initialize model
model = ExtremeValueLossModel(
    backbone='resnet50',
    loss_type='evt_loss',
    extreme_threshold=result.threshold
)

# Train with EVL (Extreme Value Loss)
train_loader = sst.create_dataloader(
    './data/sst_train.nc',
    batch_size=32,
    shuffle=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    for batch in train_loader:
        sst_obs, mask = batch
        loss = model.compute_loss(sst_obs, mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/20, Loss: {loss.item():.4f}")
```

## Step 5: Visualize Results

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: SST time series
ax1 = axes[0, 0]
ax1.plot(data.time, sst_data.mean(dim=['lat', 'lon']), alpha=0.7)
ax1.axhline(y=result.threshold, color='r', linestyle='--', label='95th percentile')
ax1.set_xlabel('Time')
ax1.set_ylabel('SST (°C)')
ax1.set_title('Mean Sea Surface Temperature Over Time')
ax1.legend()

# Plot 2: Spatial distribution
ax2 = axes[0, 1]
sst_data.isel(time=0).plot(ax=ax2, cmap='RdBu_r')
ax2.set_title('SST Spatial Distribution (First Time Step)')

# Plot 3: Distribution of extremes
ax3 = axes[1, 0]
exceedances = sst_data.values[sst_data.values > result.threshold]
ax3.hist(exceedances, bins=50, density=True, alpha=0.7, label='Exceedances')
x = np.linspace(result.threshold, exceedances.max(), 100)
ax3.plot(x, result.gpd.pdf(x - result.threshold), 'r-', label='Fitted GPD')
ax3.set_xlabel('SST (°C)')
ax3.set_ylabel('Density')
ax3.set_title('Distribution of SST Exceedances')
ax3.legend()

# Plot 4: Return levels
ax4 = axes[1, 1]
return_periods = [1, 2, 5, 10, 20, 50, 100]
return_levels = [result.return_level(rp) for rp in return_periods]
ax4.loglog(return_periods, return_levels, 'bo-')
ax4.set_xlabel('Return Period (years)')
ax4.set_ylabel('Return Level (°C)')
ax4.set_title('Return Level Plot')
ax4.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('sst_analysis_results.png', dpi=150)
```

## Expected Results

After completing this tutorial, you should have:

- Loaded and visualized SST satellite data
- Identified extreme temperature events using EVT
- Trained a model with Extreme Value Loss
- Generated publication-ready figures

## Applications

This pipeline can be applied to:

- Marine ecosystem monitoring
- Climate change detection
- Extreme weather event prediction
- Oceanographic research

## Next Steps

- [KIOST-SST-EVL Documentation](../documentations/vcm.html) - Complete API reference
- Explore integration with [python-fluid-simulation](../documentations/vcm.html) for coupled ocean modeling
- Apply similar techniques to other environmental variables (chlorophyll, salinity)

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{kiost2024sst,
  title={Sea Surface Temperature Modeling with Extreme Value Theory},
  author={SSTDV-Project},
  year={2024},
  institution={KIOST}
}
```
