---
title: BrainShapeToolKit
layout: default
parent: Software References
nav_order: 1
---

# BrainShapeToolKit

A Python toolkit for comprehensive brain shape analysis from medical imaging data.

## Overview

BrainShapeToolKit provides a unified interface for processing brain MRI data, extracting shape features, and performing statistical analysis. It is designed for neuroimaging researchers who need reproducible, well-documented shape analysis pipelines.

**Repository**: [GitHub](https://github.com/SSTDV-Project/BrainShapeToolKit)  
**Documentation**: This page  
**License**: MIT

## Installation

```bash
pip install brainshapetoolkit
```

Or from source:

```bash
git clone https://github.com/SSTDV-Project/BrainShapeToolKit.git
cd BrainShapeToolKit
pip install -e .
```

## Quick Start

```python
import brainshapetoolkit as bst

# Load and process brain mesh
processor = bst.MeshProcessor()
mesh = processor.load('brain_mesh.gii')
mesh_smooth = processor.smooth(mesh, iterations=10)

# Extract features
features = bst.ShapeFeatures().compute(mesh_smooth)
print(f"Surface area: {features['surface_area']}")
```

## Core Modules

### MeshProcessor

Handles loading, preprocessing, and mesh operations.

```python
from brainshapetoolkit import MeshProcessor

processor = MeshProcessor()

# Load mesh from various formats
mesh = processor.load('brain.gii')       # GIFTI
mesh = processor.load('brain.mesh')      # FreeSurfer
mesh = processor.load('brain.nii.gz')     # NIfTI volume

# Preprocessing
mesh_smooth = processor.smooth(mesh, iterations=10, lambda_decay=0.93)
mesh_decimated = processor.decimate(mesh, target_faces=50000)
mesh_registered = processor.register(mesh, template='fsaverage')
```

### ShapeFeatures

Extracts quantitative shape features.

```python
from brainshapetoolkit.features import ShapeFeatures

extractor = ShapeFeatures(features=['all'])

# Compute all features
features = extractor.compute(mesh)

# Compute specific features
features = extractor.compute(mesh, features=['surface_area', 'volume', 'thickness'])
```

| Feature | Description | Units |
|---------|-------------|-------|
| `surface_area` | Total surface area | mm² |
| `volume` | Enclosed volume | mm³ |
| `thickness` | Cortical thickness statistics | mm |
| `curvature` | Mean/gaussian curvature | 1/mm |
| `folding_index` | Gyral/sulcal folding pattern | - |
| `sphericity` | How spherical the shape is | - |

### SurfaceStatistics

Performs statistical analysis on shape features.

```python
from brainshapetoolkit.statistics import SurfaceStatistics

stats = SurfaceStatistics()

# Group analysis
results = stats.group_analysis(
    features_list=[cohort1_features, cohort2_features],
    groups=['control', 'patient'],
    covariate='age'
)

# Permutation testing
p_values = stats.permutation_test(features, n_permutations=10000)
```

## Supported Formats

| Format | Extension | Support Level |
|--------|-----------|---------------|
| GIFTI | .gii | Full |
| FreeSurfer | .surf, . curv | Full |
| NIfTI | .nii, .nii.gz | Volume only |
| OBJ | .obj | Mesh only |
| STL | .stl | Mesh only |

## Examples

### Example 1: Basic Shape Analysis

```python
import brainshapetoolkit as bst
import numpy as np

# Complete pipeline
processor = bst.MeshProcessor()
features = bst.ShapeFeatures()
stats = bst.SurfaceStatistics()

# Load and process
mesh = processor.load('subject001_brain.gii')
mesh_smooth = processor.smooth(mesh, iterations=5)

# Extract features
feat = features.compute(mesh_smooth)

# Statistical analysis
z_scores = stats.compute_zscores(feat, normative_data)
significant = stats.fdr_correction(z_scores, alpha=0.05)
```

### Example 2: Group Comparison

```python
# Load cohorts
control_meshes = [processor.load(f) for f in control_files]
patient_meshes = [processor.load(f) for f in patient_files]

# Extract features
control_features = [features.compute(m) for m in control_meshes]
patient_features = [features.compute(m) for m in patient_meshes]

# Group analysis
results = stats.group_analysis(
    features_list=control_features + patient_features,
    groups=['control']*len(control_features) + ['patient']*len(patient_features)
)

print(f"Significant differences: {results['significant_regions']}")
```

## API Reference

### Classes

| Class | Description |
|-------|-------------|
| `MeshProcessor` | Load, preprocess, and manipulate mesh data |
| `ShapeFeatures` | Extract quantitative shape descriptors |
| `SurfaceStatistics` | Statistical analysis of shape features |
| `SurfaceVisualizer` | Visualization utilities |
| `MeshTransform` | Geometric transformations |

### MeshProcessor Methods

```python
MeshProcessor.load(path)                    # Load mesh from file
MeshProcessor.smooth(mesh, iterations)     # Apply Laplacian smoothing
MeshProcessor.decimate(mesh, ratio)          # Reduce polygon count
MeshProcessor.register(mesh, template)       # Align to template
MeshProcessor.compute_normals(mesh)          # Compute vertex normals
MeshProcessor.fill_holes(mesh)               # Close small holes
```

## Performance

BrainShapeToolKit uses Numba JIT compilation for critical operations:

- Surface smoothing: ~100ms for 150k vertices
- Feature extraction: ~500ms for full feature set
- Group analysis (n=100): ~30s

## Dependencies

- numpy >= 1.21
- scipy >= 1.7
- nibabel >= 3.2
- numba >= 0.55
- h5py >= 3.0
- matplotlib >= 3.5

## Citation

If BrainShapeToolKit contributes to your research, please cite:

```bibtex
@article{brainshapetoolkit2024,
  title={BrainShapeToolKit: A Python Toolkit for Brain Shape Analysis},
  author={SSTDV-Project},
  year={2024},
  url={https://github.com/SSTDV-Project/BrainShapeToolKit}
}
```

## See Also

- [DiffAM](diffam.html) - Appearance modeling that can be combined with shape analysis
- [Brain Imaging Tutorial](../tutorials/brain-imaging.html) - Complete workflow example
