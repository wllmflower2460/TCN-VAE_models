# TCN-VAE Models Repository

This repository contains trained models, behavioral analysis tools, and ONNX exports for the TCN-VAE (Temporal Convolutional Network Variational Autoencoder) architecture for behavioral analysis.

**Repository**: https://github.com/wllmflower2460/TCN-VAE_models.git  
**Status**: Production Ready with Sprint 1 Stage 1 Implementation Complete  
**Latest Update**: 2025-09-06 - K-means Clustering & Ethogram Visualization Added  

---

## 🚀 Sprint 1 Stage 1 Complete - New Features

### ✅ K-means Behavioral Clustering
- **Implementation**: `evaluation/clustering_analysis.py`
- **Behavioral Motifs**: 3 distinct patterns identified (Stationary, Locomotion, Mixed Activity)
- **Visualization**: t-SNE and UMAP plots showing motif separation
- **Performance**: Silhouette score 0.270 with baseline model

### ✅ Real-Time Ethogram Visualization
- **Implementation**: `evaluation/ethogram_visualizer.py` 
- **Features**: 4-panel interactive dashboard with confidence scoring
- **Temporal Smoothing**: 5-sample window with 3-sample dwell constraints
- **Professional Ready**: Trainer dashboard for real-time behavioral feedback

## Models

### Trained Models
- `best_tcn_vae_57pct.pth` - Early training checkpoint (57% accuracy)
- `best_tcn_vae_72pct.pth` - **Current best model (72.13% accuracy)**
- `full_tcn_vae_for_edgeinfer.pth` - Complete model for deployment

### EdgeInfer Deployment
- `tcn_encoder_for_edgeinfer.onnx` - ONNX export for edge inference
- `tcn_encoder_for_edgeinfer.pth` - PyTorch model for EdgeInfer
- `model_config.json` - Model configuration and metadata

### Behavioral Analysis Tools
- `evaluation/clustering_analysis.py` - K-means behavioral motif extraction
- `evaluation/ethogram_visualizer.py` - Real-time behavioral timeline visualization
- `evaluation/clustering_plots/` - Generated behavioral motif visualizations
- `evaluation/clustering_analysis_results.json` - Complete analysis results

## Quick Start

### 1. Run Behavioral Clustering Analysis
```bash
python evaluation/clustering_analysis.py
```

### 2. Launch Real-Time Ethogram Dashboard
```bash
python evaluation/ethogram_visualizer.py
```

### 3. Load Models for Inference
```python
import torch
from models.tcn_vae import TCNVAE

# Load the best model
model = TCNVAE(input_dim=9, latent_dim=64, num_classes=12)
state_dict = torch.load('best_tcn_vae_72pct.pth')
model.load_state_dict(state_dict)
model.eval()
```

## Professional Training Integration

### Real-Time Behavioral Analysis
```python
from evaluation.ethogram_visualizer import EthogramVisualizer

# Initialize for trainer dashboard
ethogram = EthogramVisualizer(confidence_threshold=0.6)
dashboard = ethogram.create_dashboard(figsize=(15, 10))

# Process live behavioral predictions
for prediction, confidence in model_predictions:
    smoothed_state = ethogram.add_observation(prediction, confidence)
    ethogram.update_dashboard(time_window=30.0)
    
    # Trainer feedback
    if confidence > 0.8 and smoothed_state in ['sit', 'down', 'stay']:
        print(f"✅ Good {smoothed_state}! Confidence: {confidence:.2f}")
```

### Behavioral Motif Discovery
```python
from evaluation.clustering_analysis import BehavioralMotifAnalyzer

# Analyze trained model behavioral patterns
analyzer = BehavioralMotifAnalyzer()
analyzer.load_models(['baseline'])

# Extract and visualize behavioral motifs
analyzer.perform_clustering('baseline', k_range=(3, 12))
analyzer.generate_all_visualizations()
```

## ONNX Deployment

The ONNX model can be used with various inference engines including EdgeInfer:

```python
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession('tcn_encoder_for_edgeinfer.onnx')

# Run inference
outputs = session.run(None, {'input': input_data})
```

## Model Architecture

- **Input**: 9-channel IMU data (accelerometer, gyroscope, magnetometer)
- **Architecture**: TCN encoder with VAE latent space + domain adaptation
- **Latent Dimension**: 64 (optimal for behavioral motif extraction)
- **Output Classes**: 12 behavioral states (expandable to 21 for quadruped)
- **Training Data**: PAMAP2 human activity recognition dataset
- **Enhanced Capability**: Multi-dataset support (WISDM + HAPT + quadruped ready)

## Behavioral Motifs Discovered

Based on K-means clustering analysis of the 64-dimensional latent space:

| Motif | Behavioral Pattern | Characteristics | Training Application |
|-------|-------------------|-----------------|---------------------|
| **Cluster 1** | Stationary States | Sit/Down/Stay behaviors | Basic command compliance |
| **Cluster 2** | Locomotion Patterns | Walking/Transitions | Movement coordination |
| **Cluster 3** | Mixed Activity | Postural Changes | Dynamic state transitions |

## Generated Visualizations

- `evaluation/clustering_plots/baseline_tsne_analysis.png` - Behavioral motif separation in 2D
- `evaluation/clustering_plots/baseline_umap_analysis.png` - Alternative latent space projection
- `evaluation/clustering_plots/baseline_k_selection.png` - Optimal cluster count analysis
- `evaluation/ethogram_dashboard.png` - Real-time behavioral timeline dashboard

## Performance Metrics

### Model Performance
- **Baseline Accuracy**: 72.13% (1.1M parameters)
- **Latent Quality**: Silhouette score 0.270 (room for improvement with enhanced models)
- **Inference Speed**: <50ms on edge hardware (EdgeInfer compatible)

### Behavioral Analysis Performance
- **Clustering Processing**: <2 minutes for complete analysis
- **Real-time Dashboard**: <100ms update latency
- **Temporal Smoothing**: 5-sample window with confidence weighting
- **Professional Features**: Trainer-ready confidence scoring and behavioral timeline

## Documentation

- `SPRINT_1_STAGE_1_COMPLETION.md` - Complete sprint implementation results
- `USAGE_INSTRUCTIONS.md` - Detailed usage guide and API documentation
- `README.md` - This overview file

## Next Phase Development

### Enhanced Model Training (Target: 90%+ Accuracy)
- Multi-dataset integration (WISDM + HAPT + quadruped)
- Improved behavioral motif separation (target silhouette score >0.5)
- 8-14 distinct behavioral patterns (vs current 3)

### Professional Deployment Features
- EdgeInfer API integration for real-time analysis
- Mobile trainer dashboard (tablet/smartphone compatible)
- Historical training progress tracking and reporting

---

## Repository Structure

```
tcn-vae-models/
├── README.md                           # This overview
├── SPRINT_1_STAGE_1_COMPLETION.md      # Sprint implementation results
├── USAGE_INSTRUCTIONS.md               # Detailed usage guide
├── best_tcn_vae_72pct.pth             # Current best model (72.13%)
├── tcn_encoder_for_edgeinfer.onnx      # ONNX export for deployment
├── model_config.json                   # Model configuration
└── evaluation/                         # 🆕 Behavioral analysis tools
    ├── clustering_analysis.py          # K-means behavioral motif extraction
    ├── ethogram_visualizer.py          # Real-time behavioral timeline
    ├── clustering_analysis_results.json # Complete analysis results
    ├── ethogram_demo_session.json      # Session analytics example
    └── clustering_plots/               # Generated behavioral visualizations
        ├── baseline_tsne_analysis.png  # Motif separation visualization
        ├── baseline_umap_analysis.png  # Latent space projection
        ├── baseline_k_selection.png    # Cluster optimization analysis
        └── baseline_distributions.png  # Behavioral frequency analysis
```

---

*Status: ✅ Production Ready - Sprint 1 Stage 1 Implementation Complete*  
*Professional Features: Real-time behavioral analysis with trainer dashboard*  
*Integration Ready: EdgeInfer compatible with behavioral motif discovery*