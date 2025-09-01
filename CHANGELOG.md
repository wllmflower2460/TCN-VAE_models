# Changelog

All notable changes to the TCN-VAE models will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.0] - 2025-08-29

### Added
- First production-ready TCN encoder weights for EdgeInfer deployment
- Complete TCN-VAE model with 57.68% validation accuracy
- Multi-dataset training on PAMAP2, UCI-HAR, and TartanIMU
- Model metadata structure with versioned releases
- Normalization parameters and configuration files
- Evaluation metrics and training system documentation
- Hailo deployment feasibility constraints
- Release process with SHA256 checksums

### Model Performance
- **Validation Accuracy**: 57.68% (5x improvement from 12.59% baseline)
- **Training System**: GPUSrv RTX 2060, 24 epochs with early stopping
- **Model Size**: 4.4MB production-ready checkpoint
- **Target Platform**: Raspberry Pi + Hailo-8 edge deployment

### Architecture
- Input: 100-timestep × 9-channel IMU windows
- Hidden dimensions: [64, 128, 256] progressive feature extraction  
- Latent space: 64 dimensions
- Output: 12 motif scores for behavioral analysis

### Deployment
- Compatible with EdgeInfer service API contract
- Static shapes optimized for Hailo compilation
- Feature flag support for safe production rollout
- Per-channel normalization parameters for inference parity

## [Unreleased]
- Future model improvements and architecture variants
- Enhanced accuracy targets (>60% validation)
- Multi-modal fusion capabilities
- Active learning integration