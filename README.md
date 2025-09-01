# TCN-VAE Trained Models (Artifact + Metadata Repository)

Trained Time Convolutional Network Variational Autoencoder models for EdgeInfer deployment on Raspberry Pi + Hailo accelerator.

## Scope

This repo is the **source of truth for trained checkpoints** and their **metadata** (model card, config, normalization, eval metrics). **Export to ONNX** and **Hailo .hef compilation** are handled in the separate **hailo_pipeline** repository.

## Latest Release

- **Version**: v0.1.0 (2025-08-29)
- **Validation Accuracy**: 57.68%
- **Model Metadata**: see `models/tcn_encoder/v0.1.0/` for complete specifications
- **Checkpoints**: attached to [GitHub Release v0.1.0](https://github.com/wllmflower2460/TCN-VAE_models/releases/tag/v0.1.0) with `sha256sum.txt`

### Quick Links
- 📋 [Model Card](models/tcn_encoder/v0.1.0/model_card.md) - Performance and architecture details
- ⚙️ [Normalization](models/tcn_encoder/v0.1.0/normalization.json) - Critical deployment parameters
- 📊 [Eval Metrics](models/tcn_encoder/v0.1.0/eval_metrics.json) - Training validation results
- 📝 [Change Log](CHANGELOG.md) - Version history and improvements
- 🚀 [Releases](RELEASES.md) - Release process and asset management

## Deployment Target

EdgeInfer on Raspberry Pi with Hailo sidecar:
- **Input**: Fixed shape [100, 9] IMU windows (100 timesteps × 9 channels)
- **Output**: 64-dimensional latent embeddings + 12 motif scores
- **Performance**: <50ms inference latency, 20+ windows/sec throughput
- **Integration**: Via HailoRT sidecar at `http://hailo-inference:9000/infer`

## Repository Structure

```
models/
  tcn_encoder/
    v0.1.0/
      model_card.md          # Performance, architecture, usage
      model_config.json      # Original configuration (preserved)
      normalization.json     # Per-channel μ/σ, channel order
      eval_metrics.json      # Validation metrics, parity notes
CHANGELOG.md                 # Version history
RELEASES.md                  # Release process & commands
Hailo_Feasibility.md         # Hardware deployment constraints
```

## Model Performance

- **Best Model**: 57.68% validation accuracy (5x improvement from 12.59% baseline)
- **Training System**: GPUSrv with RTX 2060, 24 epochs with early stopping
- **Training Data**: Multi-dataset HAR (PAMAP2, UCI-HAR, TartanIMU)
- **Model Size**: 4.4MB production-ready checkpoint
- **Architecture**: TCN with [64, 128, 256] hidden dims → 64-dim latent space

## How This Repo Is Used

1. **Publishes trained checkpoints + metadata** (model card, config, normalization, eval metrics)
2. **ONNX export and .hef compilation** happen in **hailo_pipeline** (not here)
3. **Target deployment**: EdgeInfer on Raspberry Pi with Hailo sidecar
4. **Versioned releases**: Large binaries attached to GitHub Releases with SHA256 checksums

## Release Assets (v0.1.0)

Available as GitHub Release attachments:
- `tcn_encoder_for_edgeinfer.pth` - Encoder weights for deployment (1.6MB)
- `full_tcn_vae_for_edgeinfer.pth` - Complete TCN-VAE model (4.4MB)
- `best_tcn_vae_57pct.pth` - Best performing checkpoint (4.4MB)
- `sha256sum.txt` - Checksums for verification

## Usage Pipeline

1. **Download**: Get model assets from GitHub Releases
2. **Export**: Use `hailo_pipeline` to convert .pth → ONNX → .hef
3. **Deploy**: Run HailoRT sidecar with compiled .hef
4. **Integrate**: EdgeInfer calls sidecar via `MODEL_BACKEND_URL`

## Development Workflow

### Repository Boundaries
- **TCN-VAE_models** (this repo): Publishes **what to run** (weights + metadata, versioned)
- **hailo_pipeline**: Turns weights into **how to run on device** (ONNX → .hef → sidecar)
- **pisrv_vapor_docker**: Calls the sidecar over HTTP and surfaces health/metrics

### Model Versioning
Following [Semantic Versioning](https://semver.org/):
- **v0.1.0**: First production release (57.68% accuracy)
- **v0.x.x**: Backward-compatible improvements
- **v1.0.0**: Stable production API milestone

## Training Details (Historical)

- **Training Date**: August 29, 2025
- **Training Progress**: 12% → 48% → 54% → **57.68%** accuracy
- **Training Challenges**: Gradient explosion at epoch 5, but best checkpoint preserved
- **System**: GPUSrv RTX 2060 with extended training session
- **Datasets**: Multi-dataset approach with 13+ activity classes

## Hardware Requirements

See [Hailo_Feasibility.md](Hailo_Feasibility.md) for complete deployment constraints:
- Static input shape [100,9], fixed latent size 64
- Ops/activations chosen to be Hailo-friendly (no dynamic control flow)
- Quantization requires calibration IMU windows
- Export normalization (per-channel μ/σ) must match training exactly

## Integration Testing

Validated with:
- **EdgeInfer API**: Compatible with existing `/api/v1/analysis/motifs` endpoints
- **Feature Flags**: `USE_REAL_MODEL=true/false` support
- **Fallback**: Deterministic stub responses when sidecar unavailable
- **Performance**: <50ms inference latency on Pi + Hailo-8

---

*This repository provides the trained model artifacts and metadata for the EdgeInfer + Hailo deployment pipeline. For model compilation and deployment, see the [hailo_pipeline](https://github.com/wllmflower2460/hailo_pipeline) repository.*