# TCN Encoder — v0.1.0 (EdgeInfer)

**Purpose**: Encoder weights for EdgeInfer → Hailo export/compile in hailo_pipeline.  
**Input**: IMU window [100,9] (time × channels)  
**Output**: latent [64]  
**Training date**: 2025-08-29 (gpusrv)  
**Target deployment**: Raspberry Pi + Hailo sidecar

## Model Performance
- **Validation Accuracy**: 57.68%
- **Training System**: GPUSrv with RTX 2060
- **Training Duration**: 24 epochs with early stopping
- **Model Size**: 4.4MB (production-ready)

## Architecture
- **Hidden dims**: [64, 128, 256] - Progressive feature extraction
- **Latent space**: 64 dimensions
- **Sequence length**: 100 timesteps
- **Input channels**: 9 (3×accelerometer, 3×gyroscope, 3×magnetometer)

## Training Data
- **Datasets**: PAMAP2, UCI-HAR, TartanIMU
- **Activities**: 13+ activity classes
- **Window overlap**: 0.5
- **Normalization**: Per-channel z-score (see normalization.json)

## Artifacts (GitHub Release assets)
- `tcn_encoder_for_edgeinfer.pth` - Encoder-only weights (1.6MB)
- `full_tcn_vae_for_edgeinfer.pth` - Complete TCN-VAE model (4.4MB)
- `best_tcn_vae_57pct.pth` - Best performing model (4.4MB)
- `sha256sum.txt` - Checksums for verification

## Usage
This model is designed for deployment via the `hailo_pipeline` repository:
1. Export to ONNX format with fixed input shape [1, 100, 9]
2. Compile to Hailo .hef format using DFC
3. Deploy via HailoRT sidecar for EdgeInfer integration

See `normalization.json` and `model_config.json` for parity-critical details required for proper export and deployment.