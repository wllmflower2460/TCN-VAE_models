# TCN-VAE Trained Models

Trained Time Convolutional Network Variational Autoencoder models for EdgeInfer deployment.

## Files

- `best_tcn_vae_57pct.pth` - **BEST MODEL** - 57.68% validation accuracy (4.4MB)
- `full_tcn_vae_for_edgeinfer.pth` - Complete TCN-VAE model, 12% accuracy (4.4MB)
- `tcn_encoder_for_edgeinfer.pth` - TCN encoder only (1.7MB)  
- `model_config.json` - Model configuration and metadata

## Training Details

- **Best model**: 57.68% validation accuracy (August 29, 2025)
- Training system: GPU server (gpusrv) with RTX 2060
- Extended training: 24 epochs with early stopping
- Target deployment: EdgeInfer on Raspberry Pi with Hailo accelerator

## Training Progress

- Initial models: ~12% accuracy  
- Extended training: 48% → 54% → 57.68% accuracy
- Training collapsed at epoch 5 due to gradient explosion but best checkpoint saved

## Usage

These models are ready for integration into EdgeInfer service on Raspberry Pi.

## Model Configuration

See `model_config.json` for detailed model architecture and parameters.