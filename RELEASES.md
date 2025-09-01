# Releases Process

This document outlines the process for creating and managing releases of TCN-VAE models.

## Release Workflow

### 1. Pre-Release Validation
Before creating a release, ensure:
- [ ] Model training completed with satisfactory metrics
- [ ] Model exported and tested on target platform
- [ ] Evaluation metrics documented in `models/tcn_encoder/vX.Y.Z/eval_metrics.json`
- [ ] Normalization parameters verified in `normalization.json`
- [ ] Model card updated with performance details

### 2. Version Tagging
```bash
# Create and push version tag
git tag v0.1.0
git push --tags
```

### 3. Build Release Assets
```bash
# Generate checksums for model files
sha256sum *.pth > sha256sum.txt

# Verify checksums
sha256sum -c sha256sum.txt
```

### 4. Create GitHub Release
1. Navigate to https://github.com/wllmflower2460/TCN-VAE_models/releases
2. Click "Create a new release"
3. Select the version tag (e.g., `v0.1.0`)
4. Add release title: "TCN-VAE Models v0.1.0 - EdgeInfer Production Ready"
5. Add release description (see template below)
6. Upload release assets:
   - `tcn_encoder_for_edgeinfer.pth`
   - `full_tcn_vae_for_edgeinfer.pth`
   - `best_tcn_vae_57pct.pth`
   - `sha256sum.txt`

### 5. Update Documentation
- [ ] Update model card with release asset links
- [ ] Update CHANGELOG.md with release notes
- [ ] Verify README.md points to latest release

## Release Template

### Release Title
`TCN-VAE Models vX.Y.Z - [Brief Description]`

### Release Description
```markdown
## TCN-VAE Models vX.Y.Z

### Model Performance
- **Validation Accuracy**: X.XX%
- **Model Size**: X.XMB
- **Target Platform**: Raspberry Pi + Hailo-8

### Assets
- `tcn_encoder_for_edgeinfer.pth` - Encoder weights for deployment (X.XMB)
- `full_tcn_vae_for_edgeinfer.pth` - Complete TCN-VAE model (X.XMB)
- `best_tcn_vae_57pct.pth` - Best performing checkpoint (X.XMB)
- `sha256sum.txt` - Checksums for verification

### Usage
These models are designed for deployment via the `hailo_pipeline` repository:
1. Export to ONNX format with fixed input shape [1, 100, 9]
2. Compile to Hailo .hef format using DFC
3. Deploy via HailoRT sidecar for EdgeInfer integration

### Verification
```bash
# Verify checksums
sha256sum -c sha256sum.txt
```

See `models/tcn_encoder/vX.Y.Z/` for detailed metadata and deployment specifications.
```

## Version Numbering

Following [Semantic Versioning](https://semver.org/):

- **MAJOR** version: Incompatible API/architecture changes
- **MINOR** version: New functionality, improved accuracy, backward compatible
- **PATCH** version: Bug fixes, metadata updates, backward compatible

### Examples
- `v0.1.0` - First production release
- `v0.2.0` - Architecture improvements, new training data
- `v0.1.1` - Metadata fixes, documentation updates
- `v1.0.0` - Stable production API, major milestone

## Release Checklist

### Pre-Release
- [ ] Model training validation complete
- [ ] Evaluation metrics documented
- [ ] Normalization parameters verified
- [ ] Model card updated
- [ ] CHANGELOG.md updated
- [ ] Version tag created

### Release Creation
- [ ] GitHub release created
- [ ] All model assets uploaded
- [ ] SHA256 checksums included
- [ ] Release description complete
- [ ] Links verified

### Post-Release
- [ ] Documentation updated with release links
- [ ] Integration testing with hailo_pipeline
- [ ] EdgeInfer deployment validation
- [ ] Performance benchmarks verified

## Asset Management

### File Naming Convention
- `tcn_encoder_for_edgeinfer.pth` - Encoder-only weights
- `full_tcn_vae_for_edgeinfer.pth` - Complete model
- `best_tcn_vae_[accuracy]pct.pth` - Best checkpoint by metric
- `sha256sum.txt` - Verification checksums

### Storage Guidelines
- **Large binaries** (.pth files): GitHub Release assets only
- **Small metadata** (.json, .md): Version controlled in git
- **Checksums**: Always included with releases
- **Documentation**: Keep synchronized with releases

## Integration Points

### hailo_pipeline Repository
The `hailo_pipeline` repository consumes these models:
- Downloads release assets by version tag
- Uses normalization.json for export parity
- References model_config.json for architecture
- Validates against eval_metrics.json thresholds

### EdgeInfer Service
The EdgeInfer service expects:
- Fixed input shape [100, 9] from normalization.json
- Latent dimension 64 from eval_metrics.json
- Motif scores count from model configuration
- Performance SLOs from deployment validation