# Hailo Feasibility Constraints (Deployment Contract)

This document defines the hardware and software constraints that TCN-VAE models must satisfy for successful deployment on Hailo-8 accelerators.

## Hardware Platform Requirements

### Target Hardware
- **Edge Device**: Raspberry Pi 5 (8GB RAM recommended)
- **Accelerator**: Hailo-8 AI processor (26 TOPS)
- **Storage**: High-speed microSD (Class 10 or better)
- **Power**: 15-20W total system power budget

### Performance Targets
- **Inference Latency**: p95 < 50ms per IMU window
- **Throughput**: ≥20 windows/sec sustained
- **Memory Usage**: <512MB Pi RAM for inference
- **Power Consumption**: <5W additional for Hailo processing

## Model Architecture Constraints

### Input Specifications
- **Shape**: Static input shape [1, 100, 9] (batch, time, channels)
- **Data Type**: Float32 input, Int8 quantized inference
- **Channels**: Fixed order [ax, ay, az, gx, gy, gz, mx, my, mz]
- **Temporal**: 100 timesteps @ 100Hz sampling rate

### Output Specifications
- **Latent Space**: Fixed 64-dimensional embeddings
- **Motif Scores**: 12 motif probability scores
- **Data Type**: Float32 output from Int8 inference
- **Range**: Latent [-4, +4], motif scores [0, 1]

### Architecture Limitations
- **No Dynamic Control Flow**: Avoid conditional branches, loops
- **Static Shapes**: All tensor dimensions must be compile-time constants
- **Supported Operations**: Limited to Hailo-compatible operation set
- **Memory Layout**: Contiguous tensors, aligned memory access

## Quantization Requirements

### Calibration Data
- **Sample Size**: Minimum 1000 representative IMU windows
- **Distribution**: Cover full range of target activities
- **Quality**: Clean, artifact-free sensor data
- **Format**: Normalized according to training parameters

### Quantization Process
- **Method**: Post-training quantization (PTQ) with calibration
- **Precision**: INT8 weights and activations
- **Calibration**: Representative dataset spanning operational modes
- **Validation**: Cosine similarity >0.95 vs FP32 reference

## Normalization Contract

### Critical Requirements
- **Per-Channel Statistics**: Exact μ/σ values from training
- **Channel Order**: Fixed mapping [0:ax, 1:ay, ..., 8:mz]
- **Preprocessing**: Z-score normalization applied at inference
- **Parity Validation**: Match training normalization exactly

### Implementation
```python
# Required normalization (must match training exactly)
normalized_channel = (raw_channel - channel_mean) / channel_std
```

### Validation Criteria
- **Training Parity**: Cosine similarity >0.99 ONNX vs PyTorch
- **Inference Parity**: Cosine similarity >0.95 Hailo vs ONNX
- **End-to-End**: Functional equivalence with EdgeInfer API

## Export and Compilation Constraints

### ONNX Export Requirements
- **Opset Version**: 11 (Hailo-compatible)
- **Static Shapes**: No dynamic dimensions
- **Operation Support**: Hailo-supported ops only
- **Batch Dimension**: Static batch size = 1

### Hailo DFC Compilation
- **Target**: hailo8 architecture
- **Optimization**: Performance vs accuracy tradeoff
- **Quantization**: INT8 with calibration dataset
- **Memory**: Fit within Hailo-8 memory constraints

### Validation Pipeline
1. **ONNX Validation**: Verify against PyTorch reference
2. **HEF Compilation**: Successful DFC compilation
3. **Runtime Loading**: HailoRT model loading
4. **Inference Testing**: End-to-end latency validation
5. **Accuracy Testing**: Cosine similarity thresholds

## Deployment Integration

### EdgeInfer Service Contract
- **API Endpoint**: POST /infer
- **Input Format**: JSON with "x": [[float]*9]*100
- **Output Format**: {"latent": [float]*64, "motif_scores": [float]*12}
- **Error Handling**: Graceful fallback to stub responses

### Docker Integration
- **Service Name**: hailo-inference
- **Port**: 9000
- **Device Mapping**: /dev/hailo0
- **Health Check**: /healthz endpoint

### Feature Flag Support
- **Environment**: USE_REAL_MODEL=true/false
- **Backend URL**: MODEL_BACKEND_URL=http://hailo-inference:9000/infer
- **Fallback**: Deterministic stub responses when disabled
- **Rollback**: Instant feature flag disable capability

## Compliance Validation

### Pre-Deployment Checklist
- [ ] Model architecture uses only Hailo-supported operations
- [ ] Input/output shapes are static and documented
- [ ] Normalization parameters exactly match training
- [ ] Calibration dataset representative of deployment data
- [ ] ONNX export successful with opset 11
- [ ] DFC compilation produces valid .hef file
- [ ] HailoRT loading successful on target hardware
- [ ] Inference latency meets p95 < 50ms requirement
- [ ] Accuracy parity validated with cosine similarity
- [ ] EdgeInfer integration tested end-to-end

### Performance Validation
- [ ] Sustained throughput ≥20 windows/sec
- [ ] Memory usage <512MB on Pi
- [ ] Power consumption <5W additional
- [ ] Thermal stability <70°C under load
- [ ] Network latency <10ms EdgeInfer → sidecar

### Production Readiness
- [ ] Feature flag rollback tested
- [ ] Error handling graceful degradation
- [ ] Monitoring and metrics collection
- [ ] Documentation complete and current
- [ ] Version compatibility matrix validated

## Troubleshooting Common Issues

### Compilation Failures
- **Dynamic Shapes**: Ensure all dimensions are static
- **Unsupported Ops**: Review Hailo operation compatibility
- **Memory Constraints**: Reduce model size or complexity
- **Calibration**: Provide representative calibration data

### Runtime Issues
- **Device Access**: Verify /dev/hailo0 permissions
- **Memory Allocation**: Check Hailo memory management
- **Driver Compatibility**: Ensure HailoRT version compatibility
- **Performance**: Profile inference pipeline bottlenecks

### Accuracy Issues
- **Normalization**: Verify exact μ/σ parameter matching
- **Quantization**: Check calibration data quality
- **Export**: Validate ONNX conversion accuracy
- **Integration**: Test end-to-end pipeline parity

This contract ensures TCN-VAE models deployed via `hailo_pipeline` meet all technical requirements for successful EdgeInfer integration on Raspberry Pi + Hailo-8 platforms.