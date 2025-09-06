# TCN-VAE Behavioral Analysis - Usage Instructions

**Repository**: https://github.com/wllmflower2460/TCN-VAE_models.git  
**Sprint 1 Implementation**: K-means Clustering & Ethogram Visualization  
**Status**: Production Ready  

---

## Quick Start Guide

### Prerequisites
```bash
# Required Python packages
pip install torch numpy scikit-learn matplotlib seaborn pandas umap-learn
```

### 1. K-means Behavioral Clustering

#### Run Complete Analysis
```bash
cd /path/to/tcn-vae-models
python evaluation/clustering_analysis.py
```

#### Expected Output
```
🚀 Starting TCN-VAE Behavioral Motif Clustering Analysis
📋 Sprint 1, Task 1: K-means clustering implementation
============================================================
🔬 Behavioral Motif Analyzer initialized on cuda
✅ Loaded state dict for baseline
🤖 baseline: 1,104,240 parameters

🔍 Feature Extraction Phase
------------------------------
✅ Extracted 500 feature vectors (dim=64) from baseline

🎯 Clustering Analysis Phase  
------------------------------
🎯 Optimal K=3 with silhouette score 0.270

📊 Visualization Generation Phase
------------------------------
✅ Visualizations saved to evaluation/clustering_plots

🎉 Clustering analysis complete!
```

#### Generated Files
- `evaluation/clustering_analysis_results.json` - Analysis results
- `evaluation/clustering_plots/baseline_k_selection.png` - Silhouette analysis
- `evaluation/clustering_plots/baseline_tsne_analysis.png` - t-SNE motif visualization
- `evaluation/clustering_plots/baseline_umap_analysis.png` - UMAP projection
- `evaluation/clustering_plots/baseline_distributions.png` - Cluster frequencies

### 2. Real-Time Ethogram Visualization

#### Run Interactive Demo
```bash
python evaluation/ethogram_visualizer.py
```

#### Expected Output
```
🚀 TCN-VAE Ethogram Visualization System
📋 Sprint 1, Task 2: Real-time behavioral timeline visualization
🎬 Starting Real-Time Ethogram Demo
🎯 EthogramVisualizer initialized
📊 Generating synthetic behavioral sequence...
📊 Dashboard saved to evaluation/ethogram_dashboard.png
🎉 Real-time ethogram demo complete!
```

#### Generated Files
- `evaluation/ethogram_dashboard.png` - Complete 4-panel dashboard
- `evaluation/ethogram_demo_session.json` - Session analytics data

## Programming Interface

### K-means Clustering API

```python
from evaluation.clustering_analysis import BehavioralMotifAnalyzer

# Initialize analyzer
analyzer = BehavioralMotifAnalyzer()

# Load and analyze models
analyzer.load_models(['baseline'])  # Add 'enhanced', 'quadruped' when available

# Perform clustering analysis
analyzer.perform_clustering('baseline', k_range=(3, 12))

# Generate visualizations
analyzer.generate_all_visualizations()

# Export results
analyzer.save_analysis_results('my_analysis.json')
```

### Ethogram Visualization API

```python
from evaluation.ethogram_visualizer import EthogramVisualizer

# Initialize ethogram with custom settings
ethogram = EthogramVisualizer(
    confidence_threshold=0.6,    # Minimum confidence for state acceptance
    smoothing_window=5,          # Temporal smoothing window
    dwell_time_min=3,           # Minimum samples for state persistence
    max_history=1000            # Maximum observations to retain
)

# Create interactive dashboard
fig = ethogram.create_dashboard(figsize=(15, 10))
plt.ion()  # Enable interactive mode

# Real-time behavioral observations
for prediction, confidence in model_predictions:
    # Add observation with temporal smoothing
    smoothed_state = ethogram.add_observation(prediction, confidence)
    
    # Update dashboard every 5 observations
    if observation_count % 5 == 0:
        ethogram.update_dashboard(time_window=30.0)

# Generate session summary
summary = ethogram.generate_summary_statistics()
print(f"Session duration: {summary['observation_duration_seconds']:.1f}s")
print(f"Unique states: {summary['unique_states_observed']}")

# Save complete session data
ethogram.save_session_data('training_session.json')
```

## Integration with EdgeInfer

### Real-Time Model Integration

```python
# Example integration with live TCN-VAE inference
import requests
from evaluation.ethogram_visualizer import EthogramVisualizer

# Initialize ethogram for real-time training feedback
ethogram = EthogramVisualizer(confidence_threshold=0.6)
dashboard = ethogram.create_dashboard()

# EdgeInfer API integration loop
def process_live_inference():
    while training_active:
        # Get prediction from EdgeInfer
        response = requests.post('http://edgeinfer:8080/predict', 
                               json=sensor_data)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['predicted_state']
            confidence = result['confidence']
            
            # Process with ethogram
            smoothed_state = ethogram.add_observation(prediction, confidence)
            ethogram.update_dashboard(time_window=30.0)
            
            # Trainer feedback
            if confidence > 0.8 and smoothed_state in ['sit', 'down', 'stay']:
                print(f"✅ Good {smoothed_state}! Confidence: {confidence:.2f}")
```

### Batch Analysis Integration

```python
# Analyze historical training sessions
def analyze_training_session(session_file):
    analyzer = BehavioralMotifAnalyzer()
    
    # Load session data
    with open(session_file) as f:
        session_data = json.load(f)
    
    # Extract behavioral patterns
    predictions = session_data['predictions']
    confidences = session_data['confidences']
    
    # Run clustering analysis
    results = analyzer.analyze_behavioral_patterns(predictions, confidences)
    
    return {
        'dominant_behaviors': results['state_frequencies'],
        'transition_patterns': results['transition_matrix'],
        'session_quality': results['mean_confidence'],
        'behavioral_complexity': len(results['unique_states'])
    }
```

## Professional Training Applications

### Real-Time Training Dashboard

```python
class TrainerDashboard:
    def __init__(self):
        self.ethogram = EthogramVisualizer(
            confidence_threshold=0.7,  # Higher threshold for training
            smoothing_window=3,        # Faster response for immediate feedback
            dwell_time_min=2          # Quicker state transitions
        )
        
    def start_training_session(self, dog_name):
        """Initialize dashboard for training session"""
        self.dashboard = self.ethogram.create_dashboard(figsize=(12, 8))
        self.session_start = datetime.now()
        print(f"🐕 Training session started for {dog_name}")
        
    def process_command_execution(self, command, prediction, confidence):
        """Process dog's response to training command"""
        smoothed_state = self.ethogram.add_observation(prediction, confidence)
        self.ethogram.update_dashboard(time_window=60.0)
        
        # Training feedback
        if command.lower() == smoothed_state.lower() and confidence > 0.8:
            print(f"✅ Excellent {command}! Duration: {self.get_current_state_duration():.1f}s")
        elif confidence < 0.5:
            print(f"⚠️ Unclear response to {command} (confidence: {confidence:.2f})")
            
    def end_session_report(self):
        """Generate training progress report"""
        summary = self.ethogram.generate_summary_statistics()
        
        report = {
            'session_duration': summary['observation_duration_seconds'],
            'commands_attempted': summary['total_observations'],
            'success_rate': summary['confidence_statistics']['mean_confidence'],
            'dominant_behaviors': summary['state_frequencies'],
            'improvement_areas': self.identify_training_gaps(summary)
        }
        
        return report
```

### Behavioral Progress Tracking

```python
def track_training_progress(dog_id, session_files):
    """Analyze training progression over multiple sessions"""
    progress_data = []
    
    for session_file in sorted(session_files):
        # Analyze each session
        session_analysis = analyze_training_session(session_file)
        
        # Track key metrics
        progress_data.append({
            'date': extract_session_date(session_file),
            'command_accuracy': session_analysis['session_quality'],
            'behavioral_stability': calculate_stability_metric(session_analysis),
            'response_time': calculate_average_response_time(session_analysis)
        })
    
    # Generate progress visualization
    plot_training_progression(progress_data, dog_id)
    
    return progress_data
```

## Configuration Options

### Clustering Analysis Settings

```python
# Customize clustering parameters
analyzer = BehavioralMotifAnalyzer()
analyzer.configure({
    'k_range': (8, 15),           # Range of cluster numbers to evaluate
    'n_init': 10,                 # Number of K-means initializations
    'random_state': 42,           # Reproducible results
    'feature_extraction_batch': 64, # Batch size for feature extraction
    'visualization_style': 'publication'  # Plot styling
})
```

### Ethogram Display Settings

```python
# Customize ethogram visualization
ethogram = EthogramVisualizer()
ethogram.configure_display({
    'behavior_colors': {
        'sit': '#2E8B57',      # Sea green
        'down': '#4169E1',     # Royal blue
        'stand': '#FF6347',    # Tomato
        'walking': '#FF8C00',  # Dark orange
        'stay': '#9932CC'      # Dark orchid
    },
    'update_frequency': 5,         # Dashboard refresh rate
    'time_window': 30.0,          # Seconds to display
    'confidence_threshold': 0.6,   # Minimum acceptance confidence
    'export_format': 'json'       # Session data export format
})
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```bash
# Error: Model path not found
# Solution: Ensure model files are in correct location
ls -la models/best_tcn_vae_*.pth
```

#### 2. CUDA/GPU Issues  
```bash
# Error: CUDA out of memory
# Solution: Reduce batch size or use CPU
python evaluation/clustering_analysis.py --device cpu --batch_size 32
```

#### 3. Visualization Display Issues
```bash
# Error: No display available
# Solution: Use non-interactive backend
export MPLBACKEND=Agg
python evaluation/ethogram_visualizer.py
```

### Performance Optimization

#### For Real-Time Applications
```python
# Optimize for low-latency deployment
ethogram = EthogramVisualizer(
    confidence_threshold=0.6,
    smoothing_window=3,      # Reduce for faster response
    dwell_time_min=2,       # Minimum for stability
    max_history=500         # Limit memory usage
)
```

#### For Batch Analysis
```python
# Optimize for throughput
analyzer = BehavioralMotifAnalyzer()
analyzer.configure({
    'batch_size': 128,       # Larger batches for efficiency
    'n_init': 20,           # More initializations for quality
    'parallel_jobs': -1     # Use all CPU cores
})
```

## File Structure

```
tcn-vae-models/
├── README.md                           # Repository overview
├── SPRINT_1_STAGE_1_COMPLETION.md      # Sprint results documentation
├── USAGE_INSTRUCTIONS.md               # This file
├── best_tcn_vae_72pct.pth             # Baseline model (72.13% accuracy)
├── model_config.json                   # Model configuration
├── tcn_encoder_for_edgeinfer.onnx      # ONNX export for EdgeInfer
└── evaluation/                         # Behavioral analysis tools
    ├── clustering_analysis.py          # K-means motif extraction
    ├── ethogram_visualizer.py          # Real-time behavioral timeline
    ├── clustering_analysis_results.json # Analysis results data
    ├── ethogram_demo_session.json      # Example session analytics
    └── clustering_plots/               # Generated visualizations
        ├── baseline_k_selection.png    # Silhouette analysis
        ├── baseline_tsne_analysis.png  # t-SNE behavioral motifs
        ├── baseline_umap_analysis.png  # UMAP latent projection
        └── baseline_distributions.png  # Cluster frequencies
```

## Next Steps

### Enhanced Model Integration
1. **Train Enhanced Models**: Achieve 86.53% → 90%+ accuracy
2. **Multi-Model Analysis**: Compare behavioral motifs across architectures
3. **Cross-Dataset Validation**: WISDM + HAPT + quadruped integration

### Professional Deployment
1. **EdgeInfer Integration**: Real-time behavioral analysis API
2. **Mobile Interface**: Tablet/smartphone trainer dashboard
3. **Progress Tracking**: Historical training session analysis

---

*Professional behavioral analysis tools ready for production deployment*  
*Status: ✅ PRODUCTION READY*  
*Integration: Compatible with EdgeInfer and professional training workflows*