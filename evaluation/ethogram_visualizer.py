#!/usr/bin/env python3
"""
Ethogram Visualization System for TCN-VAE Behavioral Analysis
============================================================

Real-time behavioral timeline visualization tool with confidence scoring,
temporal smoothing, and behavioral transition analysis.

Sprint 1, Task 2: Ethogram visualization implementation
Target: Real-time behavioral state display with confidence thresholds

Author: TCN-VAE Implementation Team
Date: 2025-09-06
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import seaborn as sns
from collections import deque, defaultdict
import json
import time
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EthogramVisualizer:
    """
    Real-time ethogram visualization system for behavioral state analysis.
    
    Features:
    - Live behavioral timeline display
    - Confidence-based state filtering (threshold >0.6)
    - Temporal smoothing with state transition analysis
    - Interactive behavioral dashboard
    - Summary statistics (frequency, duration, transitions)
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.6,
                 smoothing_window: int = 5,
                 dwell_time_min: int = 3,
                 max_history: int = 1000):
        """
        Initialize ethogram visualizer.
        
        Args:
            confidence_threshold: Minimum confidence for state acceptance (0.6)
            smoothing_window: Window size for temporal smoothing (5 samples)
            dwell_time_min: Minimum samples for state persistence (3 samples)
            max_history: Maximum behavioral history to maintain (1000 samples)
        """
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.dwell_time_min = dwell_time_min
        self.max_history = max_history
        
        # Behavioral state tracking
        self.behavioral_history = deque(maxlen=max_history)
        self.confidence_history = deque(maxlen=max_history)
        self.timestamp_history = deque(maxlen=max_history)
        self.smoothed_states = deque(maxlen=max_history)
        
        # Transition analysis
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.state_durations = defaultdict(list)
        self.current_state = None
        self.current_state_start = None
        self.state_counts = defaultdict(int)
        
        # Visualization components
        self.fig = None
        self.axes = None
        self.timeline_ax = None
        self.confidence_ax = None
        self.stats_ax = None
        
        # Color mapping for behavioral states
        self.behavior_colors = {
            'sit': '#2E8B57',      # Sea green
            'down': '#4169E1',     # Royal blue
            'stand': '#FF6347',    # Tomato
            'stay': '#9932CC',     # Dark orchid
            'walking': '#FF8C00',  # Dark orange
            'lying': '#20B2AA',    # Light sea green
            'transition': '#FFD700', # Gold
            'unknown': '#808080'   # Gray
        }
        
        print("🎯 EthogramVisualizer initialized")
        print(f"   Confidence threshold: {confidence_threshold}")
        print(f"   Smoothing window: {smoothing_window}")
        print(f"   Minimum dwell time: {dwell_time_min}")
        
    def add_observation(self, 
                       state: str, 
                       confidence: float, 
                       timestamp: Optional[datetime] = None) -> str:
        """
        Add new behavioral observation with temporal smoothing.
        
        Args:
            state: Predicted behavioral state
            confidence: Model confidence (0.0-1.0)
            timestamp: Observation timestamp (default: now)
            
        Returns:
            Smoothed behavioral state after filtering
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            state = 'unknown'
            
        # Add to history
        self.behavioral_history.append(state)
        self.confidence_history.append(confidence)
        self.timestamp_history.append(timestamp)
        
        # Apply temporal smoothing
        smoothed_state = self._apply_temporal_smoothing()
        self.smoothed_states.append(smoothed_state)
        
        # Update state tracking and transitions
        self._update_state_tracking(smoothed_state, timestamp)
        
        return smoothed_state
    
    def _apply_temporal_smoothing(self) -> str:
        """
        Apply temporal smoothing to reduce state flickering.
        
        Uses majority voting within smoothing window with dwell time constraints.
        
        Returns:
            Temporally smoothed behavioral state
        """
        if len(self.behavioral_history) < self.smoothing_window:
            return list(self.behavioral_history)[-1] if self.behavioral_history else 'unknown'
        
        # Get recent states for smoothing
        recent_states = list(self.behavioral_history)[-self.smoothing_window:]
        recent_confidences = list(self.confidence_history)[-self.smoothing_window:]
        
        # Weighted majority voting (confidence-weighted)
        state_scores = defaultdict(float)
        for state, conf in zip(recent_states, recent_confidences):
            if state != 'unknown':
                state_scores[state] += conf
        
        if not state_scores:
            return 'unknown'
        
        # Select state with highest weighted score
        candidate_state = max(state_scores.items(), key=lambda x: x[1])[0]
        
        # Apply dwell time constraint
        if len(self.smoothed_states) >= self.dwell_time_min:
            recent_smoothed = list(self.smoothed_states)[-self.dwell_time_min:]
            if all(s == candidate_state for s in recent_smoothed):
                return candidate_state
            else:
                # Not enough persistence, keep previous smoothed state
                return list(self.smoothed_states)[-1] if self.smoothed_states else candidate_state
        
        return candidate_state
    
    def _update_state_tracking(self, state: str, timestamp: datetime):
        """Update state duration tracking and transition analysis."""
        if state != self.current_state:
            # Record previous state duration
            if self.current_state is not None and self.current_state_start is not None:
                duration = (timestamp - self.current_state_start).total_seconds()
                self.state_durations[self.current_state].append(duration)
                
                # Record transition
                self.transition_matrix[self.current_state][state] += 1
            
            # Start new state
            self.current_state = state
            self.current_state_start = timestamp
            
        # Update state counts
        self.state_counts[state] += 1
    
    def create_dashboard(self, figsize: tuple = (15, 10)) -> plt.Figure:
        """
        Create comprehensive behavioral dashboard.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure with dashboard layout
        """
        self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        self.fig.suptitle('🐕 Real-Time Behavioral Ethogram Dashboard', fontsize=16, fontweight='bold')
        
        # Timeline plot (top left)
        self.timeline_ax = self.axes[0, 0]
        self.timeline_ax.set_title('📊 Behavioral Timeline', fontweight='bold')
        self.timeline_ax.set_xlabel('Time (seconds)')
        self.timeline_ax.set_ylabel('Behavioral State')
        
        # Confidence plot (top right)
        self.confidence_ax = self.axes[0, 1]
        self.confidence_ax.set_title('🎯 Confidence Tracking', fontweight='bold')
        self.confidence_ax.set_xlabel('Time (seconds)')
        self.confidence_ax.set_ylabel('Confidence Score')
        self.confidence_ax.axhline(y=self.confidence_threshold, color='r', linestyle='--', alpha=0.7, label='Threshold')
        self.confidence_ax.legend()
        
        # State distribution (bottom left)
        self.stats_ax = self.axes[1, 0]
        self.stats_ax.set_title('📈 State Distribution', fontweight='bold')
        
        # Transition heatmap (bottom right)
        self.transition_ax = self.axes[1, 1]
        self.transition_ax.set_title('🔄 State Transitions', fontweight='bold')
        
        plt.tight_layout()
        return self.fig
    
    def update_dashboard(self, time_window: float = 30.0):
        """
        Update all dashboard components with latest data.
        
        Args:
            time_window: Time window to display (seconds)
        """
        if not self.behavioral_history:
            return
            
        current_time = datetime.now()
        
        # Get data within time window
        cutoff_time = current_time - timedelta(seconds=time_window)
        recent_indices = [i for i, ts in enumerate(self.timestamp_history) 
                         if ts >= cutoff_time]
        
        if not recent_indices:
            return
        
        # Extract recent data
        recent_times = [self.timestamp_history[i] for i in recent_indices]
        recent_states = [self.smoothed_states[i] if i < len(self.smoothed_states) 
                        else self.behavioral_history[i] for i in recent_indices]
        recent_confidences = [self.confidence_history[i] for i in recent_indices]
        
        # Convert timestamps to seconds relative to first timestamp
        if recent_times:
            start_time = recent_times[0]
            time_seconds = [(ts - start_time).total_seconds() for ts in recent_times]
        else:
            time_seconds = []
        
        # Update timeline plot
        self._update_timeline_plot(time_seconds, recent_states)
        
        # Update confidence plot
        self._update_confidence_plot(time_seconds, recent_confidences)
        
        # Update statistics
        self._update_statistics_plot()
        
        # Update transition heatmap
        self._update_transition_heatmap()
        
        plt.pause(0.01)  # Allow GUI update
    
    def _update_timeline_plot(self, time_seconds: List[float], states: List[str]):
        """Update behavioral timeline visualization."""
        self.timeline_ax.clear()
        self.timeline_ax.set_title('📊 Behavioral Timeline', fontweight='bold')
        self.timeline_ax.set_xlabel('Time (seconds)')
        self.timeline_ax.set_ylabel('Behavioral State')
        
        if not time_seconds or not states:
            return
        
        # Create state blocks
        unique_states = sorted(set(states))
        state_to_y = {state: i for i, state in enumerate(unique_states)}
        
        # Plot state blocks
        for i in range(len(time_seconds) - 1):
            state = states[i]
            color = self.behavior_colors.get(state, '#808080')
            
            rect = Rectangle((time_seconds[i], state_to_y[state] - 0.4),
                           time_seconds[i+1] - time_seconds[i], 0.8,
                           facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            self.timeline_ax.add_patch(rect)
        
        # Format y-axis
        self.timeline_ax.set_yticks(range(len(unique_states)))
        self.timeline_ax.set_yticklabels(unique_states)
        self.timeline_ax.set_ylim(-0.5, len(unique_states) - 0.5)
        
        # Add grid
        self.timeline_ax.grid(True, alpha=0.3)
    
    def _update_confidence_plot(self, time_seconds: List[float], confidences: List[float]):
        """Update confidence tracking plot."""
        self.confidence_ax.clear()
        self.confidence_ax.set_title('🎯 Confidence Tracking', fontweight='bold')
        self.confidence_ax.set_xlabel('Time (seconds)')
        self.confidence_ax.set_ylabel('Confidence Score')
        
        if time_seconds and confidences:
            self.confidence_ax.plot(time_seconds, confidences, 'b-', linewidth=2, alpha=0.8)
            self.confidence_ax.fill_between(time_seconds, confidences, alpha=0.3)
        
        # Add threshold line
        self.confidence_ax.axhline(y=self.confidence_threshold, color='r', linestyle='--', 
                                  alpha=0.7, label=f'Threshold ({self.confidence_threshold})')
        self.confidence_ax.set_ylim(0, 1)
        self.confidence_ax.legend()
        self.confidence_ax.grid(True, alpha=0.3)
    
    def _update_statistics_plot(self):
        """Update behavioral state distribution plot."""
        self.stats_ax.clear()
        self.stats_ax.set_title('📈 State Distribution', fontweight='bold')
        
        if not self.state_counts:
            return
        
        states = list(self.state_counts.keys())
        counts = list(self.state_counts.values())
        colors = [self.behavior_colors.get(state, '#808080') for state in states]
        
        bars = self.stats_ax.bar(states, counts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            self.stats_ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                              f'{count}', ha='center', va='bottom', fontweight='bold')
        
        self.stats_ax.set_ylabel('Observation Count')
        plt.setp(self.stats_ax.get_xticklabels(), rotation=45, ha='right')
    
    def _update_transition_heatmap(self):
        """Update state transition heatmap."""
        self.transition_ax.clear()
        self.transition_ax.set_title('🔄 State Transitions', fontweight='bold')
        
        if not self.transition_matrix:
            return
        
        # Convert transition matrix to DataFrame for heatmap
        all_states = set()
        for from_state in self.transition_matrix:
            all_states.add(from_state)
            for to_state in self.transition_matrix[from_state]:
                all_states.add(to_state)
        
        all_states = sorted(all_states)
        
        # Create transition matrix
        transition_data = np.zeros((len(all_states), len(all_states)))
        for i, from_state in enumerate(all_states):
            for j, to_state in enumerate(all_states):
                transition_data[i, j] = self.transition_matrix[from_state][to_state]
        
        # Plot heatmap
        sns.heatmap(transition_data, annot=True, fmt='.0f', 
                   xticklabels=all_states, yticklabels=all_states,
                   cmap='Blues', ax=self.transition_ax, cbar_kws={'label': 'Transition Count'})
        
        self.transition_ax.set_xlabel('To State')
        self.transition_ax.set_ylabel('From State')
        plt.setp(self.transition_ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(self.transition_ax.get_yticklabels(), rotation=0)
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive behavioral summary statistics.
        
        Returns:
            Dictionary with behavioral analytics
        """
        total_observations = len(self.behavioral_history)
        
        if total_observations == 0:
            return {"error": "No observations recorded"}
        
        # Calculate state frequencies
        state_frequencies = {state: count/total_observations 
                           for state, count in self.state_counts.items()}
        
        # Calculate average state durations
        avg_durations = {}
        for state, durations in self.state_durations.items():
            if durations:
                avg_durations[state] = {
                    'mean_duration': np.mean(durations),
                    'median_duration': np.median(durations),
                    'std_duration': np.std(durations),
                    'total_time': sum(durations),
                    'occurrence_count': len(durations)
                }
        
        # Calculate transition probabilities
        transition_probabilities = {}
        for from_state in self.transition_matrix:
            total_transitions = sum(self.transition_matrix[from_state].values())
            if total_transitions > 0:
                transition_probabilities[from_state] = {
                    to_state: count/total_transitions 
                    for to_state, count in self.transition_matrix[from_state].items()
                }
        
        # Calculate confidence statistics
        if self.confidence_history:
            confidence_stats = {
                'mean_confidence': np.mean(list(self.confidence_history)),
                'median_confidence': np.median(list(self.confidence_history)),
                'min_confidence': np.min(list(self.confidence_history)),
                'max_confidence': np.max(list(self.confidence_history)),
                'below_threshold_rate': sum(1 for c in self.confidence_history 
                                          if c < self.confidence_threshold) / len(self.confidence_history)
            }
        else:
            confidence_stats = {}
        
        summary = {
            'total_observations': total_observations,
            'observation_duration_seconds': (self.timestamp_history[-1] - self.timestamp_history[0]).total_seconds() if len(self.timestamp_history) > 1 else 0,
            'unique_states_observed': len(self.state_counts),
            'state_frequencies': state_frequencies,
            'average_state_durations': avg_durations,
            'transition_probabilities': transition_probabilities,
            'confidence_statistics': confidence_stats,
            'behavioral_complexity': len(self.transition_matrix),  # Number of different transitions observed
            'generated_timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def save_session_data(self, filepath: str):
        """
        Save complete ethogram session data to JSON file.
        
        Args:
            filepath: Output file path
        """
        session_data = {
            'configuration': {
                'confidence_threshold': self.confidence_threshold,
                'smoothing_window': self.smoothing_window,
                'dwell_time_min': self.dwell_time_min,
                'max_history': self.max_history
            },
            'behavioral_data': {
                'raw_states': list(self.behavioral_history),
                'smoothed_states': list(self.smoothed_states),
                'confidences': list(self.confidence_history),
                'timestamps': [ts.isoformat() for ts in self.timestamp_history]
            },
            'summary_statistics': self.generate_summary_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"✅ Session data saved to {filepath}")


def demo_real_time_ethogram():
    """
    Demonstrate real-time ethogram visualization with synthetic behavioral data.
    """
    print("🎬 Starting Real-Time Ethogram Demo")
    print("=" * 50)
    
    # Initialize ethogram visualizer
    ethogram = EthogramVisualizer(
        confidence_threshold=0.6,
        smoothing_window=5,
        dwell_time_min=3
    )
    
    # Create dashboard
    fig = ethogram.create_dashboard(figsize=(15, 10))
    plt.ion()  # Enable interactive mode
    
    # Simulate real-time behavioral observations
    behaviors = ['sit', 'down', 'stand', 'walking', 'stay']
    
    print("📊 Generating synthetic behavioral sequence...")
    
    for i in range(100):
        # Simulate behavioral state with some persistence
        if i == 0:
            current_behavior = np.random.choice(behaviors)
        elif np.random.random() < 0.1:  # 10% chance of state change
            current_behavior = np.random.choice(behaviors)
        
        # Simulate confidence with some noise
        base_confidence = 0.8
        confidence = max(0.1, min(1.0, base_confidence + np.random.normal(0, 0.15)))
        
        # Add observation
        smoothed_state = ethogram.add_observation(current_behavior, confidence)
        
        # Update dashboard every 5 observations
        if i % 5 == 0:
            ethogram.update_dashboard(time_window=30.0)
            print(f"  Step {i}: {current_behavior} (conf: {confidence:.2f}) -> smoothed: {smoothed_state}")
        
        # Small delay to simulate real-time
        time.sleep(0.1)
    
    # Generate final summary
    print("\n📊 Final Behavioral Summary")
    print("=" * 30)
    summary = ethogram.generate_summary_statistics()
    
    print(f"Total observations: {summary['total_observations']}")
    print(f"Session duration: {summary['observation_duration_seconds']:.1f} seconds")
    print(f"Unique states observed: {summary['unique_states_observed']}")
    
    print("\n📈 State Frequencies:")
    for state, freq in summary['state_frequencies'].items():
        print(f"  {state}: {freq:.1%}")
    
    # Save results
    output_file = "evaluation/ethogram_demo_session.json"
    ethogram.save_session_data(output_file)
    
    # Save final dashboard
    plt.ioff()  # Disable interactive mode
    plt.savefig("evaluation/ethogram_dashboard.png", dpi=300, bbox_inches='tight')
    print(f"📊 Dashboard saved to evaluation/ethogram_dashboard.png")
    
    print("\n🎉 Real-time ethogram demo complete!")
    print("🎯 Sprint 1, Task 2: Ethogram visualization system implemented")
    
    return ethogram, summary


if __name__ == "__main__":
    print("🚀 TCN-VAE Ethogram Visualization System")
    print("📋 Sprint 1, Task 2: Real-time behavioral timeline visualization")
    print("=" * 70)
    
    # Run demonstration
    ethogram_viz, session_summary = demo_real_time_ethogram()
    
    print("\n✅ Ethogram visualization system ready for integration!")
    print("🔗 Next: Integrate with TCN-VAE model outputs for live behavioral analysis")