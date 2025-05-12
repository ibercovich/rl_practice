#!/usr/bin/env python3
"""
Visualization tools for RL training
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from datetime import datetime
import pathlib

class TrainingVisualizer:
    def __init__(self, window_size=100, log_dir="training_logs"):
        self.rewards = []
        self.baselines = []
        self.losses = []
        self.running_success_rate = deque(maxlen=window_size)
        self.steps = []
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup figure
        plt.style.use('ggplot')
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.tight_layout(pad=3.0)
        
    def update(self, step, reward, baseline, loss):
        self.steps.append(step)
        self.rewards.append(reward)
        self.baselines.append(baseline)
        self.losses.append(loss.item())
        self.running_success_rate.append(reward)
        
        if step % 10 == 0:
            self.plot()
    
    def plot(self):
        if not self.steps:
            return
            
        # Clear all subplots
        for ax in self.axes.flat:
            ax.clear()
            
        # Plot rewards and baseline
        self.axes[0, 0].plot(self.steps, self.rewards, 'b-', alpha=0.3, label='Reward')
        self.axes[0, 0].plot(self.steps, self.baselines, 'r-', label='Baseline')
        self.axes[0, 0].set_title('Rewards & Baseline')
        self.axes[0, 0].set_xlabel('Step')
        self.axes[0, 0].set_ylabel('Value')
        self.axes[0, 0].legend()
        
        # Plot loss
        self.axes[0, 1].plot(self.steps, self.losses, 'g-')
        self.axes[0, 1].set_title('Policy Loss')
        self.axes[0, 1].set_xlabel('Step')
        self.axes[0, 1].set_ylabel('Loss')
        
        # Plot running success rate
        if self.steps:
            success_rate = np.mean(list(self.running_success_rate)) if self.running_success_rate else 0
            window_steps = min(len(self.steps), 100)
            running_rewards = [np.mean(self.rewards[max(0, i-window_steps):i+1]) 
                            for i in range(window_steps-1, len(self.rewards))]
            running_steps = self.steps[window_steps-1:]
            
            self.axes[1, 0].plot(running_steps, running_rewards, 'm-')
            self.axes[1, 0].set_title(f'Running Average Reward (window={window_steps})')
            self.axes[1, 0].set_xlabel('Step')
            self.axes[1, 0].set_ylabel('Avg Reward')
            
            # Additional metrics
            self.axes[1, 1].bar(['Success Rate'], [success_rate], color='green')
            self.axes[1, 1].set_title(f'Current Success Rate: {success_rate:.2f}')
            self.axes[1, 1].set_ylim(0, 1)
        
        self.fig.tight_layout()
        
        # Save the figure
        plt.savefig(self.log_dir / f"training_progress_{self.timestamp}.png")
        
        # Save metrics to CSV
        metrics_file = self.log_dir / f"training_metrics_{self.timestamp}.csv"
        if not metrics_file.exists():
            with open(metrics_file, 'w') as f:
                f.write("step,reward,baseline,loss\n")
        
        with open(metrics_file, 'a') as f:
            for i in range(len(self.steps) - (len(self.steps) % 10), len(self.steps)):
                if i < len(self.rewards) and i < len(self.baselines) and i < len(self.losses):
                    f.write(f"{self.steps[i]},{self.rewards[i]},{self.baselines[i]},{self.losses[i]}\n")
    
    def save_checkpoint_plot(self, step):
        """Save special plot at checkpoint time"""
        plt.savefig(f"training_progress_step_{step}.png") 