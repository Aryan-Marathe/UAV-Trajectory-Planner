import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import UAVConfig
from channel_model import ChannelModel


class UAVDeployment:
    def __init__(self, model, config: UAVConfig):
        self.model = model
        self.config = config
        self.channel = ChannelModel(config)
        self.model.eval()

    def _build_features(self, scenario: dict):
        return np.concatenate([
            scenario['start_position'].flatten(),
            scenario['end_position'].flatten(),
            scenario['user_positions'].flatten(),
            scenario['eavesdropper_positions'].flatten(),
            np.array([scenario['curve_scale'], scenario['lateral_bias']], dtype=np.float32),
        ]).astype(np.float32)

    def predict(self, scenario: dict):
        x = torch.tensor(self._build_features(scenario), dtype=torch.float32).unsqueeze(0)
        device = next(self.model.parameters()).device
        x = x.to(device)
        with torch.no_grad():
            traj, power = self.model(x)
        traj = traj[0].cpu().numpy()
        power = power[0].cpu().numpy()
        total_dist = float(np.sum(np.linalg.norm(traj[1:, :2] - traj[:-1, :2], axis=1)))
        avg_secrecy = float(np.mean([
            self.channel.compute_secrecy_rate(traj[t], scenario['user_positions'], scenario['eavesdropper_positions'], power[t])
            for t in range(self.config.time_slots)
        ]))
        return traj, power, {'total_distance': total_dist, 'avg_secrecy': avg_secrecy}

    def visualize_solution(self, trajectories: dict, scenario: dict, metrics: dict, save_path: str):
        fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
        styles = {
            51: dict(color='#355fb3', linestyle='-', marker='o', markevery=8, mfc='none', mec='#355fb3', label='T = 51 s'),
            70: dict(color='#d62728', linestyle='--', marker='s', markevery=8, mfc='none', mec='#d62728', label='T = 70 s'),
            100: dict(color='#222222', linestyle='-.', marker='*', markevery=8, label='T = 100 s'),
        }
        for T, traj in trajectories.items():
            st = styles[T]
            ax.plot(traj[:,0], traj[:,1], linewidth=1.6, markersize=6, **st)

        users = scenario['user_positions']
        eve = scenario['eavesdropper_positions'][0]
        ax.scatter(users[:,0], users[:,1], c='#3355a5', marker='s', s=70, zorder=5)
        ax.scatter(eve[0], eve[1], c='#222222', marker='*', s=180, zorder=6)
        ax.scatter(scenario['start_position'][0], scenario['start_position'][1], c='#222222', s=120, zorder=6)
        ax.scatter(scenario['end_position'][0], scenario['end_position'][1], c='#222222', s=120, zorder=6)

        ax.text(-49, 320, 'Sensor 1', color='#3355a5', fontsize=11)
        ax.text(-14, -45, 'Sensor 2', color='#3355a5', fontsize=11)
        ax.text(37, -358, 'Sensor 3', color='#3355a5', fontsize=11)
        ax.text(1, -286, 'Eavesdropper', color='#222222', fontsize=11)
        ax.text(-5, 430, 'Initial Point', color='#222222', fontsize=11)
        ax.text(-10, -460, 'Final Point', color='#222222', fontsize=11)

        ax.set_xlim(-50, 50)
        ax.set_ylim(-500, 500)
        ax.set_xlabel('X (m)', fontsize=14)
        ax.set_ylabel('Y (m)', fontsize=14)
        ax.grid(True, alpha=0.35)
        ax.legend(loc='upper right', framealpha=1.0, fancybox=False, edgecolor='0.3')
        plt.tight_layout()
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
