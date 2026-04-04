# deployment.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import UAVConfig
from network import TrajectoryPlannerNetwork
from channel_model import ChannelModel

class UAVDeployment:
    def __init__(self, model: TrajectoryPlannerNetwork, config: UAVConfig):
        self.config  = config
        self.model   = model
        self.model.eval()
        self.channel = ChannelModel(config)

    def predict(self, scenario: dict):
        features = np.concatenate([
            scenario['start_position'].flatten(),
            scenario['end_position'].flatten(),
            scenario['user_positions'].flatten(),
            scenario['eavesdropper_positions'].flatten(),
        ])
        device  = next(self.model.parameters()).device
        tensor  = torch.FloatTensor(features).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_traj, pred_power = self.model(tensor)

        trajectory = pred_traj.cpu().numpy()[0]
        power      = pred_power.cpu().numpy()[0]

        total_dist = sum(
            np.linalg.norm(trajectory[i] - trajectory[i - 1])
            for i in range(1, len(trajectory))
        )
        avg_secrecy = np.mean([
            self.channel.compute_secrecy_rate(
                trajectory[t],
                scenario['user_positions'],
                scenario['eavesdropper_positions'],
                power[t]
            )
            for t in range(self.config.time_slots)
        ])

        metrics = {'total_distance': total_dist, 'avg_secrecy': avg_secrecy}
        return trajectory, power, metrics

    def visualize_solution(self, trajectory: np.ndarray, scenario: dict, metrics: dict):
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(trajectory[:, 0], trajectory[:, 1],
                'b-o', linewidth=2, label='UAV Path (Least Distance)')
        ax.scatter(*scenario['start_position'][:2],
                   c='blue',   marker='s', s=150, label='Start')
        ax.scatter(*scenario['end_position'][:2],
                   c='purple', marker='*', s=200, label='End Destination')

        for i, u in enumerate(scenario['user_positions']):
            ax.scatter(u[0], u[1], c='green', marker='^', s=150,
                       label='User Node' if i == 0 else '')

        for i, e in enumerate(scenario['eavesdropper_positions']):
            ax.scatter(e[0], e[1], c='red', marker='v', s=150,
                       label='Eavesdropper' if i == 0 else '')

        ax.set_title(f"Multi-Node UAV Path | "
                     f"Total Path Length: {metrics['total_distance']:.2f} m")
        ax.legend()
        ax.grid(True)
        plt.show()