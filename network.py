# network.py
import torch
import torch.nn as nn
from config import UAVConfig

class TrajectoryPlannerNetwork(nn.Module):
    def __init__(self, config: UAVConfig):
        super().__init__()
        self.config = config

        input_dim    = 6 + config.num_users * 3 + config.num_eavesdroppers * 3
        traj_dim     = config.time_slots * 3
        power_dim    = config.time_slots
        hidden_dim   = 128  # Increased from 16 to vastly improve spatial accuracy

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim,  hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), # Added extra layer for depth
        )
        self.trajectory_head = nn.Linear(hidden_dim, traj_dim)
        self.power_head      = nn.Sequential(
            nn.Linear(hidden_dim, power_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        h              = self.feature_extractor(x)
        trajectory     = self.trajectory_head(h).view(-1, self.config.time_slots, 3)
        power_norm     = self.power_head(h)
        power          = (self.config.min_transmit_power +
                          power_norm * (self.config.max_transmit_power -
                                        self.config.min_transmit_power))
        return trajectory, power