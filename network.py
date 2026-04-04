import torch
import torch.nn as nn
from config import UAVConfig


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class TrajectoryPlannerNetwork(nn.Module):
    def __init__(self, config: UAVConfig):
        super().__init__()
        self.config = config
        self.input_dim = 6 + config.num_users * 3 + config.num_eavesdroppers * 3 + 2
        hidden = config.hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            ResidualBlock(hidden, config.dropout),
            ResidualBlock(hidden, config.dropout),
            nn.LayerNorm(hidden),
        )
        self.traj_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.time_slots * 2),
        )
        self.power_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, config.time_slots),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        traj_xy = self.traj_head(h).view(-1, self.config.time_slots, 2)
        z = torch.full((x.size(0), self.config.time_slots, 1), float(self.config.uav_height), device=x.device)
        traj = torch.cat([traj_xy, z], dim=-1)
        power_norm = self.power_head(h)
        power = self.config.min_transmit_power + power_norm * (self.config.max_transmit_power - self.config.min_transmit_power)
        return traj, power
