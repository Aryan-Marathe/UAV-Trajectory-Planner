import torch
import torch.nn as nn
from config import UAVConfig


class UAVLoss(nn.Module):
    def __init__(self, config: UAVConfig):
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss()

    def _sensor_distance_penalty(self, pred_traj, users):
        pred_xy = pred_traj[:, :, :2]
        users_xy = users[:, :, :2]
        d = torch.cdist(pred_xy, users_xy)
        min_d = d.min(dim=1).values
        return min_d.mean()

    def _eve_penalty(self, pred_traj, eves):
        pred_xy = pred_traj[:, :, :2]
        eves_xy = eves[:, :, :2]
        d = torch.cdist(pred_xy, eves_xy)
        safe_margin = 110.0
        return torch.relu(safe_margin - d).pow(2).mean()

    def forward(self, pred_traj, pred_power, target_traj, target_power, features):
        b = features.size(0)
        idx = 0
        start = features[:, idx:idx+3]; idx += 3
        end = features[:, idx:idx+3]; idx += 3
        users = features[:, idx:idx + self.config.num_users*3].view(b, self.config.num_users, 3); idx += self.config.num_users*3
        eves = features[:, idx:idx + self.config.num_eavesdroppers*3].view(b, self.config.num_eavesdroppers, 3)

        traj_loss = self.mse(pred_traj[:, :, :2], target_traj[:, :, :2])
        power_loss = self.mse(pred_power, target_power)
        vel = torch.norm(pred_traj[:, 1:, :2] - pred_traj[:, :-1, :2], dim=-1)
        vel_penalty = torch.relu(vel - self.config.max_step).pow(2).mean()
        smooth_penalty = self.mse(pred_traj[:, 2:, :2] - pred_traj[:, 1:-1, :2], target_traj[:, 2:, :2] - target_traj[:, 1:-1, :2])
        endpoint_penalty = self.mse(pred_traj[:, 0, :], start) + self.mse(pred_traj[:, -1, :], end)
        altitude_penalty = self.mse(pred_traj[:, :, 2], torch.full_like(pred_traj[:, :, 2], self.config.uav_height))
        eve_penalty = self._eve_penalty(pred_traj, eves)
        sensor_penalty = self._sensor_distance_penalty(pred_traj, users)
        shape_penalty = self.mse(pred_traj[:, :, :2], target_traj[:, :, :2])

        total = (
            self.config.lambda_traj * traj_loss
            + self.config.lambda_power * power_loss
            + self.config.lambda_shape * shape_penalty
            + self.config.lambda_velocity * vel_penalty
            + self.config.lambda_endpoint * endpoint_penalty
            + self.config.lambda_altitude * altitude_penalty
            + self.config.lambda_eve * eve_penalty
            + self.config.lambda_sensor * sensor_penalty
            + self.config.lambda_smooth * smooth_penalty
        )

        return total, {
            'total': float(total.detach().cpu()),
            'traj': float(traj_loss.detach().cpu()),
            'power': float(power_loss.detach().cpu()),
            'velocity': float(vel_penalty.detach().cpu()),
            'endpoint': float(endpoint_penalty.detach().cpu()),
            'eve': float(eve_penalty.detach().cpu()),
        }
