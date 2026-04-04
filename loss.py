# loss.py
import numpy as np
import torch
import torch.nn as nn
from config import UAVConfig
from channel_model import ChannelModel

class ConstraintPenalties:
    def __init__(self, config: UAVConfig):
        self.config  = config
        self.channel = ChannelModel(config)

    def distance_penalty(self, trajectory: torch.Tensor) -> torch.Tensor:
        step_dists  = torch.norm(trajectory[:, 1:, :] - trajectory[:, :-1, :], dim=2)
        return torch.mean(torch.sum(step_dists, dim=1))

    def destination_penalty(self, trajectory: torch.Tensor, end_pos: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sum((trajectory[:, -1, :] - end_pos) ** 2, dim=1))

    def mobility_constraint_penalty(self, trajectory: torch.Tensor) -> torch.Tensor:
        speeds     = (torch.norm(trajectory[:, 1:, :] - trajectory[:, :-1, :], dim=2)
                      / self.config.slot_duration)
        violations = torch.relu(speeds - self.config.max_velocity)
        return torch.mean(violations ** 2)

    def boundary_constraint_penalty(self, trajectory: torch.Tensor) -> torch.Tensor:
        x_lo  = torch.relu(-trajectory[:, :, 0])
        x_hi  = torch.relu(trajectory[:, :, 0]  - self.config.area_size)
        y_lo  = torch.relu(-trajectory[:, :, 1])
        y_hi  = torch.relu(trajectory[:, :, 1]  - self.config.area_size)
        h_err = torch.abs(trajectory[:, :, 2]   - self.config.uav_height)
        return torch.mean((x_lo + x_hi + y_lo + y_hi + h_err) ** 2)

    def user_visitation_penalty(self, trajectory: torch.Tensor, users_np: np.ndarray) -> torch.Tensor:
        users_tensor = torch.tensor(users_np, dtype=torch.float32, device=trajectory.device)
        # Only evaluate 2D distance (XY plane) to see if UAV flew directly over the user
        traj_2d  = trajectory[:, :, :2]     # (B, T, 2)
        users_2d = users_tensor[:, :, :2]   # (B, U, 2)
        
        # Calculate pairwise distance from every time slot to every user: results in (B, T, U)
        dists = torch.norm(traj_2d.unsqueeze(2) - users_2d.unsqueeze(1), dim=3)
        
        # Get the minimum distance recorded for each user across the flight timeline: (B, U)
        min_dists, _ = torch.min(dists, dim=1)
        return torch.mean(min_dists ** 2)

    def secrecy_rate_penalty(self, trajectory: torch.Tensor, power: torch.Tensor, 
                             users_np: np.ndarray, eves_np: np.ndarray) -> torch.Tensor:
        traj_np  = trajectory.detach().cpu().numpy()
        power_np = power.detach().cpu().numpy()
        penalties = []

        for b in range(trajectory.shape[0]):
            total = 0.0
            for t in range(self.config.time_slots):
                sr = self.channel.compute_secrecy_rate(
                    traj_np[b, t], users_np[b], eves_np[b], power_np[b, t]
                )
                if sr < self.config.min_secrecy_rate:
                    total += (self.config.min_secrecy_rate - sr) ** 2
            penalties.append(total)

        return torch.tensor(penalties, device=trajectory.device).mean()

class UAVLoss(nn.Module):
    def __init__(self, config: UAVConfig):
        super().__init__()
        self.config      = config
        self.constraints = ConstraintPenalties(config)
        self.mse         = nn.MSELoss()

    def forward(self, pred_traj, pred_power, target_traj, target_power, features):
        traj_loss  = self.mse(pred_traj,  target_traj)
        power_loss = self.mse(pred_power, target_power)

        b_size  = features.shape[0]
        end_pos = features[:, 3:6]

        idx    = 6
        n_u    = self.config.num_users * 3
        users  = features[:, idx:idx + n_u].cpu().numpy().reshape(b_size, self.config.num_users, 3)
        idx   += n_u
        n_e    = self.config.num_eavesdroppers * 3
        eves   = features[:, idx:idx + n_e].cpu().numpy().reshape(b_size, self.config.num_eavesdroppers, 3)

        mob_pen   = self.constraints.mobility_constraint_penalty(pred_traj)
        bound_pen = self.constraints.boundary_constraint_penalty(pred_traj)
        dest_pen  = self.constraints.destination_penalty(pred_traj, end_pos)
        dist_pen  = self.constraints.distance_penalty(pred_traj)
        sec_pen   = self.constraints.secrecy_rate_penalty(pred_traj, pred_power, users, eves)
        vis_pen   = self.constraints.user_visitation_penalty(pred_traj, users)

        total = (traj_loss + power_loss
                 + self.config.penalty_mobility    * mob_pen
                 + self.config.penalty_boundary    * bound_pen
                 + self.config.penalty_destination * dest_pen
                 + self.config.penalty_distance    * dist_pen
                 + self.config.penalty_secrecy     * sec_pen
                 + self.config.penalty_visitation  * vis_pen)

        return total, {
            'total': total.item(),
            'dist':  dist_pen.item(),
            'dest':  dest_pen.item(),
        }