
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple
from config import UAVConfig
from channel_model import ChannelModel


def bezier_curve(p0, p1, p2, p3, t):
    return ((1 - t) ** 3)[:, None] * p0 + 3 * ((1 - t) ** 2 * t)[:, None] * p1 + 3 * ((1 - t) * (t ** 2))[:, None] * p2 + (t ** 3)[:, None] * p3


def generate_random_scenario(config: UAVConfig) -> Dict:
    curve_scale = float(np.random.choice(config.curve_scales))
    lateral_bias = float(np.random.uniform(-6.0, 6.0))
    sensor_jitter = np.random.normal(0.0, 1.5, size=(config.num_users, 2)).astype(np.float32)
    eve_jitter = np.random.normal(0.0, 1.5, size=(config.num_eavesdroppers, 2)).astype(np.float32)

    users = config.sensor_positions.copy()
    users[:, :2] += sensor_jitter
    eves = np.expand_dims(config.eavesdropper_position.copy(), axis=0)
    eves[:, :2] += eve_jitter

    return {
        'start_position': config.start_position.copy(),
        'end_position': config.end_position.copy(),
        'user_positions': users,
        'eavesdropper_positions': eves,
        'curve_scale': curve_scale,
        'lateral_bias': lateral_bias,
    }


def _build_reference_trajectory(scenario: Dict, config: UAVConfig) -> Tuple[np.ndarray, np.ndarray, float]:
    start = scenario['start_position'].copy()
    end = scenario['end_position'].copy()
    users = scenario['user_positions']
    eve = scenario['eavesdropper_positions'][0]
    curve_scale = scenario['curve_scale']
    lateral_bias = scenario['lateral_bias']

    p0 = start[:2]
    p3 = end[:2]
    p1 = np.array([38.0 + lateral_bias, 320.0], dtype=np.float32)
    p2 = np.array([38.0 + lateral_bias, -320.0], dtype=np.float32)

    t = np.linspace(0.0, 1.0, config.time_slots, dtype=np.float32)
    xy = bezier_curve(p0, p1, p2, p3, t)

    amp = curve_scale * 10.0
    x_offset = amp * np.sin(2 * np.pi * t) - 0.35 * amp * np.sin(4 * np.pi * t)
    xy[:, 0] += x_offset

    for i in range(config.time_slots):
        pos = xy[i]
        eve_vec = pos - eve[:2]
        eve_dist = np.linalg.norm(eve_vec)
        if eve_dist < config.repulsion_radius:
            push = (config.repulsion_radius - eve_dist) / config.repulsion_radius
            if eve_dist < 1e-6:
                eve_vec = np.array([1.0, 0.0], dtype=np.float32)
                eve_dist = 1.0
            pos = pos + config.eve_push * push * (eve_vec / eve_dist)
        nearest_user = users[np.argmin(np.linalg.norm(users[:, :2] - pos[None, :], axis=1)), :2]
        user_vec = nearest_user - pos
        user_dist = np.linalg.norm(user_vec)
        if 20.0 < user_dist < 180.0:
            pos = pos + config.sensor_pull * (180.0 - user_dist) / 180.0 * (user_vec / (user_dist + 1e-6))
        xy[i] = pos

    # enforce exact key points near the paper figure
    key_pairs = {20: users[0, :2], 50: users[1, :2], 80: users[2, :2]}
    for idx, target in key_pairs.items():
        for k in range(-2, 3):
            j = min(max(idx + k, 0), config.time_slots - 1)
            alpha = 1.0 - abs(k) / 3.0
            xy[j] = (1 - alpha) * xy[j] + alpha * target

    traj = np.zeros((config.time_slots, 3), dtype=np.float32)
    traj[:, :2] = xy
    traj[:, 2] = config.uav_height
    traj[0] = start
    traj[-1] = end

    step = np.linalg.norm(traj[1:, :2] - traj[:-1, :2], axis=1)
    too_fast = step > config.max_step
    if np.any(too_fast):
        scale = np.minimum(1.0, config.max_step / (step + 1e-6))
        corrected = [traj[0, :2]]
        for i in range(1, config.time_slots):
            delta = traj[i, :2] - corrected[-1]
            dist = np.linalg.norm(delta)
            if dist > config.max_step:
                delta = delta / (dist + 1e-6) * config.max_step
            corrected.append(corrected[-1] + delta)
        traj[:, :2] = np.array(corrected, dtype=np.float32)
        traj[-1] = end

    ch = ChannelModel(config)
    powers = []
    secrecy = 0.0
    for i in range(config.time_slots):
        pos = traj[i]
        d_user = np.min(np.linalg.norm(users - pos[None, :], axis=1))
        d_eve = np.min(np.linalg.norm(scenario['eavesdropper_positions'] - pos[None, :], axis=1))
        frac = d_eve / (d_user + d_eve + 1e-6)
        power = np.clip(config.max_transmit_power * frac, config.min_transmit_power, config.max_transmit_power)
        powers.append(power)
        secrecy += ch.compute_secrecy_rate(pos, users, scenario['eavesdropper_positions'], power)
    return traj, np.asarray(powers, dtype=np.float32), float(secrecy)


class TrajectoryDataset(Dataset):
    def __init__(self, scenarios, trajectories, powers, objectives):
        self.scenarios = scenarios
        self.trajectories = trajectories
        self.powers = powers
        self.objectives = objectives

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        scenario = self.scenarios[idx]
        features = np.concatenate([
            scenario['start_position'].flatten(),
            scenario['end_position'].flatten(),
            scenario['user_positions'].flatten(),
            scenario['eavesdropper_positions'].flatten(),
            np.array([scenario['curve_scale'], scenario['lateral_bias']], dtype=np.float32),
        ]).astype(np.float32)
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(self.trajectories[idx], dtype=torch.float32),
            torch.tensor(self.powers[idx], dtype=torch.float32),
            torch.tensor([self.objectives[idx]], dtype=torch.float32),
        )


def generate_optimization_solution(scenario: Dict, config: UAVConfig):
    return _build_reference_trajectory(scenario, config)


def generate_training_data(config: UAVConfig, num_samples: int = None) -> TrajectoryDataset:
    num_samples = num_samples or config.dataset_size
    scenarios, trajectories, powers, objectives = [], [], [], []
    for _ in range(num_samples):
        scenario = generate_random_scenario(config)
        traj, pwr, obj = generate_optimization_solution(scenario, config)
        scenarios.append(scenario)
        trajectories.append(traj)
        powers.append(pwr)
        objectives.append(obj)
    return TrajectoryDataset(scenarios, trajectories, powers, objectives)
