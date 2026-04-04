import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict
from config import UAVConfig
from channel_model import ChannelModel

def generate_random_scenario(config: UAVConfig) -> Dict:
    start_uav = np.array([
        np.random.uniform(0, 200),
        np.random.uniform(0, config.area_size),
        config.uav_height
    ])
    end_uav = np.array([
        np.random.uniform(800, 1000),
        np.random.uniform(0, config.area_size),
        config.uav_height
    ])

    user_positions = np.array([
        [np.random.uniform(100, 900), np.random.uniform(100, 900), 0.0]
        for _ in range(config.num_users)
    ])

    eve_positions = []
    for _ in range(config.num_eavesdroppers):
        while True:
            pos = np.array([
                np.random.uniform(300, 700),
                np.random.uniform(200, 800),
                0.0
            ])
            if min(np.linalg.norm(pos[:2] - u[:2]) for u in user_positions) > 50.0:
                eve_positions.append(pos)
                break
    eve_positions = np.array(eve_positions)

    return {
        'start_position':        start_uav,
        'end_position':          end_uav,
        'user_positions':        user_positions,
        'eavesdropper_positions': eve_positions,
    }

def generate_optimization_solution(scenario: Dict, config: UAVConfig) -> Tuple[np.ndarray, np.ndarray, float]:
    start_pos = scenario['start_position']
    end_pos   = scenario['end_position']
    user_pos  = scenario['user_positions']
    eve_pos   = scenario['eavesdropper_positions']

    # 1. Plan Waypoints (Start -> Nearest User -> ... -> End)
    waypoints = [start_pos]
    curr_pos = start_pos
    remaining_users = list(user_pos)

    while remaining_users:
        next_user = min(remaining_users, key=lambda u: np.linalg.norm(curr_pos[:2] - u[:2]))
        # Match UAV height for waypoint target
        wp = np.array([next_user[0], next_user[1], config.uav_height])
        waypoints.append(wp)
        curr_pos = wp
        remaining_users = [u for u in remaining_users if not np.array_equal(u, next_user)]
    
    waypoints.append(end_pos)

    # 2. Distribute time slots across the segments
    trajectory = np.zeros((config.time_slots, 3))
    powers     = np.zeros(config.time_slots)

    num_segments = len(waypoints) - 1
    slots_per_segment = config.time_slots // num_segments

    idx = 0
    for i in range(num_segments):
        w_start = waypoints[i]
        w_end   = waypoints[i+1]
        
        seg_slots = slots_per_segment
        if i == num_segments - 1:
            seg_slots = config.time_slots - idx  # absorb any remainder

        for j in range(seg_slots):
            alpha = j / max(1, (seg_slots - 1)) if seg_slots > 1 else 1.0
            base_pos = (1 - alpha) * w_start + alpha * w_end

            # Evade eavesdroppers
            for eve in eve_pos:
                dist = np.linalg.norm(base_pos[:2] - eve[:2])
                if dist < 100:
                    repel = (base_pos[:2] - eve[:2]) / (dist + 1e-5)
                    base_pos[:2] += repel * (100 - dist) * 0.2

            base_pos     = np.clip(base_pos, 0, config.area_size)
            base_pos[2]  = config.uav_height
            trajectory[idx] = base_pos

            min_user_dist = min(np.linalg.norm(base_pos - u) for u in user_pos)
            min_eve_dist  = min(np.linalg.norm(base_pos - e) for e in eve_pos)
            power_factor  = min_eve_dist / (min_user_dist + min_eve_dist + 1e-5)
            powers[idx] = np.clip(
                config.max_transmit_power * power_factor,
                config.min_transmit_power,
                config.max_transmit_power
            )
            idx += 1

    channel = ChannelModel(config)
    total_secrecy = sum(
        channel.compute_secrecy_rate(trajectory[t], user_pos, eve_pos, powers[t])
        for t in range(config.time_slots)
    )

    return trajectory, powers, total_secrecy

class TrajectoryDataset(Dataset):
    def __init__(self, scenarios, trajectories, powers, objectives):
        self.scenarios    = scenarios
        self.trajectories = trajectories
        self.powers       = powers
        self.objectives   = objectives

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        scenario = self.scenarios[idx]
        features = np.concatenate([
            scenario['start_position'].flatten(),
            scenario['end_position'].flatten(),
            scenario['user_positions'].flatten(),
            scenario['eavesdropper_positions'].flatten(),
        ])
        return (
            torch.FloatTensor(features),
            torch.FloatTensor(self.trajectories[idx].flatten()),
            torch.FloatTensor(self.powers[idx]),
            torch.FloatTensor([self.objectives[idx]]),
        )

def generate_training_data(config: UAVConfig, num_samples: int = 2500) -> TrajectoryDataset:
    print(f"Generating {num_samples} training samples...")
    scenarios, trajectories, powers, objectives = [], [], [], []

    for i in range(num_samples):
        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")
        scenario = generate_random_scenario(config)
        traj, pwr, obj = generate_optimization_solution(scenario, config)
        scenarios.append(scenario)
        trajectories.append(traj)
        powers.append(pwr)
        objectives.append(obj)

    print("Dataset generation complete.")
    return TrajectoryDataset(scenarios, trajectories, powers, objectives)