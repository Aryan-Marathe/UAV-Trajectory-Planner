
import numpy as np
from config import UAVConfig


class ChannelModel:
    def __init__(self, config: UAVConfig):
        self.config = config

    def compute_distance(self, uav_pos: np.ndarray, ground_pos: np.ndarray) -> float:
        return float(np.sqrt(np.sum((uav_pos - ground_pos) ** 2)))

    def compute_path_loss(self, distance: float) -> float:
        distance = max(distance, self.config.reference_distance)
        path_loss_db = (
            self.config.reference_loss
            + 10 * self.config.path_loss_exponent * np.log10(distance / self.config.reference_distance)
        )
        return 10 ** (-path_loss_db / 10)

    def compute_channel_gain(self, uav_pos: np.ndarray, ground_pos: np.ndarray) -> float:
        return self.compute_path_loss(self.compute_distance(uav_pos, ground_pos))

    def compute_rate(self, channel_gain: float, transmit_power: float) -> float:
        snr = (transmit_power * channel_gain) / self.config.noise_power
        return float(self.config.bandwidth * np.log2(1.0 + snr))

    def compute_secrecy_rate(self, uav_pos: np.ndarray, user_positions: np.ndarray, eve_positions: np.ndarray, transmit_power: float) -> float:
        max_eve_rate = max(self.compute_rate(self.compute_channel_gain(uav_pos, eve), transmit_power) for eve in eve_positions)
        total = 0.0
        for user in user_positions:
            user_rate = self.compute_rate(self.compute_channel_gain(uav_pos, user), transmit_power)
            total += max(0.0, user_rate - max_eve_rate)
        return float(total)
