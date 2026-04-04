# channel_model.py
import numpy as np
from config import UAVConfig

class ChannelModel:
    """Physical-layer channel model for UAV communications."""

    def __init__(self, config: UAVConfig):
        self.config = config

    def compute_distance(self, uav_pos: np.ndarray, ground_pos: np.ndarray) -> float:
        return np.sqrt(np.sum((uav_pos - ground_pos) ** 2))

    def compute_path_loss(self, distance: float) -> float:
        distance = max(distance, self.config.reference_distance)
        path_loss_db = (self.config.reference_loss +
                        10 * self.config.path_loss_exponent *
                        np.log10(distance / self.config.reference_distance))
        return 10 ** (-path_loss_db / 10)

    def compute_channel_gain(self, uav_pos: np.ndarray, ground_pos: np.ndarray) -> float:
        distance = self.compute_distance(uav_pos, ground_pos)
        return self.compute_path_loss(distance)

    def compute_rate(self, channel_gain: float, transmit_power: float) -> float:
        snr = (transmit_power * channel_gain) / self.config.noise_power
        return self.config.bandwidth * np.log2(1 + snr)

    def compute_secrecy_rate(self,
                             uav_pos: np.ndarray,
                             user_positions: np.ndarray,
                             eve_positions: np.ndarray,
                             transmit_power: float) -> float:
        max_eve_rate = max(
            self.compute_rate(self.compute_channel_gain(uav_pos, eve_pos), transmit_power)
            for eve_pos in eve_positions
        )
        total_secrecy = sum(
            max(0.0, self.compute_rate(self.compute_channel_gain(uav_pos, u_pos), transmit_power)
                     - max_eve_rate)
            for u_pos in user_positions
        )
        return total_secrecy