import random
import numpy as np
import torch


class UAVConfig:
    """Configuration for secure UAV trajectory learning."""

    def __init__(self):
        self.seed = 42
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Paper-like geometry
        self.area_x = (-60.0, 60.0)
        self.area_y = (-520.0, 520.0)
        self.area_size = 1000.0
        self.uav_height = 100.0
        self.time_slots = 101
        self.slot_duration = 1.0
        self.max_velocity = 20.0
        self.max_step = self.max_velocity * self.slot_duration

        self.start_position = np.array([0.0, 500.0, self.uav_height], dtype=np.float32)
        self.end_position = np.array([0.0, -500.0, self.uav_height], dtype=np.float32)
        self.eavesdropper_position = np.array([0.0, -300.0, 0.0], dtype=np.float32)

        # Match the shown figure / paper baseline
        self.num_users = 3
        self.num_eavesdroppers = 1
        self.sensor_positions = np.array([
            [-50.0, 300.0, 0.0],
            [0.0, 0.0, 0.0],
            [50.0, -300.0, 0.0],
        ], dtype=np.float32)

        # Channel / communication
        self.bandwidth = 1e6
        self.noise_power = 1e-9
        self.path_loss_exponent = 2.0
        self.reference_distance = 1.0
        self.reference_loss = 30.0
        self.max_transmit_power = 4.0
        self.min_transmit_power = 0.05
        self.min_secrecy_rate = 5e5

        # Data generation
        self.dataset_size = 2400
        self.train_test_split = 0.85
        self.curve_scales = [0.18, 0.24, 0.30]
        self.repulsion_radius = 170.0
        self.eve_push = 40.0
        self.sensor_pull = 20.0
        self.figure_times = [51, 70, 100]

        # Training
        self.batch_size = 96
        self.learning_rate = 8e-4
        self.weight_decay = 1e-4
        self.num_epochs = 140
        self.hidden_dim = 192
        self.dropout = 0.1
        self.grad_clip = 1.0
        self.label_smoothing = 0.02

        # Loss weights
        self.lambda_traj = 8.0
        self.lambda_power = 0.5
        self.lambda_shape = 8.0
        self.lambda_velocity = 6.0
        self.lambda_endpoint = 20.0
        self.lambda_altitude = 4.0
        self.lambda_eve = 10.0
        self.lambda_sensor = 3.0
        self.lambda_smooth = 3.0

        self.model_path = 'output/best_uav_model.pt'
        self.plot_path = 'output/uav_trajectory_plot.png'

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
