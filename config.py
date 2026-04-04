# config.py

class UAVConfig:
    """Central configuration – shared across the entire project."""

    def __init__(self):
        # Environment
        self.area_size        = 1000.0
        self.uav_height       = 100.0
        self.max_velocity     = 80.0   # m/s
        self.time_slots       = 50     
        self.slot_duration    = 1.0    # seconds per slot

        # Multi-node scenario
        self.num_users        = 10
        self.num_eavesdroppers = 4     # must be strictly < num_users

        # Channel / communication
        self.carrier_freq        = 2.4e9
        self.bandwidth           = 1e6
        self.noise_power         = 1e-10
        self.path_loss_exponent  = 2.0
        self.reference_distance  = 1.0
        self.reference_loss      = 30.0   # dB

        # Power
        self.max_transmit_power  = 1.0
        self.min_transmit_power  = 0.01
        self.min_secrecy_rate    = 1.0

        # ML training
        self.batch_size          = 64
        self.learning_rate       = 1e-3
        self.num_epochs          = 20     
        self.train_test_split    = 0.8

        # Constraint penalty weights
        self.penalty_mobility    = 100.0
        self.penalty_power       = 50.0
        self.penalty_secrecy     = 1000.0 # Heavily increased to force eavesdropper avoidance
        self.penalty_collision   = 500.0
        self.penalty_boundary    = 100.0
        self.penalty_destination = 1000.0
        self.penalty_distance    = 20.0   # Increased to tighten and shorten the path
        self.penalty_visitation  = 150.0