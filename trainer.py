# trainer.py
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config import UAVConfig
from network import TrajectoryPlannerNetwork
from loss import UAVLoss

class UAVTrainer:
    def __init__(self, config: UAVConfig, model: TrajectoryPlannerNetwork):
        self.config    = config
        self.model     = model
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = UAVLoss(config)
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

    def run_epoch(self, dataloader: DataLoader, is_train: bool = True) -> dict:
        self.model.train() if is_train else self.model.eval()
        epoch_losses = []

        with torch.set_grad_enabled(is_train):
            for features, target_traj, target_power, _ in dataloader:
                features      = features.to(self.device)
                target_traj   = target_traj.view(-1, self.config.time_slots, 3).to(self.device)
                target_power  = target_power.to(self.device)

                pred_traj, pred_power = self.model(features)
                loss, loss_dict = self.criterion(
                    pred_traj, pred_power, target_traj, target_power, features
                )

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                epoch_losses.append(loss_dict)

        return {k: np.mean([d[k] for d in epoch_losses]) for k in epoch_losses[0]}

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        print("\nStarting Training...")
        best_val = float('inf')

        for epoch in range(self.config.num_epochs):
            t_losses = self.run_epoch(train_loader, is_train=True)
            v_losses = self.run_epoch(val_loader,   is_train=False)
            self.scheduler.step(v_losses['total'])

            print(f"Epoch [{epoch + 1:>3}/{self.config.num_epochs}] | "
                  f"Train Loss: {t_losses['total']:.4f} | "
                  f"Val Loss: {v_losses['total']:.4f}")

            if v_losses['total'] < best_val:
                best_val = v_losses['total']

        print(f"\nTraining complete. Best Validation Loss: {best_val:.4f}")