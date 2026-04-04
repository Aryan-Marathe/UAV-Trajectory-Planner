import copy
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config import UAVConfig
from network import TrajectoryPlannerNetwork
from loss import UAVLoss


class UAVTrainer:
    def __init__(self, config: UAVConfig, model: TrajectoryPlannerNetwork):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.criterion = UAVLoss(config)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)
        self.best_state = None
        self.best_val = float('inf')

    def run_epoch(self, dataloader: DataLoader, is_train: bool = True):
        self.model.train(is_train)
        all_losses = []
        for features, target_traj, target_power, _ in dataloader:
            features = features.to(self.device)
            target_traj = target_traj.to(self.device)
            target_power = target_power.to(self.device)
            pred_traj, pred_power = self.model(features)
            loss, loss_dict = self.criterion(pred_traj, pred_power, target_traj, target_power, features)
            if is_train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
            all_losses.append(loss_dict)
        return {k: float(np.mean([d[k] for d in all_losses])) for k in all_losses[0]}

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        for epoch in range(self.config.num_epochs):
            train_loss = self.run_epoch(train_loader, True)
            val_loss = self.run_epoch(val_loader, False)
            self.scheduler.step()
            if val_loss['total'] < self.best_val:
                self.best_val = val_loss['total']
                self.best_state = copy.deepcopy(self.model.state_dict())
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:03d}/{self.config.num_epochs} | train={train_loss['total']:.4f} | val={val_loss['total']:.4f}")
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            model_dir = os.path.dirname(self.config.model_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            torch.save(self.best_state, self.config.model_path)
        print(f"Best validation loss: {self.best_val:.4f}")
