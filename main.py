
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from config import UAVConfig
from data_generation import generate_training_data, generate_random_scenario
from network import TrajectoryPlannerNetwork
from trainer import UAVTrainer
from deployment import UAVDeployment


def main():
    config = UAVConfig()
    config.set_seed()

    dataset = generate_training_data(config, config.dataset_size)
    train_size = int(len(dataset) * config.train_test_split)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.seed))
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

    model = TrajectoryPlannerNetwork(config)
    trainer = UAVTrainer(config, model)
    trainer.train(train_loader, val_loader)

    base_scenario = generate_random_scenario(config)
    deployment = UAVDeployment(model, config)

    trajectories = {}
    metrics = {}
    base_scale = base_scenario['curve_scale']
    for T, scale in zip(config.figure_times, [0.18, 0.24, 0.30]):
        scenario = copy.deepcopy(base_scenario)
        scenario['curve_scale'] = scale
        traj, power, met = deployment.predict(scenario)
        trajectories[T] = traj
        metrics[T] = met

    deployment.visualize_solution(trajectories, base_scenario, metrics, config.plot_path)

    print('Saved model to', config.model_path)
    print('Saved trajectory figure to', config.plot_path)
    for T in config.figure_times:
        print(f"T={T}s | distance={metrics[T]['total_distance']:.2f} m | secrecy={metrics[T]['avg_secrecy']:.2f}")


if __name__ == '__main__':
    main()
