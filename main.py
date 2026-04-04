import torch
from torch.utils.data import DataLoader
from config import UAVConfig
from data_generation import generate_training_data, generate_random_scenario
from network import TrajectoryPlannerNetwork
from trainer import UAVTrainer
from deployment import UAVDeployment

def main():
    # ── 1. Configuration ─────────────────────────────────────────────────────
    config = UAVConfig()
    # ── 2. Dataset Generation ────────────────────────────────────────────────
    # Changed num_samples to 2500
    dataset    = generate_training_data(config, num_samples=2500)
    train_size = int(config.train_test_split * len(dataset))

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, range(train_size)),
        batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, range(train_size, len(dataset))),
        batch_size=config.batch_size
    )

    # ── 3. Model ─────────────────────────────────────────────────────────────
    model = TrajectoryPlannerNetwork(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 4. Training ──────────────────────────────────────────────────────────
    trainer = UAVTrainer(config, model)
    trainer.train(train_loader, val_loader)

    # ── 5. Inference & Visualisation ─────────────────────────────────────────
    deployment    = UAVDeployment(model, config)
    test_scenario = generate_random_scenario(config)
    trajectory, power, metrics = deployment.predict(test_scenario)

    print("\nTest Run:")
    print(f"  Total Distance (Hovering Path): {metrics['total_distance']:.2f} m")
    print(f"  Average Multi-Node Secrecy Rate: {metrics['avg_secrecy']:.4f} bits/s/Hz")

    deployment.visualize_solution(trajectory, test_scenario, metrics)

if __name__ == "__main__":
    main()