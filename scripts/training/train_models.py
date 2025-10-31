"""Comprehensive model training script with PyTorch and ONNX export."""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.neural_architectures import (
    TransformerEncoder,
    MemoryNetwork,
    ReasoningNetwork,
    PolicyNetwork,
    ValueNetwork,
    WorldModel
)
from src.models.pytorch_trainer import BaseTrainer, ReinforcementLearningTrainer
from src.models.onnx_utils import ONNXExporter, ModelConverter
from src.core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_dataset(input_dim: int, output_dim: int, num_samples: int = 1000):
    """Create dummy dataset for training."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, output_dim)
    return TensorDataset(X, y)


def train_transformer_model(args):
    """Train transformer encoder model."""
    logger.info("Training Transformer Encoder")

    # Model
    model = TransformerEncoder(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )

    # Create dummy sequence dataset
    num_samples = 1000
    seq_length = 50
    X = torch.randint(0, args.vocab_size, (num_samples, seq_length))
    y = torch.randint(0, args.vocab_size, (num_samples, seq_length))
    dataset = TensorDataset(X, y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Training
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        gradient_clip=1.0
    )

    save_dir = Path(args.output_dir) / "transformer"
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=str(save_dir),
        early_stopping_patience=args.patience
    )

    # Export to ONNX
    if args.export_onnx:
        logger.info("Exporting to ONNX")
        model.eval()
        dummy_input = torch.randint(0, args.vocab_size, (1, seq_length))

        onnx_path = ModelConverter.pytorch_to_onnx(
            model=model,
            model_name="transformer_encoder",
            dummy_input=dummy_input,
            output_dir=str(save_dir / "onnx"),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"}
            }
        )

        logger.info(f"Model exported to ONNX: {onnx_path}")

    return model


def train_memory_network(args):
    """Train memory network."""
    logger.info("Training Memory Network")

    model = MemoryNetwork(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        memory_size=args.memory_size,
        num_heads=args.num_heads
    )

    # Create sequence dataset
    num_samples = 1000
    seq_length = 10
    X = torch.randn(num_samples, seq_length, args.input_dim)
    y = torch.randn(num_samples, seq_length, args.input_dim)
    dataset = TensorDataset(X, y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Training
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion
    )

    save_dir = Path(args.output_dir) / "memory_network"
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=str(save_dir)
    )

    # Export to ONNX
    if args.export_onnx:
        logger.info("Exporting to ONNX")
        model.eval()
        dummy_input = torch.randn(1, seq_length, args.input_dim)

        onnx_path = ModelConverter.pytorch_to_onnx(
            model=lambda x: model(x)[0],  # Only output, not attention weights
            model_name="memory_network",
            dummy_input=dummy_input,
            output_dir=str(save_dir / "onnx"),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 1: "sequence"},
                "output": {0: "batch", 1: "sequence"}
            }
        )

        logger.info(f"Model exported to ONNX: {onnx_path}")

    return model


def train_reasoning_network(args):
    """Train reasoning network."""
    logger.info("Training Reasoning Network")

    model = ReasoningNetwork(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_reasoning_steps=args.reasoning_steps
    )

    # Create dataset
    dataset = create_dummy_dataset(args.input_dim, args.output_dim, num_samples=2000)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Training
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler
    )

    save_dir = Path(args.output_dir) / "reasoning_network"
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=str(save_dir)
    )

    # Export to ONNX
    if args.export_onnx:
        logger.info("Exporting to ONNX")
        model.eval()
        dummy_input = torch.randn(1, args.input_dim)

        onnx_path = ModelConverter.pytorch_to_onnx(
            model=model,
            model_name="reasoning_network",
            dummy_input=dummy_input,
            output_dir=str(save_dir / "onnx"),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
        )

        logger.info(f"Model exported to ONNX: {onnx_path}")

    return model


def train_rl_agent(args):
    """Train RL policy and value networks."""
    logger.info("Training RL Agent")

    policy_model = PolicyNetwork(
        observation_dim=args.observation_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        continuous=False
    )

    value_model = ValueNetwork(
        observation_dim=args.observation_dim,
        hidden_dim=args.hidden_dim
    )

    # Optimizers
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=args.learning_rate)
    value_optimizer = optim.Adam(value_model.parameters(), lr=args.learning_rate)

    trainer = ReinforcementLearningTrainer(
        policy_model=policy_model,
        value_model=value_model,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda
    )

    # Simulated training (in real scenario, would interact with environment)
    save_dir = Path(args.output_dir) / "rl_agent"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # Simulate episode data
        states = torch.randn(100, args.observation_dim).to(trainer.device)
        actions = torch.randint(0, args.action_dim, (100,)).to(trainer.device)
        rewards = torch.randn(100).to(trainer.device)
        dones = torch.zeros(100).to(trainer.device)

        # Compute values and advantages
        with torch.no_grad():
            values = value_model(states).squeeze()
        advantages = trainer.compute_advantages(rewards, values, dones)
        returns = advantages + values

        # Get old log probs
        with torch.no_grad():
            logits = policy_model(states)
            old_log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions)

        # Training step
        losses = trainer.train_step_ppo(
            states, actions, old_log_probs, advantages, returns
        )

        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}: Policy Loss = {losses['policy_loss']:.4f}, "
                f"Value Loss = {losses['value_loss']:.4f}"
            )

    # Save models
    torch.save(policy_model.state_dict(), save_dir / "policy_model.pt")
    torch.save(value_model.state_dict(), save_dir / "value_model.pt")

    # Export to ONNX
    if args.export_onnx:
        logger.info("Exporting RL models to ONNX")
        policy_model.eval()
        value_model.eval()

        dummy_obs = torch.randn(1, args.observation_dim)

        # Export policy
        policy_path = ModelConverter.pytorch_to_onnx(
            model=policy_model,
            model_name="policy_network",
            dummy_input=dummy_obs,
            output_dir=str(save_dir / "onnx"),
            input_names=["observation"],
            output_names=["action_logits"],
            dynamic_axes={"observation": {0: "batch"}, "action_logits": {0: "batch"}}
        )

        # Export value
        value_path = ModelConverter.pytorch_to_onnx(
            model=value_model,
            model_name="value_network",
            dummy_input=dummy_obs,
            output_dir=str(save_dir / "onnx"),
            input_names=["observation"],
            output_names=["value"],
            dynamic_axes={"observation": {0: "batch"}, "value": {0: "batch"}}
        )

        logger.info(f"Policy exported to: {policy_path}")
        logger.info(f"Value exported to: {value_path}")

    return policy_model, value_model


def train_world_model(args):
    """Train world model."""
    logger.info("Training World Model")

    model = WorldModel(
        observation_dim=args.observation_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim
    )

    # Create dataset
    num_samples = 2000
    observations = torch.randn(num_samples, args.observation_dim)
    actions = torch.randn(num_samples, args.action_dim)
    next_observations = torch.randn(num_samples, args.observation_dim)
    rewards = torch.randn(num_samples, 1)

    dataset = TensorDataset(observations, actions, next_observations, rewards)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Custom training loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    save_dir = Path(args.output_dir) / "world_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            obs, action, next_obs, reward = [b.to(device) for b in batch]

            # Forward
            pred_next_obs, pred_reward, (mean, logvar, next_mean, next_logvar) = model(obs, action)

            # Losses
            reconstruction_loss = nn.MSELoss()(pred_next_obs, next_obs)
            reward_loss = nn.MSELoss()(pred_reward, reward)

            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            kl_loss += -0.5 * torch.sum(1 + next_logvar - next_mean.pow(2) - next_logvar.exp())

            loss = reconstruction_loss + reward_loss + 0.001 * kl_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    # Save
    torch.save(model.state_dict(), save_dir / "world_model.pt")

    # Export to ONNX
    if args.export_onnx:
        logger.info("Exporting World Model to ONNX")
        model.eval()
        dummy_obs = torch.randn(1, args.observation_dim)
        dummy_action = torch.randn(1, args.action_dim)

        onnx_path = ModelConverter.pytorch_to_onnx(
            model=model,
            model_name="world_model",
            dummy_input=(dummy_obs, dummy_action),
            output_dir=str(save_dir / "onnx"),
            input_names=["observation", "action"],
            output_names=["next_observation", "reward", "latent_info"],
            dynamic_axes={
                "observation": {0: "batch"},
                "action": {0: "batch"},
                "next_observation": {0: "batch"},
                "reward": {0: "batch"}
            }
        )

        logger.info(f"Model exported to ONNX: {onnx_path}")

    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PyTorch models with ONNX export")
    parser.add_argument("--model", type=str, required=True,
                      choices=["transformer", "memory", "reasoning", "rl", "world"],
                      help="Model type to train")
    parser.add_argument("--output-dir", type=str, default="data/models/trained",
                      help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")

    # Transformer args
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Memory network args
    parser.add_argument("--memory-size", type=int, default=100)
    parser.add_argument("--num-heads", type=int, default=4)

    # Common args
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=64)

    # Reasoning args
    parser.add_argument("--reasoning-steps", type=int, default=3)

    # RL args
    parser.add_argument("--observation-dim", type=int, default=64)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)

    # World model args
    parser.add_argument("--latent-dim", type=int, default=64)

    args = parser.parse_args()

    # Train selected model
    if args.model == "transformer":
        model = train_transformer_model(args)
    elif args.model == "memory":
        model = train_memory_network(args)
    elif args.model == "reasoning":
        model = train_reasoning_network(args)
    elif args.model == "rl":
        model = train_rl_agent(args)
    elif args.model == "world":
        model = train_world_model(args)

    logger.info(f"Training complete for {args.model} model!")


if __name__ == "__main__":
    main()
