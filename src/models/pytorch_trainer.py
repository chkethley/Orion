"""PyTorch training utilities and trainers."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import logging
from tqdm import tqdm
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Base trainer for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scheduler: Optional[Any] = None,
        gradient_clip: Optional[float] = None
    ):
        """Initialize trainer."""
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            loss = self.train_step(batch)
            total_loss += loss

            # Update progress bar
            pbar.set_postfix({'loss': loss, 'avg_loss': total_loss / (batch_idx + 1)})

            self.global_step += 1

        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

    def train_step(self, batch: Any) -> float:
        """Single training step."""
        # Move batch to device
        if isinstance(batch, (list, tuple)):
            batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
            inputs, targets = batch[0], batch[1]
        else:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)

        for batch in tqdm(val_loader, desc="Validating"):
            loss = self.validation_step(batch)
            total_loss += loss

        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

    @torch.no_grad()
    def validation_step(self, batch: Any) -> float:
        """Single validation step."""
        # Move batch to device
        if isinstance(batch, (list, tuple)):
            batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
            inputs, targets = batch[0], batch[1]
        else:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        return loss.item()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        save_dir: str = "checkpoints",
        early_stopping_patience: Optional[int] = None
    ):
        """Train model for multiple epochs."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        patience_counter = 0

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_metrics['loss'])

            logger.info(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}")

            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])

                logger.info(f"Epoch {epoch}: Val Loss = {val_metrics['loss']:.4f}")

                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint(save_path / "best_model.pt")
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'] if val_loader else train_metrics['loss'])
                else:
                    self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(save_path / f"checkpoint_epoch_{epoch}.pt")

        # Save final model
        self.save_checkpoint(save_path / "final_model.pt")

        # Save training history
        with open(save_path / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Checkpoint loaded from {path}")


class ReinforcementLearningTrainer(BaseTrainer):
    """Trainer for reinforcement learning agents."""

    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        policy_optimizer: optim.Optimizer,
        value_optimizer: optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """Initialize RL trainer."""
        self.policy_model = policy_model.to(device)
        self.value_model = value_model.to(device)
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        return advantages

    def train_step_ppo(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        clip_epsilon: float = 0.2
    ) -> Dict[str, float]:
        """PPO training step."""
        # Policy update
        action_dist = self.policy_model(states)
        log_probs = torch.distributions.Categorical(logits=action_dist).log_prob(actions)

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Value update
        values = self.value_model(states).squeeze()
        value_loss = nn.MSELoss()(values, returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }


class MultiTaskTrainer(BaseTrainer):
    """Trainer for multi-task learning."""

    def __init__(
        self,
        model: nn.Module,
        task_criterions: Dict[str, nn.Module],
        optimizer: optim.Optimizer,
        task_weights: Optional[Dict[str, float]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize multi-task trainer."""
        super().__init__(model, optimizer, None, device)
        self.task_criterions = task_criterions
        self.task_weights = task_weights or {task: 1.0 for task in task_criterions}

    def train_step(self, batch: Dict[str, Any]) -> float:
        """Multi-task training step."""
        self.optimizer.zero_grad()

        total_loss = 0.0
        task_losses = {}

        for task_name, criterion in self.task_criterions.items():
            if task_name in batch:
                inputs = batch[task_name]['input'].to(self.device)
                targets = batch[task_name]['target'].to(self.device)

                outputs = self.model(inputs, task=task_name)
                loss = criterion(outputs, targets)

                weighted_loss = loss * self.task_weights[task_name]
                total_loss += weighted_loss
                task_losses[task_name] = loss.item()

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()


class DistributedTrainer(BaseTrainer):
    """Trainer for distributed training."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
        world_size: int = 1,
        rank: int = 0
    ):
        """Initialize distributed trainer."""
        super().__init__(model, optimizer, criterion, device)
        self.world_size = world_size
        self.rank = rank

        # Wrap model with DDP
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[rank]
            )

    def setup_distributed(self):
        """Setup distributed training."""
        torch.distributed.init_process_group(backend='nccl')

    def cleanup_distributed(self):
        """Cleanup distributed training."""
        torch.distributed.destroy_process_group()
