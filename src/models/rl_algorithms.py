"""Advanced reinforcement learning algorithms."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import numpy as np
from collections import deque


class DQN(nn.Module):
    """Deep Q-Network."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        """Initialize DQN."""
        super().__init__()

        layers = []
        prev_dim = observation_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returns Q-values for all actions."""
        return self.network(obs)

    def get_action(
        self,
        obs: torch.Tensor,
        epsilon: float = 0.0
    ) -> torch.Tensor:
        """Epsilon-greedy action selection."""
        if np.random.random() < epsilon:
            return torch.randint(0, self.network[-1].out_features, (obs.shape[0],))
        else:
            with torch.no_grad():
                return self.forward(obs).argmax(dim=1)


class DoubleDQN(DQN):
    """Double DQN to reduce overestimation bias."""

    def compute_target(
        self,
        next_obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        target_network: nn.Module
    ) -> torch.Tensor:
        """Compute Double DQN target."""
        with torch.no_grad():
            # Use online network to select actions
            next_actions = self.forward(next_obs).argmax(dim=1, keepdim=True)

            # Use target network to evaluate actions
            next_q_values = target_network(next_obs).gather(1, next_actions).squeeze()

            targets = rewards + gamma * next_q_values * (1 - dones)

        return targets


class DuelingDQN(nn.Module):
    """Dueling DQN with separate value and advantage streams."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        """Initialize Dueling DQN."""
        super().__init__()

        # Shared feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass combining value and advantage."""
        features = self.feature_layer(obs)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine using dueling architecture
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


class SoftActorCritic(nn.Module):
    """Soft Actor-Critic (SAC) for continuous control."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        alpha: float = 0.2
    ):
        """Initialize SAC."""
        super().__init__()

        self.action_dim = action_dim
        self.alpha = alpha

        # Actor (policy network)
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        # Two Q-networks (critics)
        self.q1 = self._create_q_network(observation_dim, action_dim, hidden_dim)
        self.q2 = self._create_q_network(observation_dim, action_dim, hidden_dim)

    def _create_q_network(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int
    ) -> nn.Module:
        """Create Q-network."""
        return nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        features = self.actor(obs)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            # Reparameterization trick
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)

            # Compute log probability
            log_prob = normal.log_prob(x_t)
            # Enforce action bounds
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def get_q_values(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values from both critics."""
        obs_action = torch.cat([obs, action], dim=1)
        q1 = self.q1(obs_action)
        q2 = self.q2(obs_action)
        return q1, q2


class TD3(nn.Module):
    """Twin Delayed Deep Deterministic Policy Gradient (TD3)."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 256
    ):
        """Initialize TD3."""
        super().__init__()

        self.action_dim = action_dim
        self.max_action = max_action

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        # Twin Q-networks
        self.q1 = self._create_q_network(observation_dim, action_dim, hidden_dim)
        self.q2 = self._create_q_network(observation_dim, action_dim, hidden_dim)

    def _create_q_network(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int
    ) -> nn.Module:
        """Create Q-network."""
        return nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get deterministic action from actor."""
        return self.max_action * self.actor(obs)

    def get_q_values(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values from both critics."""
        obs_action = torch.cat([obs, action], dim=1)
        q1 = self.q1(obs_action)
        q2 = self.q2(obs_action)
        return q1, q2


class A3C_Network(nn.Module):
    """Asynchronous Advantage Actor-Critic (A3C) network."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        """Initialize A3C network."""
        super().__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy logits and value."""
        features = self.shared(obs)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and get value."""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


class PPOClipNetwork(nn.Module):
    """PPO with clipped objective."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = False
    ):
        """Initialize PPO network."""
        super().__init__()

        self.continuous = continuous

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, entropy, and value."""
        features = self.shared(obs)
        value = self.critic(features)

        if self.continuous:
            mean = self.actor_mean(features)
            std = torch.exp(self.actor_log_std)
            dist = torch.distributions.Normal(mean, std)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        else:
            logits = self.actor(features)
            dist = torch.distributions.Categorical(logits=logits)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


class IQN(nn.Module):
    """Implicit Quantile Network for distributional RL."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_quantiles: int = 64,
        embedding_dim: int = 64
    ):
        """Initialize IQN."""
        super().__init__()

        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.embedding_dim = embedding_dim

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU()
        )

        # Quantile embedding
        self.quantile_embedding = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU()
        )

        # Combined network
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(
        self,
        obs: torch.Tensor,
        num_quantiles: Optional[int] = None
    ) -> torch.Tensor:
        """Forward pass with implicit quantiles."""
        if num_quantiles is None:
            num_quantiles = self.num_quantiles

        batch_size = obs.shape[0]

        # Encode observation
        obs_features = self.obs_encoder(obs)  # (batch, hidden_dim)

        # Sample quantiles
        tau = torch.rand(batch_size, num_quantiles, device=obs.device)  # (batch, num_quantiles)

        # Compute quantile embeddings using cosine basis
        i_pi = np.pi * torch.arange(1, self.embedding_dim + 1, device=obs.device).view(1, 1, -1)
        cos_tau = torch.cos(tau.unsqueeze(-1) * i_pi)  # (batch, num_quantiles, embedding_dim)

        # Embed quantiles
        quantile_features = self.quantile_embedding(cos_tau)  # (batch, num_quantiles, hidden_dim)

        # Combine observation and quantile features
        combined = obs_features.unsqueeze(1) * quantile_features  # (batch, num_quantiles, hidden_dim)

        # Compute quantile values
        quantile_values = self.network(combined)  # (batch, num_quantiles, action_dim)

        return quantile_values, tau


class ReplayBuffer:
    """Experience replay buffer for off-policy RL."""

    def __init__(self, capacity: int):
        """Initialize replay buffer."""
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        """Return buffer size."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        """Initialize prioritized replay buffer."""
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience with maximum priority."""
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4
    ) -> Tuple:
        """Sample batch with priorities."""
        buffer_size = len(self.buffer)
        priorities = self.priorities[:buffer_size]

        # Compute sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(buffer_size, batch_size, p=probs, replace=False)

        # Compute importance sampling weights
        weights = (buffer_size * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        """Return buffer size."""
        return len(self.buffer)
