"""Advanced generative models for AGI."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder (VAE)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 256, 128],
        latent_dim: int = 64,
        beta: float = 1.0
    ):
        """Initialize VAE."""
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute VAE loss (ELBO)."""
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='sum')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        loss = recon_loss + self.beta * kl_loss

        return loss, {
            'total_loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }

    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Sample from latent space."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


class ConditionalVAE(VariationalAutoencoder):
    """Conditional VAE with class conditioning."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list = [512, 256, 128],
        latent_dim: int = 64,
        beta: float = 1.0
    ):
        """Initialize conditional VAE."""
        super().__init__(input_dim + num_classes, hidden_dims, latent_dim, beta)
        self.num_classes = num_classes
        self.input_dim = input_dim

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with conditioning."""
        # One-hot encode labels
        labels_onehot = F.one_hot(labels, self.num_classes).float()

        # Concatenate input with labels
        x_cond = torch.cat([x, labels_onehot], dim=1)

        mu, logvar = self.encode(x_cond)
        z = self.reparameterize(mu, logvar)

        # Concatenate z with labels for decoding
        z_cond = torch.cat([z, labels_onehot], dim=1)
        recon = self.decode(z_cond)[:, :self.input_dim]  # Remove label part

        return recon, mu, logvar


class GenerativeAdversarialNetwork(nn.Module):
    """Basic GAN architecture."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: list = [128, 256, 512]
    ):
        """Initialize GAN."""
        super().__init__()

        self.latent_dim = latent_dim

        # Generator
        gen_layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            gen_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        gen_layers.append(nn.Linear(hidden_dims[-1], output_dim))
        gen_layers.append(nn.Tanh())
        self.generator = nn.Sequential(*gen_layers)

        # Discriminator
        disc_layers = []
        prev_dim = output_dim
        for hidden_dim in reversed(hidden_dims):
            disc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        disc_layers.append(nn.Linear(hidden_dims[0], 1))
        disc_layers.append(nn.Sigmoid())
        self.discriminator = nn.Sequential(*disc_layers)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """Generate samples from noise."""
        return self.generator(z)

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminate real vs fake."""
        return self.discriminator(x)

    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Sample from the generator."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.generate(z)


class DiffusionModel(nn.Module):
    """Denoising Diffusion Probabilistic Model (DDPM)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        """Initialize diffusion model."""
        super().__init__()

        self.input_dim = input_dim
        self.num_timesteps = num_timesteps

        # Define beta schedule
        self.register_buffer(
            'betas',
            torch.linspace(beta_start, beta_end, num_timesteps)
        )
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer(
            'alphas_cumprod_prev',
            F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        )

        # Denoising network (time-conditioned)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.denoiser = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward_diffusion(
        self,
        x0: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to data (forward process)."""
        noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1)

        # q(x_t | x_0)
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

        return xt, noise

    def predict_noise(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise at timestep t."""
        # Time embedding
        t_normalized = t.float() / self.num_timesteps
        t_embed = self.time_embed(t_normalized.view(-1, 1))

        # Concatenate and denoise
        xt_with_time = torch.cat([xt, t_embed], dim=1)
        predicted_noise = self.denoiser(xt_with_time)

        return predicted_noise

    def forward(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass."""
        batch_size = x0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x0.device)

        # Forward diffusion
        xt, noise = self.forward_diffusion(x0, t)

        # Predict noise
        predicted_noise = self.predict_noise(xt, t)

        return predicted_noise, noise

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Sample from the model (reverse diffusion)."""
        # Start from pure noise
        xt = torch.randn(num_samples, self.input_dim).to(device)

        # Reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.predict_noise(xt, t_tensor)

            # Compute coefficients
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            # Reverse step
            if t > 0:
                noise = torch.randn_like(xt)
            else:
                noise = 0

            xt = (
                1 / alpha_t.sqrt() * (
                    xt - (beta_t / (1 - alpha_cumprod_t).sqrt()) * predicted_noise
                )
                + beta_t.sqrt() * noise
            )

        return xt


class FlowMatchingModel(nn.Module):
    """Flow Matching / Rectified Flow model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        """Initialize flow matching model."""
        super().__init__()

        self.input_dim = input_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Vector field network
        layers = []
        layers.append(nn.Linear(input_dim + hidden_dim, hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, input_dim))

        self.velocity_net = nn.Sequential(*layers)

    def predict_velocity(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity field at time t."""
        # Time embedding
        t_embed = self.time_embed(t.view(-1, 1))

        # Concatenate and predict
        xt_with_time = torch.cat([xt, t_embed], dim=1)
        velocity = self.velocity_net(xt_with_time)

        return velocity

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass."""
        batch_size = x0.shape[0]

        # Sample random times
        t = torch.rand(batch_size, device=x0.device)

        # Linear interpolation
        xt = (1 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x1

        # Target velocity (x1 - x0)
        target_velocity = x1 - x0

        # Predicted velocity
        predicted_velocity = self.predict_velocity(xt, t)

        return predicted_velocity, target_velocity

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        num_steps: int = 100,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Sample using ODE solver."""
        # Start from noise
        xt = torch.randn(num_samples, self.input_dim).to(device)

        dt = 1.0 / num_steps

        # Euler integration
        for step in range(num_steps):
            t = torch.full((num_samples,), step * dt, device=device)
            velocity = self.predict_velocity(xt, t)
            xt = xt + velocity * dt

        return xt


class NormalizingFlow(nn.Module):
    """Normalizing Flow model with coupling layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 8
    ):
        """Initialize normalizing flow."""
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers

        # Coupling layers
        self.coupling_layers = nn.ModuleList([
            AffineCouplingLayer(input_dim, hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (data to latent)."""
        log_det_jacobian = torch.zeros(x.shape[0], device=x.device)

        for layer in self.coupling_layers:
            x, ldj = layer(x)
            log_det_jacobian += ldj

        return x, log_det_jacobian

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse pass (latent to data)."""
        for layer in reversed(self.coupling_layers):
            z = layer.inverse(z)

        return z

    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Sample from the flow."""
        z = torch.randn(num_samples, self.input_dim).to(device)
        return self.inverse(z)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability."""
        z, log_det = self.forward(x)

        # Standard normal log prob
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=1)

        # Change of variables
        log_px = log_pz + log_det

        return log_px


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flows."""

    def __init__(self, input_dim: int, hidden_dim: int):
        """Initialize coupling layer."""
        super().__init__()

        self.input_dim = input_dim
        self.half_dim = input_dim // 2

        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - self.half_dim),
            nn.Tanh()
        )

        self.translation_net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - self.half_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward transformation."""
        x1, x2 = x[:, :self.half_dim], x[:, self.half_dim:]

        s = self.scale_net(x1)
        t = self.translation_net(x1)

        y2 = x2 * torch.exp(s) + t
        y = torch.cat([x1, y2], dim=1)

        log_det = s.sum(dim=1)

        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse transformation."""
        y1, y2 = y[:, :self.half_dim], y[:, self.half_dim:]

        s = self.scale_net(y1)
        t = self.translation_net(y1)

        x2 = (y2 - t) * torch.exp(-s)
        x = torch.cat([y1, x2], dim=1)

        return x
