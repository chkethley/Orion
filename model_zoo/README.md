# Orion Model Zoo

This directory contains pre-trained models and model configurations for the Orion AGI system.

## Directory Structure

```
model_zoo/
├── configs/          # Model configuration files
├── checkpoints/      # Pre-trained model weights
├── onnx/            # ONNX exported models
├── architectures/   # Architecture definitions
└── benchmarks/      # Benchmark results
```

## Available Models

### Reasoning Models

**reasoning_network_v1**
- Architecture: Multi-layer reasoning network with GRU cells
- Input: 64-dim vectors
- Output: 32-dim reasoning embeddings
- Parameters: ~150K
- Performance: TBD

### Memory Models

**memory_network_v1**
- Architecture: Attention-based memory network
- Memory Size: 100 slots
- Parameters: ~200K
- Use Case: Working memory simulation

### Generative Models

**vae_v1**
- Architecture: Variational Autoencoder
- Latent Dim: 64
- Parameters: ~500K
- Use Case: Representation learning

**diffusion_v1**
- Architecture: Denoising Diffusion Model
- Timesteps: 1000
- Parameters: ~1M
- Use Case: Sample generation

### RL Models

**policy_network_v1**
- Architecture: Actor network for discrete actions
- Observation: 64-dim
- Actions: 4 discrete
- Parameters: ~80K

**dqn_v1**
- Architecture: Deep Q-Network
- Observation: 64-dim
- Actions: 4
- Parameters: ~100K

### Multi-Modal Models

**multimodal_encoder_v1**
- Architecture: Text + Vision + Audio fusion
- Modalities: 3
- Parameters: ~2M
- Use Case: Multi-modal understanding

## Usage

### Loading Pre-trained Models

```python
import torch
from pathlib import Path

# Load PyTorch checkpoint
model_path = Path("model_zoo/checkpoints/reasoning_network_v1.pt")
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
```

### Loading ONNX Models

```python
from src.models.onnx_utils import ONNXInferenceEngine

model_path = "model_zoo/onnx/reasoning_network_v1.onnx"
engine = ONNXInferenceEngine(model_path)

# Inference
import numpy as np
input_data = np.random.randn(1, 64).astype(np.float32)
output = engine.infer(input_data)
```

### Using Pre-configured Architectures

```python
import json
from src.models.neural_architecture_search import build_model_from_architecture

# Load architecture config
with open("model_zoo/configs/reasoning_network_v1.json") as f:
    arch_config = json.load(f)

# Build model
model = build_model_from_architecture(
    architecture=arch_config,
    input_dim=64,
    output_dim=32
)
```

## Training Your Own Models

See the training scripts in `scripts/training/` for examples of training models.

```bash
# Train reasoning network
python scripts/training/train_models.py \
    --model reasoning \
    --epochs 50 \
    --export-onnx \
    --output-dir model_zoo/checkpoints

# Export to ONNX
python scripts/training/export_all_models.py
```

## Benchmarks

All models in the zoo have been benchmarked on standard hardware. See `benchmarks/` for detailed results.

Example benchmark results:

| Model | Latency (ms) | Throughput (samples/s) | Memory (MB) |
|-------|--------------|------------------------|-------------|
| reasoning_network_v1 | 2.5 | 400 | 50 |
| memory_network_v1 | 3.8 | 263 | 75 |
| policy_network_v1 | 1.2 | 833 | 30 |

## Contributing Models

To contribute a new model to the zoo:

1. Train and validate your model
2. Export to ONNX format
3. Create configuration file
4. Run benchmarks
5. Document in this README
6. Submit pull request

## License

All models in the zoo are released under the MIT License unless otherwise specified.

## Citation

If you use these models in your research, please cite:

```bibtex
@software{orion_model_zoo_2024,
  title = {Orion AGI Model Zoo},
  author = {Orion Team},
  year = {2024},
  url = {https://github.com/your-org/orion-agi}
}
```
