# PyTorch Training and ONNX Export Guide

This guide covers training PyTorch models and exporting them to ONNX format for optimized inference.

## Table of Contents

1. [Model Architectures](#model-architectures)
2. [Training Models](#training-models)
3. [Exporting to ONNX](#exporting-to-onnx)
4. [ONNX Inference](#onnx-inference)
5. [Model Optimization](#model-optimization)
6. [Best Practices](#best-practices)

## Model Architectures

Orion provides several pre-built PyTorch architectures:

### 1. Transformer Encoder

```python
from src.models.neural_architectures import TransformerEncoder

model = TransformerEncoder(
    vocab_size=10000,
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048,
    dropout=0.1
)
```

**Use cases**: Language modeling, sequence processing, text generation

### 2. Memory Network

```python
from src.models.neural_architectures import MemoryNetwork

model = MemoryNetwork(
    input_dim=128,
    hidden_dim=256,
    memory_size=100,
    num_heads=4
)
```

**Use cases**: Working memory simulation, attention mechanisms, context retention

### 3. Reasoning Network

```python
from src.models.neural_architectures import ReasoningNetwork

model = ReasoningNetwork(
    input_dim=128,
    hidden_dim=256,
    output_dim=64,
    num_reasoning_steps=3
)
```

**Use cases**: Multi-step reasoning, problem solving, logical inference

### 4. World Model

```python
from src.models.neural_architectures import WorldModel

model = WorldModel(
    observation_dim=64,
    action_dim=4,
    hidden_dim=256,
    latent_dim=64
)
```

**Use cases**: Model-based RL, environment prediction, planning

### 5. Policy & Value Networks

```python
from src.models.neural_architectures import PolicyNetwork, ValueNetwork

policy = PolicyNetwork(
    observation_dim=64,
    action_dim=4,
    hidden_dim=256,
    continuous=False
)

value = ValueNetwork(
    observation_dim=64,
    hidden_dim=256
)
```

**Use cases**: Reinforcement learning, decision making, action selection

## Training Models

### Basic Training with BaseTrainer

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.neural_architectures import ReasoningNetwork
from src.models.pytorch_trainer import BaseTrainer

# Create model
model = ReasoningNetwork(
    input_dim=64,
    hidden_dim=128,
    output_dim=32
)

# Prepare data
X = torch.randn(1000, 64)
y = torch.randn(1000, 32)
dataset = TensorDataset(X, y)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Setup trainer
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

trainer = BaseTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    gradient_clip=1.0
)

# Train
trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    save_dir="checkpoints/reasoning",
    early_stopping_patience=5
)
```

### Training with Command Line

```bash
# Train transformer
python scripts/training/train_models.py \
    --model transformer \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --export-onnx

# Train memory network
python scripts/training/train_models.py \
    --model memory \
    --epochs 15 \
    --batch-size 16 \
    --input-dim 128 \
    --hidden-dim 256 \
    --export-onnx

# Train RL agent
python scripts/training/train_models.py \
    --model rl \
    --epochs 100 \
    --observation-dim 64 \
    --action-dim 4 \
    --export-onnx

# Train world model
python scripts/training/train_models.py \
    --model world \
    --epochs 50 \
    --observation-dim 64 \
    --action-dim 4 \
    --latent-dim 32 \
    --export-onnx
```

### Reinforcement Learning Training

```python
from src.models.pytorch_trainer import ReinforcementLearningTrainer

policy_optimizer = optim.Adam(policy.parameters(), lr=0.0003)
value_optimizer = optim.Adam(value.parameters(), lr=0.001)

rl_trainer = ReinforcementLearningTrainer(
    policy_model=policy,
    value_model=value,
    policy_optimizer=policy_optimizer,
    value_optimizer=value_optimizer,
    gamma=0.99,
    gae_lambda=0.95
)

# Training loop with environment interaction
for episode in range(num_episodes):
    states, actions, rewards, dones = collect_episode_data()

    # Compute advantages
    with torch.no_grad():
        values = value_model(states)
    advantages = rl_trainer.compute_advantages(rewards, values, dones)

    # PPO update
    losses = rl_trainer.train_step_ppo(
        states, actions, old_log_probs, advantages, returns
    )
```

## Exporting to ONNX

### Method 1: Using ModelConverter

```python
from src.models.onnx_utils import ModelConverter

# Export model
onnx_path = ModelConverter.pytorch_to_onnx(
    model=model,
    model_name="my_model",
    dummy_input=torch.randn(1, 64),
    output_dir="data/models/onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"}
    }
)
```

### Method 2: Using ONNXExporter

```python
from src.models.onnx_utils import ONNXExporter

exporter = ONNXExporter(model, "my_model")

# Export with validation
onnx_path = exporter.export_with_validation(
    dummy_input=torch.randn(1, 64),
    output_path="models/my_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch", 1: "sequence"}}
)
```

### Batch Export All Models

```python
# Export all common models
from src.models.onnx_utils import export_common_models

exported_paths = export_common_models()
```

Or via command line:

```bash
python scripts/training/export_all_models.py
```

### Dynamic Axes

Dynamic axes allow variable batch sizes and sequence lengths:

```python
dynamic_axes = {
    "input_ids": {
        0: "batch",      # Variable batch size
        1: "sequence"    # Variable sequence length
    },
    "output": {
        0: "batch",
        1: "sequence"
    }
}
```

## ONNX Inference

### Basic Inference

```python
from src.models.onnx_utils import ONNXInferenceEngine
import numpy as np

# Load model
engine = ONNXInferenceEngine("models/my_model.onnx")

# Run inference
input_data = np.random.randn(1, 64).astype(np.float32)
output = engine.infer(input_data)
```

### Batch Inference

```python
# Process large batch
data = np.random.randn(10000, 64).astype(np.float32)
results = engine.batch_infer(data, batch_size=32)
```

### Model Information

```python
# Get model metadata
info = engine.get_model_info()
print(f"Inputs: {info['input_names']}")
print(f"Outputs: {info['output_names']}")
print(f"Input shapes: {info['inputs']}")
```

### Command Line Inference

```bash
# Single model inference
python scripts/inference/onnx_inference.py \
    --model-path models/reasoning_network.onnx \
    --model-type reasoning \
    --batch-size 1 \
    --benchmark \
    --num-runs 1000

# Compare multiple models
python scripts/inference/onnx_inference.py \
    --compare-models \
        models/model1.onnx \
        models/model2.onnx \
        models/model3.onnx \
    --benchmark
```

## Model Optimization

### Quantization

Quantization reduces model size and improves inference speed:

```python
from src.models.onnx_utils import ModelConverter

# Quantize model
quantized_path = ModelConverter.quantize_onnx(
    model_path="models/my_model.onnx",
    output_path="models/my_model_quantized.onnx",
    quantization_mode="dynamic"
)
```

### ONNX Graph Optimization

```python
# Optimize ONNX graph
optimized_path = ModelConverter.optimize_onnx(
    model_path="models/my_model.onnx",
    output_path="models/my_model_optimized.onnx"
)
```

### Execution Providers

Use different execution providers for acceleration:

```python
# GPU inference
engine = ONNXInferenceEngine(
    "models/my_model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# CPU-only inference
engine = ONNXInferenceEngine(
    "models/my_model.onnx",
    providers=['CPUExecutionProvider']
)

# TensorRT (NVIDIA GPUs)
engine = ONNXInferenceEngine(
    "models/my_model.onnx",
    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider']
)
```

## Best Practices

### 1. Model Design

- Keep models simple for ONNX compatibility
- Avoid dynamic control flow when possible
- Use standard PyTorch operations
- Test export early in development

### 2. Training

- Use gradient clipping to prevent exploding gradients
- Implement early stopping to avoid overfitting
- Save checkpoints regularly
- Log metrics for monitoring

### 3. ONNX Export

- Always validate exports against PyTorch outputs
- Use dynamic axes for flexibility
- Test with different input shapes
- Document input/output specifications

### 4. Inference

- Warm up models before benchmarking
- Use appropriate batch sizes
- Enable provider optimizations
- Profile for bottlenecks

### 5. Production Deployment

- Quantize models when possible
- Optimize ONNX graphs
- Use appropriate execution providers
- Implement proper error handling
- Monitor inference latency

## Common Issues and Solutions

### Issue: Export Fails

**Solution**: Check for unsupported operations. Use `torch.onnx.export` with `verbose=True` to see details.

### Issue: Output Mismatch

**Solution**: Ensure model is in eval mode. Check for operations with randomness.

### Issue: Slow Inference

**Solution**:
- Use GPU execution provider
- Quantize model
- Optimize batch size
- Enable graph optimizations

### Issue: Dynamic Shapes Not Working

**Solution**: Properly define dynamic axes in export. Test with different input shapes.

## Example Workflows

### Complete Training and Deployment Pipeline

```bash
# 1. Train model
python scripts/training/train_models.py \
    --model reasoning \
    --epochs 50 \
    --export-onnx \
    --output-dir models/trained

# 2. Optimize model
python -c "
from src.models.onnx_utils import ModelConverter
ModelConverter.optimize_onnx('models/trained/reasoning_network/onnx/reasoning_network.onnx')
ModelConverter.quantize_onnx('models/trained/reasoning_network/onnx/reasoning_network_optimized.onnx')
"

# 3. Benchmark
python scripts/inference/onnx_inference.py \
    --model-path models/trained/reasoning_network/onnx/reasoning_network_quantized.onnx \
    --benchmark \
    --num-runs 1000

# 4. Deploy (integrate into application)
```

### Jupyter Notebook Workflow

See `experiments/notebooks/pytorch_training_tutorial.ipynb` for interactive examples.

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [ONNX Documentation](https://onnx.ai/onnx/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Model Architecture Reference](../architecture/overview.md)
- [API Reference](../api/reference.md)
