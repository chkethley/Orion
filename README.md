# Orion AGI Development Hub

> Unified AI Development Platform for Artificial General Intelligence

[![CI/CD](https://github.com/orion/orion-agi/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/orion/orion-agi/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Orion is a comprehensive development hub for building and deploying Artificial General Intelligence systems. It provides a complete framework with memory systems, reasoning engines, planning capabilities, and adaptive learning mechanisms.

## Key Features

- **Advanced Memory System**: Multi-tiered memory architecture (short-term, working, long-term, episodic, semantic, procedural)
- **Sophisticated Reasoning Engine**: Multiple reasoning strategies (deductive, inductive, abductive, analogical, causal, probabilistic)
- **Goal-Oriented Planning**: Hierarchical task decomposition with dependency management
- **Adaptive Learning**: Multiple learning strategies (supervised, reinforcement, transfer, meta-learning)
- **Autonomous Agents**: Complete perception-reasoning-planning-action-learning cycles
- **Experiment Tracking**: Integration with MLflow and Weights & Biases
- **Data Pipelines**: Flexible multi-stage data processing infrastructure
- **Production-Ready API**: FastAPI-based REST API with comprehensive endpoints
- **PyTorch Models**: 20+ neural architectures (Transformers, Memory Networks, World Models, RL agents)
- **Generative Models**: VAE, GAN, Diffusion, Flow Matching, Normalizing Flows
- **Advanced RL**: DQN, Double DQN, Dueling DQN, SAC, TD3, PPO, A3C, IQN with replay buffers
- **ONNX Export & Inference**: Optimized model deployment with ONNX Runtime
- **Model Serving API**: Production-ready REST API for model inference with benchmarking
- **Interactive Chat Interface**: Web-based chat UI with real-time messaging and model switching
- **Neural Architecture Search**: DARTS, Evolutionary NAS, Hyperparameter optimization
- **Multi-Agent Systems**: Collaborative agents, swarm intelligence, auction-based coordination
- **Comprehensive Benchmarking**: System-wide performance profiling
- **Model Zoo**: Pre-configured architectures and training recipes
- **Containerized Deployment**: Docker and Kubernetes support
- **Comprehensive Testing**: Unit, integration, and end-to-end tests

## Architecture

```
┌─────────────────────────────────────────────┐
│           API Layer (FastAPI)               │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────┴──────────────────────────┐
│              AGI Agent                      │
│  ┌─────────┬─────────┬─────────┬─────────┐ │
│  │Perceive │ Reason  │  Plan   │   Act   │ │
│  └────┬────┴────┬────┴────┬────┴────┬────┘ │
│       │         │         │         │      │
│  ┌────┴─────────┴─────────┴─────────┴───┐  │
│  │            Learn                     │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
          │         │         │
    ┌─────┴───┐ ┌──┴───┐ ┌──┴─────┐
    │ Memory  │ │Reason│ │Planning│
    │ System  │ │Engine│ │ System │
    └─────────┘ └──────┘ └────────┘
```

## Quick Start

### Using Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/your-org/orion-agi.git
cd orion-agi

# Copy environment configuration
cp .env.example .env

# Start all services
make docker-up

# Access services:
# - API: http://localhost:8000
# - Jupyter: http://localhost:8888
# - MLflow: http://localhost:5000
# - Grafana: http://localhost:3000
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
make install-dev

# Run tests
make test

# Start main API server
make serve

# Start chat interface (separate terminal)
python scripts/start_chat.py
# Chat will be available at http://localhost:8002
```

## Usage Example

```python
from src.agents.agi_agent import AGIAgent
from src.core.base import Goal
from datetime import datetime
import asyncio

# Create an AGI agent
agent = AGIAgent(
    agent_id="agent_1",
    name="Research Assistant",
    capabilities=["reasoning", "learning", "planning"],
    config={
        "short_term_capacity": 10,
        "working_capacity": 7
    }
)

# Set a goal
goal = Goal(
    id="goal_1",
    description="Analyze research papers and summarize findings",
    priority=1,
    status="active",
    created_at=datetime.now()
)
agent.add_goal(goal)
agent.current_goal = goal

# Execute agent cycle
async def run():
    environment = {
        "papers": ["paper1.pdf", "paper2.pdf"],
        "context": "machine learning"
    }
    result = await agent.run_cycle(environment)
    print(f"Agent completed cycle {result['cycle']}")
    print(f"Status: {agent.get_status()}")

asyncio.run(run())
```

## Project Structure

```
orion-agi/
├── src/                      # Source code
│   ├── core/                # Core components
│   ├── agents/              # Agent implementations
│   ├── memory/              # Memory system
│   ├── reasoning/           # Reasoning engine
│   ├── planning/            # Planning system
│   ├── learning/            # Learning framework
│   ├── perception/          # Perception modules
│   ├── models/              # ML models
│   ├── api/                 # API endpoints
│   └── utils/               # Utilities
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── e2e/                # End-to-end tests
├── data/                    # Data storage
├── configs/                 # Configuration files
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
├── experiments/            # Experiment tracking
└── deployments/            # Deployment configs
```

## Documentation

- [Getting Started Guide](docs/guides/getting-started.md)
- [Architecture Overview](docs/architecture/overview.md)
- [PyTorch & ONNX Guide](docs/guides/pytorch_onnx_guide.md)
- [Chat Interface Guide](docs/guides/chat_interface.md)
- [API Reference](docs/api/reference.md)
- [Development Guide](docs/guides/development.md)
- [Examples](docs/guides/examples.md)

## Development

### Running Tests

```bash
# All tests with coverage
make test

# Specific test file
pytest tests/unit/test_memory.py -v

# Integration tests only
pytest tests/integration/ -v
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Type checking
mypy src/
```

### Training Models

```bash
# Train transformer encoder
python scripts/training/train_models.py \
    --model transformer \
    --epochs 20 \
    --batch-size 32 \
    --export-onnx

# Train reasoning network
python scripts/training/train_models.py \
    --model reasoning \
    --epochs 15 \
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

# Export all models to ONNX
python scripts/training/export_all_models.py
```

### ONNX Inference

```bash
# Run inference with ONNX model
python scripts/inference/onnx_inference.py \
    --model-path data/models/onnx/reasoning_network.onnx \
    --benchmark \
    --num-runs 1000

# Compare multiple models
python scripts/inference/onnx_inference.py \
    --compare-models \
        data/models/onnx/model1.onnx \
        data/models/onnx/model2.onnx
```

## API Endpoints

### Main API (Port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/agents` | POST | Create agent |
| `/agents` | GET | List agents |
| `/agents/{id}` | GET | Get agent details |
| `/agents/{id}/goals` | POST | Set agent goal |
| `/agents/{id}/execute` | POST | Execute agent cycle |
| `/agents/{id}/memory` | GET | Get memory stats |
| `/agents/{id}/learning` | GET | Get learning history |

### Chat API (Port 8002)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface (HTML) |
| `/chat` | POST | Send chat message |
| `/sessions/{id}/history` | GET | Get chat history |
| `/sessions/{id}` | DELETE | Delete session |
| `/models` | GET | List available models |
| `/ws/{id}` | WS | WebSocket chat stream |

See [Chat Interface Guide](docs/guides/chat_interface.md) for detailed usage.

## Technology Stack

- **Framework**: Python 3.10+
- **Web API**: FastAPI (2 separate APIs: Main + Model Serving)
- **ML/AI**: PyTorch, ONNX, ONNX Runtime, Transformers, LangChain
- **Models**: 20+ architectures (Transformers, VAE, GAN, Diffusion, DQN, SAC, PPO, etc.)
- **Experiment Tracking**: MLflow, Weights & Biases
- **Vector DB**: ChromaDB, Pinecone
- **Database**: PostgreSQL
- **Cache/Queue**: Redis, Celery
- **Monitoring**: Prometheus, Grafana
- **Containerization**: Docker, Kubernetes
- **Testing**: Pytest, comprehensive benchmarking suite
- **CI/CD**: GitHub Actions
- **NAS**: DARTS, Evolutionary algorithms, Bayesian optimization

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/guides/contributing.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Roadmap

- [ ] Multi-agent collaboration system
- [ ] Advanced neural architecture search
- [ ] Distributed training support
- [ ] Real-time learning capabilities
- [ ] Enhanced visualization dashboard
- [ ] Mobile app interface
- [ ] Cloud deployment templates

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Orion in your research, please cite:

```bibtex
@software{orion_agi_2024,
  title = {Orion AGI Development Hub},
  author = {Orion Team},
  year = {2024},
  url = {https://github.com/your-org/orion-agi}
}
```

## Acknowledgments

- Research community for AGI foundations
- Open source libraries and frameworks
- Contributors and testers

## Contact

- Website: https://orion-agi.io
- Email: contact@orion-agi.io
- GitHub: https://github.com/your-org/orion-agi

---

**Built with by the Orion Team**
