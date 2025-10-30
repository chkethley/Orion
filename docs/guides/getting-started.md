# Getting Started with Orion AGI

This guide will help you get started with the Orion AGI development hub.

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Git
- 8GB RAM minimum (16GB recommended)
- CUDA-capable GPU (optional, for training)

## Installation

### Option 1: Local Development Setup

1. **Clone the repository**

```bash
git clone https://github.com/your-org/orion-agi.git
cd orion-agi
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
make install-dev
# or
pip install -e ".[dev]"
```

4. **Set up environment variables**

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. **Initialize pre-commit hooks**

```bash
pre-commit install
```

### Option 2: Docker Setup

1. **Clone the repository**

```bash
git clone https://github.com/your-org/orion-agi.git
cd orion-agi
```

2. **Configure environment**

```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start services**

```bash
make docker-up
# or
docker-compose up -d
```

4. **Verify installation**

```bash
docker-compose ps
```

## Quick Start

### 1. Start the API Server

**Local:**
```bash
make serve
# or
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Docker:**
```bash
docker-compose up api
```

The API will be available at http://localhost:8000

### 2. Create Your First Agent

```python
import requests

# Create an agent
response = requests.post(
    "http://localhost:8000/agents",
    json={
        "name": "My First Agent",
        "capabilities": ["reasoning", "learning", "planning"],
        "config": {
            "short_term_capacity": 10,
            "working_capacity": 7
        }
    }
)

agent_data = response.json()
agent_id = agent_data["agent_id"]
print(f"Created agent: {agent_id}")
```

### 3. Set a Goal for the Agent

```python
# Set a goal
response = requests.post(
    f"http://localhost:8000/agents/{agent_id}/goals",
    json={
        "description": "Learn to navigate a simple environment",
        "priority": 1
    }
)

print(f"Goal set: {response.json()}")
```

### 4. Execute Agent Cycle

```python
# Execute one cycle
response = requests.post(
    f"http://localhost:8000/agents/{agent_id}/execute",
    json={
        "environment": {
            "position": [0, 0],
            "objects": ["wall", "door", "key"],
            "state": "exploring"
        }
    }
)

result = response.json()
print(f"Cycle completed: {result['cycle']}")
print(f"Learning complete: {result['learning_complete']}")
```

### 5. Check Agent Status

```python
# Get agent status
response = requests.get(f"http://localhost:8000/agents/{agent_id}")
status = response.json()

print(f"Agent: {status['name']}")
print(f"Interactions: {status['interaction_count']}")
print(f"Memory stats: {status['memory_statistics']}")
```

## Using Python Directly

You can also use the Orion components directly in Python:

```python
from src.agents.agi_agent import AGIAgent
from src.core.base import Goal
from datetime import datetime

# Create agent
agent = AGIAgent(
    agent_id="agent_1",
    name="Direct Agent",
    capabilities=["reasoning", "learning"],
    config={}
)

# Set goal
goal = Goal(
    id="goal_1",
    description="Explore environment",
    priority=1,
    status="active",
    created_at=datetime.now()
)
agent.add_goal(goal)
agent.current_goal = goal

# Run cycle
import asyncio

async def main():
    environment = {"test": "environment"}
    result = await agent.run_cycle(environment)
    print(f"Result: {result}")

asyncio.run(main())
```

## Exploring the System

### 1. Jupyter Notebooks

Start Jupyter for interactive exploration:

```bash
make jupyter
# or
docker-compose up jupyter
```

Access at http://localhost:8888

### 2. MLflow Experiment Tracking

View experiments at http://localhost:5000

```bash
# MLflow is automatically started with docker-compose
docker-compose up mlflow
```

### 3. Grafana Monitoring

View metrics at http://localhost:3000 (admin/admin)

```bash
docker-compose up grafana prometheus
```

## Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/unit/test_memory.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Training Models

```bash
# Run training script
python scripts/training/train.py \
    --experiment-name my_experiment \
    --epochs 10 \
    --batch-size 32 \
    --use-mlflow
```

## Next Steps

- Read the [Architecture Overview](../architecture/overview.md)
- Explore [API Documentation](../api/reference.md)
- Try the [Examples](./examples.md)
- Read [Development Guide](./development.md)

## Common Issues

### Port Already in Use

If ports are already in use, edit `docker-compose.yml` to change port mappings.

### CUDA/GPU Issues

For GPU support, use the GPU-enabled Docker target:

```bash
docker build --target gpu -t orion-agi:gpu .
```

### Memory Issues

Increase Docker memory allocation in Docker Desktop settings (minimum 8GB).

## Getting Help

- Check the [documentation](./README.md)
- Open an [issue](https://github.com/your-org/orion-agi/issues)
- Join our community chat

## Clean Up

```bash
# Stop services
make docker-down

# Remove containers and volumes
docker-compose down -v

# Clean build artifacts
make clean
```
