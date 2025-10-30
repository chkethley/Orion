"""FastAPI application for Orion AGI."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.core.config import settings
from src.agents.agi_agent import AGIAgent
from src.core.base import Goal


app = FastAPI(
    title="Orion AGI API",
    description="API for interacting with the Orion AGI system",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance (in production, use proper agent management)
agents: Dict[str, AGIAgent] = {}


# Request/Response Models
class CreateAgentRequest(BaseModel):
    """Request model for creating an agent."""
    name: str
    capabilities: List[str]
    config: Optional[Dict[str, Any]] = None


class SetGoalRequest(BaseModel):
    """Request model for setting agent goal."""
    description: str
    priority: int = 1
    deadline: Optional[str] = None


class ExecuteCycleRequest(BaseModel):
    """Request model for executing agent cycle."""
    environment: Dict[str, Any]


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Orion AGI API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": settings.environment
    }


@app.post("/agents", status_code=201)
async def create_agent(request: CreateAgentRequest):
    """Create a new AGI agent."""
    agent_id = f"agent_{len(agents) + 1}"

    agent = AGIAgent(
        agent_id=agent_id,
        name=request.name,
        capabilities=request.capabilities,
        config=request.config or {}
    )

    agents[agent_id] = agent

    return {
        "agent_id": agent_id,
        "name": request.name,
        "capabilities": request.capabilities,
        "status": "created"
    }


@app.get("/agents")
async def list_agents():
    """List all agents."""
    return {
        "agents": [
            {
                "agent_id": agent_id,
                "name": agent.name,
                "capabilities": agent.capabilities
            }
            for agent_id, agent in agents.items()
        ]
    }


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get agent details."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = agents[agent_id]
    return agent.get_status()


@app.post("/agents/{agent_id}/goals")
async def set_goal(agent_id: str, request: SetGoalRequest):
    """Set a goal for the agent."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = agents[agent_id]

    goal = Goal(
        id=f"goal_{len(agent.goals) + 1}",
        description=request.description,
        priority=request.priority,
        status="active",
        created_at=datetime.now(),
        deadline=datetime.fromisoformat(request.deadline) if request.deadline else None
    )

    agent.add_goal(goal)
    agent.current_goal = goal

    return {
        "goal_id": goal.id,
        "description": goal.description,
        "status": "set"
    }


@app.post("/agents/{agent_id}/execute")
async def execute_cycle(agent_id: str, request: ExecuteCycleRequest):
    """Execute one agent cycle."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = agents[agent_id]

    try:
        result = await agent.run_cycle(request.environment)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/{agent_id}/memory")
async def get_memory_stats(agent_id: str):
    """Get agent memory statistics."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = agents[agent_id]
    return agent.memory_system.get_statistics()


@app.get("/agents/{agent_id}/learning")
async def get_learning_history(agent_id: str, limit: int = 10):
    """Get agent learning history."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = agents[agent_id]
    return {
        "history": agent.learner.get_learning_history(limit=limit),
        "evaluation": agent.learner.evaluate_learning()
    }


@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    del agents[agent_id]
    return {"status": "deleted", "agent_id": agent_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
