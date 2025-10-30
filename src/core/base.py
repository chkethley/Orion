"""Base classes and interfaces for the Orion AGI framework."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ComponentStatus(Enum):
    """Status of a component in the system."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class Message:
    """Represents a message in the system."""
    content: str
    role: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }


@dataclass
class Goal:
    """Represents a goal for the AGI system."""
    id: str
    description: str
    priority: int
    status: str
    created_at: datetime
    deadline: Optional[datetime] = None
    parent_goal_id: Optional[str] = None
    sub_goals: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "parent_goal_id": self.parent_goal_id,
            "sub_goals": self.sub_goals or []
        }


class BaseComponent(ABC):
    """Abstract base class for all AGI components."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the component."""
        self.name = name
        self.config = config or {}
        self.status = ComponentStatus.IDLE
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""
        pass

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input and return output."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    def get_status(self) -> ComponentStatus:
        """Get current status of the component."""
        return self.status

    def set_status(self, status: ComponentStatus) -> None:
        """Set the status of the component."""
        self.status = status


class BaseAgent(ABC):
    """Abstract base class for intelligent agents."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        capabilities: List[str],
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the agent."""
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.config = config or {}
        self.memory = []
        self.goals = []

    @abstractmethod
    async def perceive(self, environment: Any) -> Dict[str, Any]:
        """Perceive the environment."""
        pass

    @abstractmethod
    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Reason about the perception."""
        pass

    @abstractmethod
    async def plan(self, reasoning: Dict[str, Any]) -> List[Any]:
        """Create a plan based on reasoning."""
        pass

    @abstractmethod
    async def act(self, plan: List[Any]) -> Any:
        """Execute the plan."""
        pass

    @abstractmethod
    async def learn(self, experience: Dict[str, Any]) -> None:
        """Learn from experience."""
        pass

    def add_goal(self, goal: Goal) -> None:
        """Add a goal to the agent."""
        self.goals.append(goal)

    def get_goals(self) -> List[Goal]:
        """Get all goals."""
        return self.goals


class BaseModel(ABC):
    """Abstract base class for ML models."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the model."""
        self.model_name = model_name
        self.config = config or {}
        self.model = None

    @abstractmethod
    async def load(self) -> None:
        """Load the model."""
        pass

    @abstractmethod
    async def predict(self, input_data: Any) -> Any:
        """Make predictions."""
        pass

    @abstractmethod
    async def train(self, training_data: Any) -> None:
        """Train the model."""
        pass

    @abstractmethod
    async def save(self, path: str) -> None:
        """Save the model."""
        pass
