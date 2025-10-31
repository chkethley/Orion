"""Multi-agent coordination and collaboration system."""
import torch
import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from src.agents.agi_agent import AGIAgent
from src.core.base import Goal, Message

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Roles agents can play in multi-agent systems."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"
    OBSERVER = "observer"


class CommunicationProtocol(Enum):
    """Communication protocols between agents."""
    BROADCAST = "broadcast"
    DIRECT = "direct"
    HIERARCHICAL = "hierarchical"
    AUCTION = "auction"
    NEGOTIATION = "negotiation"


@dataclass
class AgentMessage:
    """Message between agents."""
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    content: Any
    message_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Task to be executed by agents."""
    task_id: str
    description: str
    requirements: List[str]  # Required capabilities
    priority: int
    status: str = "pending"
    assigned_to: Optional[str] = None
    result: Optional[Any] = None


class MessageBus:
    """Central message bus for agent communication."""

    def __init__(self):
        """Initialize message bus."""
        self.messages: List[AgentMessage] = []
        self.subscriptions: Dict[str, Set[str]] = {}  # agent_id -> message_types

    def subscribe(self, agent_id: str, message_types: List[str]):
        """Subscribe agent to message types."""
        if agent_id not in self.subscriptions:
            self.subscriptions[agent_id] = set()
        self.subscriptions[agent_id].update(message_types)

    def publish(self, message: AgentMessage):
        """Publish message to the bus."""
        self.messages.append(message)
        logger.debug(f"Message published: {message.sender_id} -> {message.receiver_id}")

    def get_messages_for_agent(
        self,
        agent_id: str,
        since: Optional[datetime] = None
    ) -> List[AgentMessage]:
        """Get messages for specific agent."""
        messages = []

        for msg in self.messages:
            # Check if message is for this agent
            if msg.receiver_id == agent_id or msg.receiver_id is None:
                # Check subscription
                if agent_id in self.subscriptions:
                    if msg.message_type in self.subscriptions[agent_id]:
                        if since is None or msg.timestamp > since:
                            messages.append(msg)

        return messages


class MultiAgentCoordinator:
    """Coordinates multiple AGI agents for collaborative tasks."""

    def __init__(self, coordination_protocol: CommunicationProtocol = CommunicationProtocol.HIERARCHICAL):
        """Initialize multi-agent coordinator."""
        self.agents: Dict[str, AGIAgent] = {}
        self.agent_roles: Dict[str, AgentRole] = {}
        self.message_bus = MessageBus()
        self.protocol = coordination_protocol
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []

    def register_agent(
        self,
        agent: AGIAgent,
        role: AgentRole = AgentRole.WORKER
    ):
        """Register an agent with the coordinator."""
        self.agents[agent.agent_id] = agent
        self.agent_roles[agent.agent_id] = role

        # Subscribe to relevant messages
        if role == AgentRole.COORDINATOR:
            self.message_bus.subscribe(agent.agent_id, ["task_request", "task_result", "status_update"])
        elif role == AgentRole.WORKER:
            self.message_bus.subscribe(agent.agent_id, ["task_assignment", "coordination"])
        elif role == AgentRole.SPECIALIST:
            self.message_bus.subscribe(agent.agent_id, ["specialist_request"])

        logger.info(f"Registered agent {agent.agent_id} with role {role.value}")

    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.agent_roles[agent_id]
            logger.info(f"Unregistered agent {agent_id}")

    async def assign_task(self, task: Task):
        """Assign task to most suitable agent."""
        # Find agents with required capabilities
        suitable_agents = []

        for agent_id, agent in self.agents.items():
            # Check if agent has required capabilities
            if all(cap in agent.capabilities for cap in task.requirements):
                # Skip coordinators for regular tasks
                if self.agent_roles[agent_id] != AgentRole.COORDINATOR:
                    suitable_agents.append((agent_id, agent))

        if not suitable_agents:
            logger.warning(f"No suitable agents for task {task.task_id}")
            return

        # Simple assignment: choose first available
        # In practice, could use more sophisticated methods
        agent_id, agent = suitable_agents[0]

        task.assigned_to = agent_id
        task.status = "assigned"

        # Send task assignment message
        message = AgentMessage(
            sender_id="coordinator",
            receiver_id=agent_id,
            content=task,
            message_type="task_assignment"
        )
        self.message_bus.publish(message)

        logger.info(f"Assigned task {task.task_id} to agent {agent_id}")

    async def execute_collaborative_task(
        self,
        goal: str,
        environment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a task collaboratively with multiple agents."""
        logger.info(f"Starting collaborative task: {goal}")

        # Decompose goal into subtasks
        subtasks = self._decompose_goal(goal)

        # Assign subtasks to agents
        for subtask in subtasks:
            await self.assign_task(subtask)

        # Execute subtasks
        results = []

        for subtask in subtasks:
            if subtask.assigned_to:
                agent = self.agents[subtask.assigned_to]

                # Execute subtask
                agent_goal = Goal(
                    id=subtask.task_id,
                    description=subtask.description,
                    priority=subtask.priority,
                    status="active",
                    created_at=datetime.now()
                )
                agent.add_goal(agent_goal)
                agent.current_goal = agent_goal

                result = await agent.run_cycle(environment)
                subtask.result = result
                subtask.status = "completed"
                results.append(result)

                # Publish result
                message = AgentMessage(
                    sender_id=subtask.assigned_to,
                    receiver_id="coordinator",
                    content=result,
                    message_type="task_result",
                    metadata={"task_id": subtask.task_id}
                )
                self.message_bus.publish(message)

        # Aggregate results
        aggregated_result = self._aggregate_results(results)

        return {
            "goal": goal,
            "num_agents": len(self.agents),
            "num_subtasks": len(subtasks),
            "results": results,
            "aggregated_result": aggregated_result
        }

    def _decompose_goal(self, goal: str) -> List[Task]:
        """Decompose high-level goal into subtasks."""
        # Simplified decomposition
        # In practice, would use more sophisticated planning
        subtasks = [
            Task(
                task_id=f"task_1_{goal[:10]}",
                description=f"Analyze: {goal}",
                requirements=["reasoning"],
                priority=3
            ),
            Task(
                task_id=f"task_2_{goal[:10]}",
                description=f"Plan for: {goal}",
                requirements=["planning"],
                priority=2
            ),
            Task(
                task_id=f"task_3_{goal[:10]}",
                description=f"Execute: {goal}",
                requirements=["reasoning", "learning"],
                priority=1
            )
        ]

        return subtasks

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple agents."""
        # Simple aggregation
        return {
            "total_cycles": sum(r.get("cycle", 0) for r in results),
            "all_learning_complete": all(r.get("learning_complete", False) for r in results),
            "num_results": len(results)
        }

    async def consensus_decision(
        self,
        question: str,
        options: List[str]
    ) -> str:
        """Make consensus decision among agents."""
        logger.info(f"Seeking consensus on: {question}")

        votes: Dict[str, int] = {option: 0 for option in options}

        # Get vote from each agent
        for agent_id, agent in self.agents.items():
            # Skip observers
            if self.agent_roles[agent_id] == AgentRole.OBSERVER:
                continue

            # Simplified voting - in practice, agents would reason about options
            vote = options[hash(agent_id) % len(options)]
            votes[vote] += 1

            logger.debug(f"Agent {agent_id} voted for: {vote}")

        # Determine winner
        winner = max(votes.items(), key=lambda x: x[1])[0]

        logger.info(f"Consensus reached: {winner} with {votes[winner]} votes")

        return winner

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of multi-agent system."""
        return {
            "num_agents": len(self.agents),
            "agent_roles": {
                role.value: sum(1 for r in self.agent_roles.values() if r == role)
                for role in AgentRole
            },
            "active_tasks": len([t for t in self.task_queue if t.status == "assigned"]),
            "completed_tasks": len(self.completed_tasks),
            "protocol": self.protocol.value
        }


class SwarmIntelligence:
    """Swarm-based multi-agent coordination."""

    def __init__(self, num_agents: int, neighborhood_size: int = 3):
        """Initialize swarm."""
        self.num_agents = num_agents
        self.neighborhood_size = neighborhood_size
        self.agents: List[AGIAgent] = []
        self.positions: List[torch.Tensor] = []  # Agent positions in solution space
        self.velocities: List[torch.Tensor] = []
        self.best_positions: List[torch.Tensor] = []
        self.best_scores: List[float] = []
        self.global_best_position: Optional[torch.Tensor] = None
        self.global_best_score: float = float('-inf')

    def initialize(self, solution_dim: int):
        """Initialize swarm with random positions."""
        for i in range(self.num_agents):
            pos = torch.randn(solution_dim)
            vel = torch.randn(solution_dim) * 0.1

            self.positions.append(pos)
            self.velocities.append(vel)
            self.best_positions.append(pos.clone())
            self.best_scores.append(float('-inf'))

    def update(
        self,
        fitness_scores: List[float],
        inertia: float = 0.7,
        cognitive: float = 1.5,
        social: float = 1.5
    ):
        """Update swarm positions (PSO algorithm)."""
        for i in range(self.num_agents):
            # Update personal best
            if fitness_scores[i] > self.best_scores[i]:
                self.best_scores[i] = fitness_scores[i]
                self.best_positions[i] = self.positions[i].clone()

            # Update global best
            if fitness_scores[i] > self.global_best_score:
                self.global_best_score = fitness_scores[i]
                self.global_best_position = self.positions[i].clone()

            # Update velocity
            r1, r2 = torch.rand(2)

            cognitive_component = cognitive * r1 * (self.best_positions[i] - self.positions[i])
            social_component = social * r2 * (self.global_best_position - self.positions[i])

            self.velocities[i] = (
                inertia * self.velocities[i] +
                cognitive_component +
                social_component
            )

            # Update position
            self.positions[i] = self.positions[i] + self.velocities[i]


class AuctionBasedCoordination:
    """Auction-based task allocation for multi-agent systems."""

    def __init__(self):
        """Initialize auction coordinator."""
        self.pending_auctions: Dict[str, Dict[str, Any]] = {}
        self.bids: Dict[str, List[Tuple[str, float]]] = {}  # task_id -> [(agent_id, bid)]

    def create_auction(
        self,
        task: Task,
        deadline: Optional[datetime] = None
    ) -> str:
        """Create auction for a task."""
        auction_id = f"auction_{task.task_id}"

        self.pending_auctions[auction_id] = {
            "task": task,
            "deadline": deadline or datetime.now(),
            "status": "open"
        }
        self.bids[auction_id] = []

        logger.info(f"Created auction {auction_id} for task {task.task_id}")

        return auction_id

    def place_bid(
        self,
        auction_id: str,
        agent_id: str,
        bid_value: float
    ):
        """Place bid in auction."""
        if auction_id not in self.pending_auctions:
            raise ValueError(f"Auction {auction_id} not found")

        if self.pending_auctions[auction_id]["status"] != "open":
            raise ValueError(f"Auction {auction_id} is not open")

        self.bids[auction_id].append((agent_id, bid_value))
        logger.debug(f"Agent {agent_id} bid {bid_value} on {auction_id}")

    def resolve_auction(self, auction_id: str) -> str:
        """Resolve auction and assign task to winner."""
        if auction_id not in self.pending_auctions:
            raise ValueError(f"Auction {auction_id} not found")

        bids = self.bids[auction_id]

        if not bids:
            logger.warning(f"No bids for auction {auction_id}")
            return None

        # Winner: highest bid
        winner_id, winner_bid = max(bids, key=lambda x: x[1])

        self.pending_auctions[auction_id]["status"] = "closed"
        self.pending_auctions[auction_id]["winner"] = winner_id

        logger.info(f"Auction {auction_id} won by {winner_id} with bid {winner_bid}")

        return winner_id
