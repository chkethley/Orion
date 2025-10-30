"""Main AGI agent implementation combining all components."""
from typing import Any, Dict, List, Optional
import logging

from src.core.base import BaseAgent, Goal
from src.memory.memory_system import MemorySystem, MemoryType
from src.reasoning.reasoning_engine import ReasoningEngine, ReasoningType
from src.planning.planner import Planner
from src.learning.learner import Learner


logger = logging.getLogger(__name__)


class AGIAgent(BaseAgent):
    """Advanced General Intelligence Agent."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        capabilities: List[str],
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize AGI agent."""
        super().__init__(agent_id, name, capabilities, config)

        # Initialize core components
        self.memory_system = MemorySystem(
            short_term_capacity=config.get("short_term_capacity", 10),
            working_capacity=config.get("working_capacity", 7)
        )
        self.reasoning_engine = ReasoningEngine(config.get("reasoning", {}))
        self.planner = Planner(config.get("planning", {}))
        self.learner = Learner(config.get("learning", {}))

        # Agent state
        self.current_goal: Optional[Goal] = None
        self.current_plan: Optional[Any] = None
        self.interaction_count = 0

    async def perceive(self, environment: Any) -> Dict[str, Any]:
        """Perceive and process environmental inputs."""
        logger.info(f"Agent {self.name} perceiving environment")

        perception = {
            "timestamp": "now",
            "environment_state": environment,
            "agent_state": self._get_internal_state()
        }

        # Store perception in memory
        self.memory_system.store(
            content=perception,
            memory_type=MemoryType.SHORT_TERM,
            importance=0.6
        )

        return perception

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Apply reasoning to perception."""
        logger.info(f"Agent {self.name} reasoning about perception")

        # Get relevant context from memory
        context_memories = self.memory_system.get_context(limit=5)
        context = {
            "perception": perception,
            "memories": [mem.content for mem in context_memories],
            "current_goal": self.current_goal.description if self.current_goal else None
        }

        # Apply reasoning
        reasoning_chain = await self.reasoning_engine.reason(
            goal=self.current_goal.description if self.current_goal else "general_understanding",
            context=context,
            reasoning_type=ReasoningType.DEDUCTIVE
        )

        reasoning_result = {
            "reasoning_chain": reasoning_chain.to_dict(),
            "conclusion": reasoning_chain.conclusion,
            "confidence": reasoning_chain.confidence
        }

        # Store reasoning in memory
        self.memory_system.store(
            content=reasoning_result,
            memory_type=MemoryType.WORKING,
            importance=0.7
        )

        return reasoning_result

    async def plan(self, reasoning: Dict[str, Any]) -> List[Any]:
        """Create an action plan based on reasoning."""
        logger.info(f"Agent {self.name} creating plan")

        if not self.current_goal:
            logger.warning("No current goal set, cannot create plan")
            return []

        # Create plan using planner
        plan = await self.planner.create_plan(
            goal=self.current_goal.description,
            context=reasoning
        )

        self.current_plan = plan
        return plan.tasks

    async def act(self, plan: List[Any]) -> Any:
        """Execute the planned actions."""
        logger.info(f"Agent {self.name} executing plan")

        if not self.current_plan:
            logger.warning("No plan to execute")
            return None

        # Execute plan
        result = await self.planner.execute_plan(self.current_plan.id)

        # Store action results in episodic memory
        self.memory_system.store(
            content=result,
            memory_type=MemoryType.EPISODIC,
            importance=0.8
        )

        return result

    async def learn(self, experience: Dict[str, Any]) -> None:
        """Learn from experience."""
        logger.info(f"Agent {self.name} learning from experience")

        # Learn using reinforcement learning by default
        learning_result = await self.learner.learn(
            experience=experience,
            strategy="reinforcement"
        )

        # Store learning outcomes
        self.memory_system.store(
            content=learning_result,
            memory_type=MemoryType.LONG_TERM,
            importance=0.9
        )

        logger.info(f"Learning result: {learning_result}")

    async def run_cycle(self, environment: Any) -> Dict[str, Any]:
        """Execute one complete perception-reasoning-planning-action-learning cycle."""
        logger.info(f"Agent {self.name} starting execution cycle {self.interaction_count}")

        # 1. Perceive
        perception = await self.perceive(environment)

        # 2. Reason
        reasoning = await self.reason(perception)

        # 3. Plan
        plan = await self.plan(reasoning)

        # 4. Act
        action_result = await self.act(plan)

        # 5. Learn
        experience = {
            "perception": perception,
            "reasoning": reasoning,
            "plan": plan,
            "action_result": action_result,
            "reward": self._calculate_reward(action_result)
        }
        await self.learn(experience)

        # Consolidate memories
        self.memory_system.consolidate()

        self.interaction_count += 1

        return {
            "cycle": self.interaction_count,
            "perception": perception,
            "reasoning": reasoning,
            "action_result": action_result,
            "learning_complete": True
        }

    def _get_internal_state(self) -> Dict[str, Any]:
        """Get current internal state of the agent."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "capabilities": self.capabilities,
            "current_goal": self.current_goal.to_dict() if self.current_goal else None,
            "memory_stats": self.memory_system.get_statistics(),
            "interaction_count": self.interaction_count
        }

    def _calculate_reward(self, action_result: Any) -> float:
        """Calculate reward from action result."""
        # Simple reward calculation
        if action_result and isinstance(action_result, dict):
            if action_result.get("status") == "completed":
                return 1.0
            elif action_result.get("status") == "failed":
                return -0.5
        return 0.0

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "capabilities": self.capabilities,
            "current_goal": self.current_goal.to_dict() if self.current_goal else None,
            "interaction_count": self.interaction_count,
            "memory_statistics": self.memory_system.get_statistics(),
            "learning_evaluation": self.learner.evaluate_learning(),
            "active_plans": len(self.planner.plans)
        }
