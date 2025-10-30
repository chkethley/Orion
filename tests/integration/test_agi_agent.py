"""Integration tests for AGI agent."""
import pytest
from datetime import datetime

from src.agents.agi_agent import AGIAgent
from src.core.base import Goal


class TestAGIAgent:
    """Test AGIAgent class."""

    @pytest.fixture
    def agent(self):
        """Create test agent."""
        return AGIAgent(
            agent_id="test_agent_1",
            name="Test Agent",
            capabilities=["reasoning", "learning", "planning"],
            config={
                "short_term_capacity": 5,
                "working_capacity": 3
            }
        )

    def test_agent_creation(self, agent):
        """Test agent creation."""
        assert agent.agent_id == "test_agent_1"
        assert agent.name == "Test Agent"
        assert len(agent.capabilities) == 3
        assert agent.interaction_count == 0

    @pytest.mark.asyncio
    async def test_perceive(self, agent):
        """Test perception."""
        environment = {"temperature": 25, "objects": ["table", "chair"]}
        perception = await agent.perceive(environment)

        assert "environment_state" in perception
        assert perception["environment_state"] == environment

    @pytest.mark.asyncio
    async def test_reason(self, agent):
        """Test reasoning."""
        perception = {"environment_state": {"test": "data"}}
        reasoning = await agent.reason(perception)

        assert "reasoning_chain" in reasoning
        assert "conclusion" in reasoning

    @pytest.mark.asyncio
    async def test_plan(self, agent):
        """Test planning."""
        goal = Goal(
            id="goal_1",
            description="Test goal",
            priority=1,
            status="active",
            created_at=datetime.now()
        )
        agent.add_goal(goal)
        agent.current_goal = goal

        reasoning = {"conclusion": "Need to act"}
        plan = await agent.plan(reasoning)

        assert isinstance(plan, list)

    @pytest.mark.asyncio
    async def test_complete_cycle(self, agent):
        """Test complete agent cycle."""
        goal = Goal(
            id="goal_1",
            description="Analyze environment",
            priority=1,
            status="active",
            created_at=datetime.now()
        )
        agent.add_goal(goal)
        agent.current_goal = goal

        environment = {"test": "environment"}
        result = await agent.run_cycle(environment)

        assert "cycle" in result
        assert "perception" in result
        assert "reasoning" in result
        assert result["learning_complete"] is True

    def test_get_status(self, agent):
        """Test getting agent status."""
        status = agent.get_status()

        assert "agent_id" in status
        assert "name" in status
        assert "capabilities" in status
        assert "memory_statistics" in status
