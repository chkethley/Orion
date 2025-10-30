"""Unit tests for reasoning engine."""
import pytest

from src.reasoning.reasoning_engine import (
    ReasoningEngine,
    ReasoningChain,
    ReasoningType
)


class TestReasoningChain:
    """Test ReasoningChain class."""

    def test_reasoning_chain_creation(self):
        """Test reasoning chain creation."""
        chain = ReasoningChain(goal="test goal")
        assert chain.goal == "test goal"
        assert len(chain.steps) == 0
        assert chain.conclusion is None

    def test_add_step(self):
        """Test adding steps to reasoning chain."""
        chain = ReasoningChain(goal="test")
        chain.add_step(
            step_type="deductive",
            premise="premise",
            inference="inference",
            confidence=0.8
        )

        assert len(chain.steps) == 1
        assert chain.steps[0]["confidence"] == 0.8

    def test_set_conclusion(self):
        """Test setting conclusion."""
        chain = ReasoningChain(goal="test")
        chain.set_conclusion("conclusion", 0.9)

        assert chain.conclusion == "conclusion"
        assert chain.confidence == 0.9

    def test_to_dict(self):
        """Test converting to dictionary."""
        chain = ReasoningChain(goal="test")
        chain.add_step("deductive", "p", "i", 0.8)
        chain.set_conclusion("c", 0.9)

        result = chain.to_dict()
        assert "goal" in result
        assert "steps" in result
        assert "conclusion" in result


class TestReasoningEngine:
    """Test ReasoningEngine class."""

    @pytest.mark.asyncio
    async def test_reasoning_engine_creation(self):
        """Test reasoning engine creation."""
        engine = ReasoningEngine()
        assert len(engine.reasoning_history) == 0

    @pytest.mark.asyncio
    async def test_deductive_reasoning(self):
        """Test deductive reasoning."""
        engine = ReasoningEngine()
        context = {
            "premises": ["All humans are mortal", "Socrates is human"],
            "facts": ["Socrates"]
        }

        chain = await engine.reason(
            goal="Determine if Socrates is mortal",
            context=context,
            reasoning_type=ReasoningType.DEDUCTIVE
        )

        assert chain.conclusion is not None
        assert len(chain.steps) > 0
        assert chain.confidence > 0

    @pytest.mark.asyncio
    async def test_inductive_reasoning(self):
        """Test inductive reasoning."""
        engine = ReasoningEngine()
        context = {
            "observations": ["Swan 1 is white", "Swan 2 is white", "Swan 3 is white"]
        }

        chain = await engine.reason(
            goal="All swans are white",
            context=context,
            reasoning_type=ReasoningType.INDUCTIVE
        )

        assert chain.conclusion is not None
        assert len(chain.steps) > 0

    @pytest.mark.asyncio
    async def test_reasoning_history(self):
        """Test reasoning history tracking."""
        engine = ReasoningEngine()
        context = {"premises": ["test"]}

        await engine.reason("goal1", context, ReasoningType.DEDUCTIVE)
        await engine.reason("goal2", context, ReasoningType.INDUCTIVE)

        history = engine.get_reasoning_history()
        assert len(history) == 2
