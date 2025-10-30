"""Unit tests for memory system."""
import pytest
from datetime import datetime
import numpy as np

from src.memory.memory_system import Memory, MemorySystem, MemoryType


class TestMemory:
    """Test Memory class."""

    def test_memory_creation(self):
        """Test memory creation."""
        memory = Memory(
            content="test content",
            memory_type=MemoryType.SHORT_TERM,
            importance=0.8
        )

        assert memory.content == "test content"
        assert memory.memory_type == MemoryType.SHORT_TERM
        assert memory.importance == 0.8
        assert memory.access_count == 0

    def test_memory_access(self):
        """Test memory access tracking."""
        memory = Memory("test", MemoryType.SHORT_TERM)
        initial_count = memory.access_count

        memory.access()
        assert memory.access_count == initial_count + 1

    def test_memory_age(self):
        """Test memory age calculation."""
        memory = Memory("test", MemoryType.SHORT_TERM)
        age = memory.get_age()
        assert age >= 0

    def test_memory_relevance(self):
        """Test relevance calculation."""
        memory = Memory(
            content="test",
            memory_type=MemoryType.SHORT_TERM,
            importance=0.8
        )
        relevance = memory.calculate_relevance()
        assert 0 <= relevance <= 1


class TestMemorySystem:
    """Test MemorySystem class."""

    def test_memory_system_creation(self):
        """Test memory system creation."""
        memory_system = MemorySystem()
        assert len(memory_system.short_term_memory) == 0
        assert len(memory_system.long_term_memory) == 0

    def test_store_short_term(self):
        """Test storing short-term memory."""
        memory_system = MemorySystem()
        memory = memory_system.store(
            content="test",
            memory_type=MemoryType.SHORT_TERM
        )

        assert len(memory_system.short_term_memory) == 1
        assert memory.content == "test"

    def test_store_long_term(self):
        """Test storing long-term memory."""
        memory_system = MemorySystem()
        memory = memory_system.store(
            content="test",
            memory_type=MemoryType.LONG_TERM
        )

        assert len(memory_system.long_term_memory) == 1
        assert memory.id in memory_system.long_term_memory

    def test_retrieve_memories(self):
        """Test memory retrieval."""
        memory_system = MemorySystem()

        # Store some memories
        for i in range(5):
            memory_system.store(
                content=f"test {i}",
                memory_type=MemoryType.SHORT_TERM,
                importance=0.5 + i * 0.1
            )

        # Retrieve memories
        retrieved = memory_system.retrieve(
            query="test",
            memory_type=MemoryType.SHORT_TERM,
            limit=3
        )

        assert len(retrieved) <= 3

    def test_consolidation(self):
        """Test memory consolidation."""
        memory_system = MemorySystem(consolidation_threshold=0.5)

        # Store high-importance memory
        memory = memory_system.store(
            content="important",
            memory_type=MemoryType.SHORT_TERM,
            importance=0.9
        )

        # Consolidate
        memory_system.consolidate()

        # Check if moved to long-term
        assert memory.id in memory_system.long_term_memory

    def test_forget(self):
        """Test forgetting low-relevance memories."""
        memory_system = MemorySystem()

        # Store low-importance memory
        memory_system.store(
            content="unimportant",
            memory_type=MemoryType.LONG_TERM,
            importance=0.05
        )

        initial_count = len(memory_system.long_term_memory)

        # Forget low-relevance memories
        forgotten_count = memory_system.forget(threshold=0.1)

        assert len(memory_system.long_term_memory) <= initial_count

    def test_get_statistics(self):
        """Test getting memory statistics."""
        memory_system = MemorySystem()

        memory_system.store("test1", MemoryType.SHORT_TERM)
        memory_system.store("test2", MemoryType.LONG_TERM)

        stats = memory_system.get_statistics()

        assert "short_term_count" in stats
        assert "long_term_count" in stats
        assert stats["total_memories"] >= 2
