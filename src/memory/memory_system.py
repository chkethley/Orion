"""Advanced memory system for AGI."""
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
import numpy as np


class MemoryType:
    """Types of memory in the system."""
    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class Memory:
    """Represents a single memory."""

    def __init__(
        self,
        content: Any,
        memory_type: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        embedding: Optional[np.ndarray] = None
    ):
        """Initialize memory."""
        self.id = self._generate_id()
        self.content = content
        self.memory_type = memory_type
        self.importance = importance
        self.tags = tags or []
        self.embedding = embedding
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0

    def _generate_id(self) -> str:
        """Generate unique ID for memory."""
        return f"mem_{datetime.now().timestamp()}_{np.random.randint(10000)}"

    def access(self) -> None:
        """Mark memory as accessed."""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def get_age(self) -> float:
        """Get age of memory in hours."""
        return (datetime.now() - self.created_at).total_seconds() / 3600

    def calculate_relevance(self, current_time: Optional[datetime] = None) -> float:
        """Calculate relevance score based on recency, frequency, and importance."""
        if current_time is None:
            current_time = datetime.now()

        # Recency component (exponential decay)
        time_delta = (current_time - self.last_accessed).total_seconds() / 3600
        recency = np.exp(-0.1 * time_delta)

        # Frequency component
        frequency = np.log1p(self.access_count)

        # Combine components
        relevance = (0.4 * recency + 0.3 * frequency + 0.3 * self.importance)
        return relevance


class MemorySystem:
    """Advanced memory system with multiple memory types."""

    def __init__(
        self,
        short_term_capacity: int = 10,
        working_capacity: int = 7,
        consolidation_threshold: float = 0.7
    ):
        """Initialize memory system."""
        self.short_term_memory = deque(maxlen=short_term_capacity)
        self.working_memory = deque(maxlen=working_capacity)
        self.long_term_memory: Dict[str, Memory] = {}
        self.episodic_memory: List[Memory] = []
        self.semantic_memory: Dict[str, Any] = {}
        self.procedural_memory: Dict[str, Any] = {}

        self.consolidation_threshold = consolidation_threshold

    def store(
        self,
        content: Any,
        memory_type: str = MemoryType.SHORT_TERM,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        embedding: Optional[np.ndarray] = None
    ) -> Memory:
        """Store new memory."""
        memory = Memory(content, memory_type, importance, tags, embedding)

        if memory_type == MemoryType.SHORT_TERM:
            self.short_term_memory.append(memory)
        elif memory_type == MemoryType.WORKING:
            self.working_memory.append(memory)
        elif memory_type == MemoryType.LONG_TERM:
            self.long_term_memory[memory.id] = memory
        elif memory_type == MemoryType.EPISODIC:
            self.episodic_memory.append(memory)
        elif memory_type == MemoryType.SEMANTIC:
            self.semantic_memory[memory.id] = memory
        elif memory_type == MemoryType.PROCEDURAL:
            self.procedural_memory[memory.id] = memory

        return memory

    def retrieve(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 5,
        min_relevance: float = 0.3
    ) -> List[Memory]:
        """Retrieve memories based on query."""
        all_memories = []

        if memory_type is None or memory_type == MemoryType.SHORT_TERM:
            all_memories.extend(self.short_term_memory)
        if memory_type is None or memory_type == MemoryType.WORKING:
            all_memories.extend(self.working_memory)
        if memory_type is None or memory_type == MemoryType.LONG_TERM:
            all_memories.extend(self.long_term_memory.values())
        if memory_type is None or memory_type == MemoryType.EPISODIC:
            all_memories.extend(self.episodic_memory)

        # Calculate relevance and filter
        relevant_memories = []
        for memory in all_memories:
            relevance = memory.calculate_relevance()
            if relevance >= min_relevance:
                memory.access()
                relevant_memories.append((memory, relevance))

        # Sort by relevance and return top results
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in relevant_memories[:limit]]

    def consolidate(self) -> None:
        """Consolidate short-term memories to long-term storage."""
        for memory in list(self.short_term_memory):
            relevance = memory.calculate_relevance()
            if relevance >= self.consolidation_threshold:
                # Move to long-term memory
                memory.memory_type = MemoryType.LONG_TERM
                self.long_term_memory[memory.id] = memory

    def forget(self, threshold: float = 0.1) -> int:
        """Remove low-relevance memories from long-term storage."""
        to_remove = []
        for mem_id, memory in self.long_term_memory.items():
            if memory.calculate_relevance() < threshold:
                to_remove.append(mem_id)

        for mem_id in to_remove:
            del self.long_term_memory[mem_id]

        return len(to_remove)

    def get_context(self, limit: int = 5) -> List[Memory]:
        """Get current working context."""
        context = list(self.working_memory)
        if len(context) < limit:
            # Add relevant short-term memories
            context.extend(
                list(self.short_term_memory)[: limit - len(context)]
            )
        return context

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "short_term_count": len(self.short_term_memory),
            "working_count": len(self.working_memory),
            "long_term_count": len(self.long_term_memory),
            "episodic_count": len(self.episodic_memory),
            "semantic_count": len(self.semantic_memory),
            "procedural_count": len(self.procedural_memory),
            "total_memories": (
                len(self.short_term_memory)
                + len(self.working_memory)
                + len(self.long_term_memory)
                + len(self.episodic_memory)
                + len(self.semantic_memory)
                + len(self.procedural_memory)
            )
        }
