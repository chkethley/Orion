"""Goal planning and task decomposition for AGI."""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """Represents a single task in a plan."""
    id: str
    description: str
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "estimated_duration": self.estimated_duration,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error
        }


@dataclass
class Plan:
    """Represents a complete plan with multiple tasks."""
    id: str
    goal: str
    tasks: List[Task] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"

    def add_task(self, task: Task) -> None:
        """Add a task to the plan."""
        self.tasks.append(task)

    def get_next_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute."""
        ready_tasks = []
        completed_ids = {
            task.id for task in self.tasks if task.status == TaskStatus.COMPLETED
        }

        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                if all(dep_id in completed_ids for dep_id in task.dependencies):
                    ready_tasks.append(task)

        # Sort by priority
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
        return ready_tasks

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return {
            "id": self.id,
            "goal": self.goal,
            "tasks": [task.to_dict() for task in self.tasks],
            "created_at": self.created_at.isoformat(),
            "status": self.status
        }


class Planner:
    """Goal-oriented planning system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize planner."""
        self.config = config or {}
        self.plans: Dict[str, Plan] = {}
        self.task_counter = 0

    async def create_plan(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Plan:
        """Create a plan to achieve a goal."""
        logger.info(f"Creating plan for goal: {goal}")

        plan = Plan(
            id=self._generate_plan_id(),
            goal=goal
        )

        # Decompose goal into tasks
        tasks = await self._decompose_goal(goal, context or {})

        for task_desc in tasks:
            task = Task(
                id=self._generate_task_id(),
                description=task_desc["description"],
                priority=task_desc.get("priority", 1),
                dependencies=task_desc.get("dependencies", []),
                estimated_duration=task_desc.get("estimated_duration")
            )
            plan.add_task(task)

        self.plans[plan.id] = plan
        return plan

    async def _decompose_goal(
        self,
        goal: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Decompose a goal into subtasks."""
        logger.info(f"Decomposing goal: {goal}")

        # This is a simplified decomposition
        # In a real system, this would use more sophisticated planning algorithms
        tasks = [
            {
                "description": f"Analyze requirements for: {goal}",
                "priority": 3,
                "dependencies": [],
                "estimated_duration": 1.0
            },
            {
                "description": f"Gather resources for: {goal}",
                "priority": 2,
                "dependencies": [],
                "estimated_duration": 2.0
            },
            {
                "description": f"Execute main task: {goal}",
                "priority": 1,
                "dependencies": [],
                "estimated_duration": 5.0
            },
            {
                "description": f"Verify completion of: {goal}",
                "priority": 1,
                "dependencies": [],
                "estimated_duration": 1.0
            }
        ]

        return tasks

    async def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """Execute a plan."""
        if plan_id not in self.plans:
            raise ValueError(f"Plan {plan_id} not found")

        plan = self.plans[plan_id]
        logger.info(f"Executing plan: {plan.id} for goal: {plan.goal}")

        results = []
        while True:
            next_tasks = plan.get_next_tasks()
            if not next_tasks:
                break

            for task in next_tasks:
                result = await self._execute_task(task)
                results.append(result)

        return {
            "plan_id": plan.id,
            "goal": plan.goal,
            "status": "completed",
            "results": results
        }

    async def _execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a single task."""
        logger.info(f"Executing task: {task.id} - {task.description}")

        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        try:
            # Simulate task execution
            # In a real system, this would call appropriate executors
            result = {
                "task_id": task.id,
                "description": task.description,
                "status": "completed",
                "output": f"Successfully completed: {task.description}"
            }

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result

            return result

        except Exception as e:
            logger.error(f"Task {task.id} failed: {str(e)}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            return {
                "task_id": task.id,
                "status": "failed",
                "error": str(e)
            }

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get a plan by ID."""
        return self.plans.get(plan_id)

    def list_plans(self) -> List[Dict[str, Any]]:
        """List all plans."""
        return [plan.to_dict() for plan in self.plans.values()]

    def _generate_plan_id(self) -> str:
        """Generate unique plan ID."""
        return f"plan_{datetime.now().timestamp()}"

    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        self.task_counter += 1
        return f"task_{self.task_counter}"
