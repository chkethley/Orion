"""Data processing pipeline for AGI."""
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DataPoint:
    """Represents a single data point in the pipeline."""
    id: str
    data: Any
    metadata: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class PipelineStage:
    """Represents a single stage in a data processing pipeline."""

    def __init__(
        self,
        name: str,
        transform_fn: Callable[[Any], Any],
        validate_fn: Optional[Callable[[Any], bool]] = None
    ):
        """Initialize pipeline stage."""
        self.name = name
        self.transform_fn = transform_fn
        self.validate_fn = validate_fn
        self.processed_count = 0
        self.error_count = 0

    async def process(self, data: Any) -> Any:
        """Process data through this stage."""
        try:
            # Validate input if validation function provided
            if self.validate_fn and not self.validate_fn(data):
                raise ValueError(f"Validation failed in stage: {self.name}")

            # Transform data
            result = self.transform_fn(data)
            self.processed_count += 1

            logger.debug(f"Stage '{self.name}' processed data successfully")
            return result

        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in stage '{self.name}': {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get stage statistics."""
        return {
            "name": self.name,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "success_rate": (
                self.processed_count / (self.processed_count + self.error_count)
                if (self.processed_count + self.error_count) > 0
                else 0
            )
        }


class DataPipeline:
    """Multi-stage data processing pipeline."""

    def __init__(self, name: str, stages: Optional[List[PipelineStage]] = None):
        """Initialize data pipeline."""
        self.name = name
        self.stages = stages or []
        self.processed_items = []

    def add_stage(self, stage: PipelineStage) -> None:
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        logger.info(f"Added stage '{stage.name}' to pipeline '{self.name}'")

    async def process(self, data: Any) -> Any:
        """Process data through all pipeline stages."""
        logger.info(f"Processing data through pipeline '{self.name}'")

        result = data
        for stage in self.stages:
            result = await stage.process(result)

        # Store processed item
        data_point = DataPoint(
            id=f"dp_{len(self.processed_items)}",
            data=result,
            metadata={"pipeline": self.name},
            timestamp=datetime.now()
        )
        self.processed_items.append(data_point)

        return result

    async def batch_process(self, data_batch: List[Any]) -> List[Any]:
        """Process a batch of data."""
        logger.info(f"Batch processing {len(data_batch)} items")

        results = []
        for item in data_batch:
            try:
                result = await self.process(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item: {str(e)}")
                results.append(None)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "pipeline_name": self.name,
            "total_stages": len(self.stages),
            "processed_items": len(self.processed_items),
            "stage_stats": [stage.get_stats() for stage in self.stages]
        }


# Common transformation functions
def normalize_text(text: str) -> str:
    """Normalize text data."""
    return text.lower().strip()


def tokenize_text(text: str) -> List[str]:
    """Simple text tokenization."""
    return text.split()


def remove_duplicates(items: List[Any]) -> List[Any]:
    """Remove duplicates from list."""
    return list(set(items))


def filter_empty(items: List[Any]) -> List[Any]:
    """Filter out empty items."""
    return [item for item in items if item]


# Pre-built pipelines
def create_text_preprocessing_pipeline() -> DataPipeline:
    """Create a text preprocessing pipeline."""
    pipeline = DataPipeline(name="text_preprocessing")

    # Add stages
    pipeline.add_stage(
        PipelineStage(
            name="normalize",
            transform_fn=normalize_text
        )
    )
    pipeline.add_stage(
        PipelineStage(
            name="tokenize",
            transform_fn=tokenize_text
        )
    )
    pipeline.add_stage(
        PipelineStage(
            name="filter_empty",
            transform_fn=filter_empty
        )
    )

    return pipeline


def create_data_cleaning_pipeline() -> DataPipeline:
    """Create a data cleaning pipeline."""
    pipeline = DataPipeline(name="data_cleaning")

    # Add cleaning stages
    def remove_nulls(data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in data.items() if v is not None}

    def standardize_keys(data: Dict[str, Any]) -> Dict[str, Any]:
        return {k.lower().replace(" ", "_"): v for k, v in data.items()}

    pipeline.add_stage(
        PipelineStage(name="remove_nulls", transform_fn=remove_nulls)
    )
    pipeline.add_stage(
        PipelineStage(name="standardize_keys", transform_fn=standardize_keys)
    )

    return pipeline
