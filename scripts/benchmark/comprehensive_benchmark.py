"""Comprehensive benchmarking suite for models and systems."""
import torch
import time
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import psutil
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.onnx_utils import ONNXInferenceEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result data."""
    model_name: str
    model_type: str
    batch_size: int
    input_shape: tuple
    avg_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_per_sec: float
    memory_used_mb: float
    gpu_used: bool
    num_runs: int


class ModelBenchmark:
    """Comprehensive model benchmarking."""

    def __init__(self, warmup_runs: int = 10, num_runs: int = 100):
        """Initialize benchmark."""
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs

    def benchmark_pytorch_model(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        batch_size: int = 1,
        device: str = "cpu"
    ) -> BenchmarkResult:
        """Benchmark PyTorch model."""
        logger.info(f"Benchmarking PyTorch model on {device}")

        model = model.to(device)
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(dummy_input)

        # Synchronize for GPU
        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        memory_before = self._get_memory_usage(device)

        with torch.no_grad():
            for _ in range(self.num_runs):
                start = time.time()
                _ = model(dummy_input)

                if device == "cuda":
                    torch.cuda.synchronize()

                latencies.append((time.time() - start) * 1000)

        memory_after = self._get_memory_usage(device)

        return BenchmarkResult(
            model_name=model.__class__.__name__,
            model_type="pytorch",
            batch_size=batch_size,
            input_shape=input_shape,
            avg_latency_ms=np.mean(latencies),
            std_latency_ms=np.std(latencies),
            min_latency_ms=np.min(latencies),
            max_latency_ms=np.max(latencies),
            throughput_per_sec=1000.0 / np.mean(latencies) * batch_size,
            memory_used_mb=memory_after - memory_before,
            gpu_used=(device == "cuda"),
            num_runs=self.num_runs
        )

    def benchmark_onnx_model(
        self,
        model_path: str,
        input_shape: tuple,
        batch_size: int = 1,
        providers: Optional[List[str]] = None
    ) -> BenchmarkResult:
        """Benchmark ONNX model."""
        logger.info(f"Benchmarking ONNX model: {model_path}")

        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        engine = ONNXInferenceEngine(model_path, providers=providers)

        # Create dummy input
        dummy_input = np.random.randn(batch_size, *input_shape).astype(np.float32)

        # Warmup
        for _ in range(self.warmup_runs):
            _ = engine.infer(dummy_input)

        # Benchmark
        latencies = []
        memory_before = self._get_memory_usage('cpu')

        for _ in range(self.num_runs):
            start = time.time()
            _ = engine.infer(dummy_input)
            latencies.append((time.time() - start) * 1000)

        memory_after = self._get_memory_usage('cpu')

        return BenchmarkResult(
            model_name=Path(model_path).stem,
            model_type="onnx",
            batch_size=batch_size,
            input_shape=input_shape,
            avg_latency_ms=np.mean(latencies),
            std_latency_ms=np.std(latencies),
            min_latency_ms=np.min(latencies),
            max_latency_ms=np.max(latencies),
            throughput_per_sec=1000.0 / np.mean(latencies) * batch_size,
            memory_used_mb=memory_after - memory_before,
            gpu_used=('CUDAExecutionProvider' in engine.session.get_providers()),
            num_runs=self.num_runs
        )

    def _get_memory_usage(self, device: str) -> float:
        """Get current memory usage in MB."""
        if device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            return psutil.Process().memory_info().rss / 1024 / 1024

    def compare_models(
        self,
        pytorch_model: Optional[torch.nn.Module] = None,
        onnx_model_path: Optional[str] = None,
        input_shape: tuple = (64,),
        batch_sizes: List[int] = [1, 8, 32]
    ) -> Dict[str, List[BenchmarkResult]]:
        """Compare PyTorch and ONNX models across batch sizes."""
        results = {"pytorch": [], "onnx": []}

        for batch_size in batch_sizes:
            if pytorch_model is not None:
                result = self.benchmark_pytorch_model(
                    pytorch_model, input_shape, batch_size
                )
                results["pytorch"].append(result)

            if onnx_model_path is not None:
                result = self.benchmark_onnx_model(
                    onnx_model_path, input_shape, batch_size
                )
                results["onnx"].append(result)

        return results


class SystemBenchmark:
    """Benchmark entire system components."""

    def __init__(self):
        """Initialize system benchmark."""
        pass

    def benchmark_memory_system(self) -> Dict[str, Any]:
        """Benchmark memory system performance."""
        from src.memory.memory_system import MemorySystem, MemoryType

        logger.info("Benchmarking memory system")

        memory = MemorySystem(short_term_capacity=100, working_capacity=10)

        # Benchmark storage
        num_memories = 1000
        start = time.time()

        for i in range(num_memories):
            memory.store(
                content=f"memory_{i}",
                memory_type=MemoryType.SHORT_TERM,
                importance=np.random.rand()
            )

        storage_time = time.time() - start

        # Benchmark retrieval
        start = time.time()

        for _ in range(100):
            _ = memory.retrieve(query="test", limit=10)

        retrieval_time = time.time() - start

        # Benchmark consolidation
        start = time.time()
        memory.consolidate()
        consolidation_time = time.time() - start

        return {
            "storage_time_ms": storage_time * 1000,
            "storage_throughput": num_memories / storage_time,
            "retrieval_time_ms": retrieval_time * 1000,
            "consolidation_time_ms": consolidation_time * 1000,
            "memory_stats": memory.get_statistics()
        }

    def benchmark_reasoning_engine(self) -> Dict[str, Any]:
        """Benchmark reasoning engine."""
        from src.reasoning.reasoning_engine import ReasoningEngine, ReasoningType

        logger.info("Benchmarking reasoning engine")

        engine = ReasoningEngine()

        results = {}

        for reasoning_type in ReasoningType:
            context = {
                "premises": ["test premise"],
                "observations": ["observation"],
                "facts": ["fact"]
            }

            start = time.time()

            import asyncio
            chain = asyncio.run(
                engine.reason(
                    goal="test goal",
                    context=context,
                    reasoning_type=reasoning_type
                )
            )

            elapsed = (time.time() - start) * 1000

            results[reasoning_type.value] = {
                "time_ms": elapsed,
                "num_steps": len(chain.steps),
                "confidence": chain.confidence
            }

        return results

    def benchmark_agent_cycle(self) -> Dict[str, Any]:
        """Benchmark complete agent cycle."""
        from src.agents.agi_agent import AGIAgent
        from src.core.base import Goal

        logger.info("Benchmarking agent cycle")

        agent = AGIAgent(
            agent_id="bench_agent",
            name="Benchmark Agent",
            capabilities=["reasoning", "learning", "planning"],
            config={}
        )

        goal = Goal(
            id="bench_goal",
            description="Benchmark goal",
            priority=1,
            status="active",
            created_at=time.time()
        )
        agent.add_goal(goal)
        agent.current_goal = goal

        environment = {"test": "environment"}

        # Run cycles
        num_cycles = 10
        cycle_times = []

        import asyncio

        for _ in range(num_cycles):
            start = time.time()
            _ = asyncio.run(agent.run_cycle(environment))
            cycle_times.append((time.time() - start) * 1000)

        return {
            "num_cycles": num_cycles,
            "avg_cycle_time_ms": np.mean(cycle_times),
            "total_time_ms": np.sum(cycle_times),
            "memory_stats": agent.memory_system.get_statistics()
        }


def generate_benchmark_report(
    results: Dict[str, Any],
    output_file: str = "benchmark_report.json"
):
    """Generate benchmark report."""
    logger.info(f"Generating benchmark report: {output_file}")

    # Convert results to JSON-serializable format
    json_results = {}

    for key, value in results.items():
        if isinstance(value, list) and value and isinstance(value[0], BenchmarkResult):
            json_results[key] = [asdict(r) for r in value]
        else:
            json_results[key] = value

    # Write report
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    logger.info(f"Report saved to {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    for key, value in json_results.items():
        print(f"\n{key.upper()}:")
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and 'model_name' in item:
                    print(f"  {item['model_name']}:")
                    print(f"    Batch: {item['batch_size']}, "
                          f"Latency: {item['avg_latency_ms']:.2f}ms, "
                          f"Throughput: {item['throughput_per_sec']:.2f}/s")
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, dict):
                    print(f"  {k}: {json.dumps(v, indent=4)}")
                else:
                    print(f"  {k}: {v}")


def main():
    """Run comprehensive benchmark suite."""
    logger.info("Starting comprehensive benchmark suite")

    results = {}

    # Model benchmarks
    model_benchmark = ModelBenchmark(warmup_runs=10, num_runs=100)

    # Benchmark a simple model
    from src.models.neural_architectures import ReasoningNetwork

    model = ReasoningNetwork(input_dim=64, hidden_dim=128, output_dim=32)

    pytorch_results = []
    for batch_size in [1, 8, 32]:
        result = model_benchmark.benchmark_pytorch_model(
            model=model,
            input_shape=(64,),
            batch_size=batch_size,
            device="cpu"
        )
        pytorch_results.append(result)

    results["pytorch_models"] = pytorch_results

    # System benchmarks
    system_benchmark = SystemBenchmark()

    results["memory_system"] = system_benchmark.benchmark_memory_system()
    results["reasoning_engine"] = system_benchmark.benchmark_reasoning_engine()
    results["agent_cycle"] = system_benchmark.benchmark_agent_cycle()

    # Generate report
    generate_benchmark_report(results, "comprehensive_benchmark_report.json")

    logger.info("Benchmark suite completed!")


if __name__ == "__main__":
    main()
