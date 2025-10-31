"""ONNX model inference script."""
import argparse
import numpy as np
import sys
import os
from pathlib import Path
import time
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.onnx_utils import ONNXInferenceEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_inference(
    engine: ONNXInferenceEngine,
    input_data: np.ndarray,
    num_runs: int = 100
):
    """Benchmark inference speed."""
    logger.info(f"Warming up with 10 runs...")
    for _ in range(10):
        _ = engine.infer(input_data)

    logger.info(f"Benchmarking with {num_runs} runs...")
    start_time = time.time()
    for _ in range(num_runs):
        _ = engine.infer(input_data)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_runs

    logger.info(f"Total time: {total_time:.4f}s")
    logger.info(f"Average time per inference: {avg_time*1000:.2f}ms")
    logger.info(f"Throughput: {1/avg_time:.2f} inferences/second")

    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'throughput': 1/avg_time
    }


def infer_transformer(model_path: str, args):
    """Inference with transformer model."""
    logger.info("Running transformer inference")

    engine = ONNXInferenceEngine(model_path)

    # Create dummy input
    batch_size = args.batch_size
    seq_length = args.seq_length
    input_data = np.random.randint(0, args.vocab_size, (batch_size, seq_length)).astype(np.int64)

    # Run inference
    output = engine.infer(input_data)
    logger.info(f"Input shape: {input_data.shape}")
    logger.info(f"Output shape: {output.shape}")

    # Benchmark
    if args.benchmark:
        single_input = input_data[:1]
        benchmark_inference(engine, single_input, args.num_runs)

    return output


def infer_memory_network(model_path: str, args):
    """Inference with memory network."""
    logger.info("Running memory network inference")

    engine = ONNXInferenceEngine(model_path)

    # Create dummy input
    batch_size = args.batch_size
    seq_length = args.seq_length
    input_dim = args.input_dim
    input_data = np.random.randn(batch_size, seq_length, input_dim).astype(np.float32)

    # Run inference
    output = engine.infer(input_data)
    logger.info(f"Input shape: {input_data.shape}")
    logger.info(f"Output shape: {output.shape}")

    # Benchmark
    if args.benchmark:
        single_input = input_data[:1]
        benchmark_inference(engine, single_input, args.num_runs)

    return output


def infer_reasoning_network(model_path: str, args):
    """Inference with reasoning network."""
    logger.info("Running reasoning network inference")

    engine = ONNXInferenceEngine(model_path)

    # Create dummy input
    batch_size = args.batch_size
    input_dim = args.input_dim
    input_data = np.random.randn(batch_size, input_dim).astype(np.float32)

    # Run inference
    output = engine.infer(input_data)
    logger.info(f"Input shape: {input_data.shape}")
    logger.info(f"Output shape: {output.shape}")

    # Benchmark
    if args.benchmark:
        single_input = input_data[:1]
        benchmark_inference(engine, single_input, args.num_runs)

    return output


def infer_policy_network(model_path: str, args):
    """Inference with policy network."""
    logger.info("Running policy network inference")

    engine = ONNXInferenceEngine(model_path)

    # Create dummy observation
    batch_size = args.batch_size
    obs_dim = args.observation_dim
    observation = np.random.randn(batch_size, obs_dim).astype(np.float32)

    # Run inference
    action_logits = engine.infer(observation)
    logger.info(f"Observation shape: {observation.shape}")
    logger.info(f"Action logits shape: {action_logits.shape}")

    # Get action probabilities
    action_probs = np.exp(action_logits) / np.sum(np.exp(action_logits), axis=-1, keepdims=True)
    logger.info(f"Action probabilities:\n{action_probs}")

    # Benchmark
    if args.benchmark:
        single_obs = observation[:1]
        benchmark_inference(engine, single_obs, args.num_runs)

    return action_logits


def infer_value_network(model_path: str, args):
    """Inference with value network."""
    logger.info("Running value network inference")

    engine = ONNXInferenceEngine(model_path)

    # Create dummy observation
    batch_size = args.batch_size
    obs_dim = args.observation_dim
    observation = np.random.randn(batch_size, obs_dim).astype(np.float32)

    # Run inference
    values = engine.infer(observation)
    logger.info(f"Observation shape: {observation.shape}")
    logger.info(f"State values shape: {values.shape}")
    logger.info(f"State values:\n{values}")

    # Benchmark
    if args.benchmark:
        single_obs = observation[:1]
        benchmark_inference(engine, single_obs, args.num_runs)

    return values


def batch_inference_example(model_path: str, args):
    """Example of batch inference."""
    logger.info("Running batch inference example")

    engine = ONNXInferenceEngine(model_path)

    # Create large batch
    num_samples = args.num_samples
    input_dim = args.input_dim
    data = np.random.randn(num_samples, input_dim).astype(np.float32)

    logger.info(f"Processing {num_samples} samples in batches of {args.batch_size}")

    start_time = time.time()
    results = engine.batch_infer(data, batch_size=args.batch_size)
    end_time = time.time()

    logger.info(f"Batch inference completed in {end_time - start_time:.4f}s")
    logger.info(f"Results shape: {results.shape}")

    return results


def compare_models(model_paths: list, args):
    """Compare multiple ONNX models."""
    logger.info("Comparing multiple models")

    results = {}

    for model_path in model_paths:
        model_name = Path(model_path).stem
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing model: {model_name}")
        logger.info(f"{'='*60}")

        try:
            engine = ONNXInferenceEngine(model_path)

            # Get model info
            info = engine.get_model_info()
            logger.info(f"Model info:\n{json.dumps(info, indent=2)}")

            # Create appropriate input based on model
            input_shape = info['inputs'][0]['shape']
            # Replace dynamic dims with fixed values
            input_shape = [args.batch_size if d == 'batch' else
                          args.seq_length if d == 'sequence' else
                          d for d in input_shape]

            if 'int' in info['inputs'][0]['type']:
                input_data = np.random.randint(0, 1000, input_shape).astype(np.int64)
            else:
                input_data = np.random.randn(*input_shape).astype(np.float32)

            # Benchmark
            single_input = input_data[:1] if len(input_shape) > 1 else input_data
            benchmark_results = benchmark_inference(engine, single_input, num_runs=100)

            results[model_name] = {
                'model_info': info,
                'benchmark': benchmark_results
            }

        except Exception as e:
            logger.error(f"Error testing {model_name}: {str(e)}")
            results[model_name] = {'error': str(e)}

    # Print comparison
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON RESULTS")
    logger.info(f"{'='*60}")

    for model_name, result in results.items():
        if 'error' not in result:
            logger.info(f"\n{model_name}:")
            logger.info(f"  Avg inference time: {result['benchmark']['avg_time']*1000:.2f}ms")
            logger.info(f"  Throughput: {result['benchmark']['throughput']:.2f} inf/s")

    return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="ONNX model inference")
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to ONNX model")
    parser.add_argument("--model-type", type=str,
                      choices=["transformer", "memory", "reasoning", "policy", "value", "auto"],
                      default="auto", help="Model type")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--compare-models", nargs='+', help="Compare multiple models")

    # Model-specific args
    parser.add_argument("--seq-length", type=int, default=50)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--observation-dim", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=1000)

    args = parser.parse_args()

    if args.compare_models:
        results = compare_models(args.compare_models, args)
        # Save results
        output_file = "benchmark_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
    else:
        # Single model inference
        model_type = args.model_type

        if model_type == "auto":
            # Auto-detect from filename
            model_name = Path(args.model_path).stem.lower()
            if "transformer" in model_name:
                model_type = "transformer"
            elif "memory" in model_name:
                model_type = "memory"
            elif "reasoning" in model_name:
                model_type = "reasoning"
            elif "policy" in model_name:
                model_type = "policy"
            elif "value" in model_name:
                model_type = "value"
            else:
                logger.error("Could not auto-detect model type. Please specify --model-type")
                return

        logger.info(f"Detected model type: {model_type}")

        # Run appropriate inference
        if model_type == "transformer":
            output = infer_transformer(args.model_path, args)
        elif model_type == "memory":
            output = infer_memory_network(args.model_path, args)
        elif model_type == "reasoning":
            output = infer_reasoning_network(args.model_path, args)
        elif model_type == "policy":
            output = infer_policy_network(args.model_path, args)
        elif model_type == "value":
            output = infer_value_network(args.model_path, args)

        logger.info("Inference completed successfully!")


if __name__ == "__main__":
    main()
