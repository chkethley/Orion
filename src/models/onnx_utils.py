"""ONNX model export and inference utilities."""
import torch
import torch.onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ONNXExporter:
    """Export PyTorch models to ONNX format."""

    def __init__(self, model: torch.nn.Module, model_name: str):
        """Initialize ONNX exporter."""
        self.model = model
        self.model_name = model_name
        self.model.eval()

    def export(
        self,
        dummy_input: Union[torch.Tensor, tuple],
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 14
    ) -> str:
        """Export model to ONNX format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        logger.info(f"Exporting model {self.model_name} to ONNX format")

        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )

            logger.info(f"Model exported successfully to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to export model: {str(e)}")
            raise

    def export_with_validation(
        self,
        dummy_input: Union[torch.Tensor, tuple],
        output_path: str,
        **kwargs
    ) -> str:
        """Export model and validate output."""
        # Export
        onnx_path = self.export(dummy_input, output_path, **kwargs)

        # Validate
        self.validate_export(onnx_path, dummy_input)

        return onnx_path

    def validate_export(
        self,
        onnx_path: str,
        test_input: Union[torch.Tensor, tuple],
        tolerance: float = 1e-5
    ) -> bool:
        """Validate ONNX export against PyTorch model."""
        logger.info("Validating ONNX export")

        # Get PyTorch output
        with torch.no_grad():
            if isinstance(test_input, tuple):
                pytorch_output = self.model(*test_input)
            else:
                pytorch_output = self.model(test_input)

        # Get ONNX output
        ort_session = ort.InferenceSession(onnx_path)

        if isinstance(test_input, tuple):
            onnx_inputs = {
                ort_session.get_inputs()[i].name: inp.cpu().numpy()
                for i, inp in enumerate(test_input)
            }
        else:
            onnx_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}

        onnx_outputs = ort_session.run(None, onnx_inputs)

        # Compare outputs
        if isinstance(pytorch_output, tuple):
            pytorch_output = pytorch_output[0]

        pytorch_np = pytorch_output.cpu().numpy()
        onnx_np = onnx_outputs[0]

        diff = np.abs(pytorch_np - onnx_np).max()

        if diff < tolerance:
            logger.info(f"Validation passed! Max difference: {diff}")
            return True
        else:
            logger.warning(f"Validation failed! Max difference: {diff}")
            return False


class ONNXInferenceEngine:
    """ONNX inference engine for fast model inference."""

    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None
    ):
        """Initialize ONNX inference engine."""
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Get input/output info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        logger.info(f"Loaded ONNX model from {model_path}")
        logger.info(f"Input names: {self.input_names}")
        logger.info(f"Output names: {self.output_names}")

    def infer(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray], torch.Tensor]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Run inference on inputs."""
        # Convert inputs to dict format
        if isinstance(inputs, torch.Tensor):
            inputs = {self.input_names[0]: inputs.cpu().numpy()}
        elif isinstance(inputs, np.ndarray):
            inputs = {self.input_names[0]: inputs}

        # Run inference
        outputs = self.session.run(self.output_names, inputs)

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def batch_infer(
        self,
        inputs: Union[List[np.ndarray], np.ndarray],
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """Run inference on batches."""
        if isinstance(inputs, np.ndarray):
            num_samples = inputs.shape[0]
            results = []

            for i in range(0, num_samples, batch_size):
                batch = inputs[i:i + batch_size]
                batch_result = self.infer(batch)
                results.append(batch_result)

            return np.concatenate(results, axis=0)
        else:
            results = []
            for inp in inputs:
                result = self.infer(inp)
                results.append(result)
            return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_path": self.model_path,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "inputs": [
                {
                    "name": inp.name,
                    "shape": inp.shape,
                    "type": inp.type
                }
                for inp in self.session.get_inputs()
            ],
            "outputs": [
                {
                    "name": out.name,
                    "shape": out.shape,
                    "type": out.type
                }
                for out in self.session.get_outputs()
            ]
        }


class ModelConverter:
    """Convert models between different formats."""

    @staticmethod
    def pytorch_to_onnx(
        model: torch.nn.Module,
        model_name: str,
        dummy_input: Union[torch.Tensor, tuple],
        output_dir: str = "data/models/onnx",
        **kwargs
    ) -> str:
        """Convert PyTorch model to ONNX."""
        output_path = Path(output_dir) / f"{model_name}.onnx"
        exporter = ONNXExporter(model, model_name)
        return exporter.export_with_validation(dummy_input, str(output_path), **kwargs)

    @staticmethod
    def quantize_onnx(
        model_path: str,
        output_path: Optional[str] = None,
        quantization_mode: str = "dynamic"
    ) -> str:
        """Quantize ONNX model for faster inference."""
        try:
            from onnxruntime.quantization import quantize_dynamic, quantize_static
            from onnxruntime.quantization import QuantType

            if output_path is None:
                output_path = model_path.replace(".onnx", "_quantized.onnx")

            if quantization_mode == "dynamic":
                quantize_dynamic(
                    model_path,
                    output_path,
                    weight_type=QuantType.QUInt8
                )
            else:
                raise ValueError(f"Unsupported quantization mode: {quantization_mode}")

            logger.info(f"Quantized model saved to {output_path}")
            return output_path

        except ImportError:
            logger.error("onnxruntime quantization not available")
            raise

    @staticmethod
    def optimize_onnx(model_path: str, output_path: Optional[str] = None) -> str:
        """Optimize ONNX model."""
        try:
            import onnx
            from onnx import optimizer

            if output_path is None:
                output_path = model_path.replace(".onnx", "_optimized.onnx")

            # Load model
            model = onnx.load(model_path)

            # Optimize
            optimized_model = optimizer.optimize(model)

            # Save
            onnx.save(optimized_model, output_path)

            logger.info(f"Optimized model saved to {output_path}")
            return output_path

        except ImportError:
            logger.error("onnx optimizer not available")
            raise


def export_common_models():
    """Export common model architectures to ONNX."""
    from src.models.neural_architectures import (
        TransformerEncoder,
        MemoryNetwork,
        ReasoningNetwork,
        PolicyNetwork,
        ValueNetwork
    )

    output_dir = Path("data/models/onnx")
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_export = [
        {
            "model": TransformerEncoder(vocab_size=10000, d_model=256, nhead=8, num_layers=3),
            "name": "transformer_encoder",
            "input": torch.randint(0, 10000, (1, 50)),
            "input_names": ["input_ids"],
            "dynamic_axes": {"input_ids": {0: "batch", 1: "sequence"}}
        },
        {
            "model": MemoryNetwork(input_dim=128, hidden_dim=256, memory_size=100),
            "name": "memory_network",
            "input": torch.randn(1, 10, 128),
            "input_names": ["input"],
            "dynamic_axes": {"input": {0: "batch", 1: "sequence"}}
        },
        {
            "model": ReasoningNetwork(input_dim=128, hidden_dim=256, output_dim=64),
            "name": "reasoning_network",
            "input": torch.randn(1, 128),
            "input_names": ["input"],
            "dynamic_axes": {"input": {0: "batch"}}
        },
        {
            "model": PolicyNetwork(observation_dim=64, action_dim=4),
            "name": "policy_network",
            "input": torch.randn(1, 64),
            "input_names": ["observation"],
            "dynamic_axes": {"observation": {0: "batch"}}
        },
        {
            "model": ValueNetwork(observation_dim=64),
            "name": "value_network",
            "input": torch.randn(1, 64),
            "input_names": ["observation"],
            "dynamic_axes": {"observation": {0: "batch"}}
        }
    ]

    exported_paths = []
    for model_config in models_to_export:
        model = model_config["model"]
        name = model_config["name"]
        dummy_input = model_config["input"]

        output_path = output_dir / f"{name}.onnx"

        exporter = ONNXExporter(model, name)
        path = exporter.export_with_validation(
            dummy_input,
            str(output_path),
            input_names=model_config.get("input_names"),
            dynamic_axes=model_config.get("dynamic_axes")
        )

        exported_paths.append(path)
        logger.info(f"Exported {name} to {path}")

    return exported_paths
