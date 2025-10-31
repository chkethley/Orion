"""Model serving API for PyTorch and ONNX models."""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from src.models.onnx_utils import ONNXInferenceEngine

logger = logging.getLogger(__name__)

# Global model registry
MODEL_REGISTRY: Dict[str, Any] = {}
ONNX_REGISTRY: Dict[str, ONNXInferenceEngine] = {}


class ModelLoadRequest(BaseModel):
    """Request to load a model."""
    model_name: str
    model_path: str
    model_type: str  # 'pytorch' or 'onnx'
    device: Optional[str] = 'cpu'


class InferenceRequest(BaseModel):
    """Request for model inference."""
    model_name: str
    inputs: List[List[float]]  # Batch of inputs
    parameters: Optional[Dict[str, Any]] = {}


class InferenceResponse(BaseModel):
    """Response from model inference."""
    model_name: str
    outputs: List[List[float]]
    inference_time_ms: float
    timestamp: str


class ModelInfo(BaseModel):
    """Model information."""
    model_name: str
    model_type: str
    loaded_at: str
    num_inferences: int
    avg_inference_time_ms: float


class ModelServing:
    """Model serving manager."""

    def __init__(self):
        """Initialize model serving."""
        self.stats: Dict[str, Dict[str, Any]] = {}

    def load_pytorch_model(
        self,
        model_name: str,
        model_path: str,
        model_class: Any,
        device: str = 'cpu'
    ):
        """Load PyTorch model."""
        try:
            # Instantiate model (assumes checkpoint contains architecture info)
            checkpoint = torch.load(model_path, map_location=device)

            if 'model_state_dict' in checkpoint:
                model = model_class()  # Need to know architecture
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model = checkpoint

            model.eval()
            model.to(device)

            MODEL_REGISTRY[model_name] = {
                'model': model,
                'device': device,
                'type': 'pytorch'
            }

            self.stats[model_name] = {
                'loaded_at': datetime.now().isoformat(),
                'num_inferences': 0,
                'total_inference_time': 0.0,
                'model_type': 'pytorch'
            }

            logger.info(f"Loaded PyTorch model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load PyTorch model {model_name}: {str(e)}")
            raise

    def load_onnx_model(
        self,
        model_name: str,
        model_path: str,
        providers: Optional[List[str]] = None
    ):
        """Load ONNX model."""
        try:
            if providers is None:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

            engine = ONNXInferenceEngine(model_path, providers=providers)

            ONNX_REGISTRY[model_name] = engine

            self.stats[model_name] = {
                'loaded_at': datetime.now().isoformat(),
                'num_inferences': 0,
                'total_inference_time': 0.0,
                'model_type': 'onnx'
            }

            logger.info(f"Loaded ONNX model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load ONNX model {model_name}: {str(e)}")
            raise

    def infer_pytorch(
        self,
        model_name: str,
        inputs: np.ndarray
    ) -> np.ndarray:
        """Run PyTorch inference."""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model {model_name} not found in registry")

        model_info = MODEL_REGISTRY[model_name]
        model = model_info['model']
        device = model_info['device']

        # Convert to tensor
        input_tensor = torch.from_numpy(inputs).float().to(device)

        # Inference
        import time
        start = time.time()

        with torch.no_grad():
            output = model(input_tensor)

        inference_time = (time.time() - start) * 1000  # ms

        # Update stats
        self.stats[model_name]['num_inferences'] += 1
        self.stats[model_name]['total_inference_time'] += inference_time

        # Convert to numpy
        if isinstance(output, tuple):
            output = output[0]

        return output.cpu().numpy()

    def infer_onnx(
        self,
        model_name: str,
        inputs: np.ndarray
    ) -> np.ndarray:
        """Run ONNX inference."""
        if model_name not in ONNX_REGISTRY:
            raise ValueError(f"ONNX model {model_name} not found in registry")

        engine = ONNX_REGISTRY[model_name]

        # Inference
        import time
        start = time.time()

        output = engine.infer(inputs.astype(np.float32))

        inference_time = (time.time() - start) * 1000  # ms

        # Update stats
        self.stats[model_name]['num_inferences'] += 1
        self.stats[model_name]['total_inference_time'] += inference_time

        return output

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information and statistics."""
        if model_name not in self.stats:
            raise ValueError(f"Model {model_name} not found")

        stats = self.stats[model_name]
        avg_time = 0.0
        if stats['num_inferences'] > 0:
            avg_time = stats['total_inference_time'] / stats['num_inferences']

        return {
            'model_name': model_name,
            'model_type': stats['model_type'],
            'loaded_at': stats['loaded_at'],
            'num_inferences': stats['num_inferences'],
            'avg_inference_time_ms': avg_time
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """List all loaded models."""
        return [self.get_model_info(name) for name in self.stats.keys()]

    def unload_model(self, model_name: str):
        """Unload model from memory."""
        if model_name in MODEL_REGISTRY:
            del MODEL_REGISTRY[model_name]
        if model_name in ONNX_REGISTRY:
            del ONNX_REGISTRY[model_name]
        if model_name in self.stats:
            del self.stats[model_name]

        logger.info(f"Unloaded model: {model_name}")


# Global serving instance
serving = ModelServing()

# FastAPI app
app = FastAPI(title="Orion Model Serving API", version="1.0.0")


@app.post("/models/load")
async def load_model(request: ModelLoadRequest):
    """Load a model into the serving system."""
    try:
        if request.model_type == 'pytorch':
            # Note: This is simplified - in production, need to handle architecture
            raise HTTPException(
                status_code=501,
                detail="PyTorch model loading requires architecture specification"
            )

        elif request.model_type == 'onnx':
            serving.load_onnx_model(
                model_name=request.model_name,
                model_path=request.model_path
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model type: {request.model_type}"
            )

        return {
            "status": "success",
            "message": f"Model {request.model_name} loaded successfully",
            "model_type": request.model_type
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """Run inference on a loaded model."""
    try:
        # Convert inputs to numpy
        inputs = np.array(request.inputs, dtype=np.float32)

        # Get model type
        if request.model_name not in serving.stats:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_name} not found"
            )

        model_type = serving.stats[request.model_name]['model_type']

        # Run inference
        import time
        start = time.time()

        if model_type == 'pytorch':
            outputs = serving.infer_pytorch(request.model_name, inputs)
        else:  # onnx
            outputs = serving.infer_onnx(request.model_name, inputs)

        inference_time = (time.time() - start) * 1000

        return InferenceResponse(
            model_name=request.model_name,
            outputs=outputs.tolist(),
            inference_time_ms=inference_time,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get information about a loaded model."""
    try:
        info = serving.get_model_info(model_name)
        return ModelInfo(**info)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/models")
async def list_models():
    """List all loaded models."""
    return {"models": serving.list_models()}


@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a model from memory."""
    try:
        serving.unload_model(model_name)
        return {
            "status": "success",
            "message": f"Model {model_name} unloaded successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_name}/benchmark")
async def benchmark_model(
    model_name: str,
    num_runs: int = 100,
    batch_size: int = 1,
    input_dim: int = 64
):
    """Benchmark model inference speed."""
    try:
        if model_name not in serving.stats:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not found"
            )

        # Generate dummy input
        inputs = np.random.randn(batch_size, input_dim).astype(np.float32)

        # Warmup
        model_type = serving.stats[model_name]['model_type']
        for _ in range(10):
            if model_type == 'pytorch':
                _ = serving.infer_pytorch(model_name, inputs)
            else:
                _ = serving.infer_onnx(model_name, inputs)

        # Benchmark
        import time
        times = []

        for _ in range(num_runs):
            start = time.time()

            if model_type == 'pytorch':
                _ = serving.infer_pytorch(model_name, inputs)
            else:
                _ = serving.infer_onnx(model_name, inputs)

            times.append((time.time() - start) * 1000)

        return {
            "model_name": model_name,
            "num_runs": num_runs,
            "batch_size": batch_size,
            "avg_inference_time_ms": np.mean(times),
            "std_inference_time_ms": np.std(times),
            "min_inference_time_ms": np.min(times),
            "max_inference_time_ms": np.max(times),
            "throughput_per_sec": 1000.0 / np.mean(times)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "loaded_models": len(serving.stats),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
