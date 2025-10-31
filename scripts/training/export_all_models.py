"""Export all trained PyTorch models to ONNX format."""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.onnx_utils import export_common_models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Export all common models to ONNX."""
    logger.info("Starting batch export of models to ONNX")

    try:
        exported_paths = export_common_models()

        logger.info("\n" + "="*60)
        logger.info("EXPORT SUMMARY")
        logger.info("="*60)

        for path in exported_paths:
            logger.info(f"✓ {path}")

        logger.info(f"\nTotal models exported: {len(exported_paths)}")
        logger.info("All models exported successfully!")

    except Exception as e:
        logger.error(f"Error during export: {str(e)}")
        raise


if __name__ == "__main__":
    main()
