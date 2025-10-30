"""Data preprocessing script."""
import argparse
import logging
import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.data_pipeline import (
    create_text_preprocessing_pipeline,
    create_data_cleaning_pipeline
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def preprocess_text_data(input_file: str, output_file: str) -> None:
    """Preprocess text data."""
    logger.info(f"Preprocessing text data from {input_file}")

    # Create pipeline
    pipeline = create_text_preprocessing_pipeline()

    # Read input data
    with open(input_file, 'r') as f:
        texts = f.readlines()

    # Process data
    results = await pipeline.batch_process(texts)

    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for result in results:
            if result:
                f.write(json.dumps(result) + '\n')

    # Print statistics
    stats = pipeline.get_statistics()
    logger.info(f"Pipeline statistics: {json.dumps(stats, indent=2)}")


async def clean_structured_data(input_file: str, output_file: str) -> None:
    """Clean structured data."""
    logger.info(f"Cleaning structured data from {input_file}")

    # Create pipeline
    pipeline = create_data_cleaning_pipeline()

    # Read input data
    with open(input_file, 'r') as f:
        data_items = [json.loads(line) for line in f]

    # Process data
    results = await pipeline.batch_process(data_items)

    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for result in results:
            if result:
                f.write(json.dumps(result) + '\n')

    # Print statistics
    stats = pipeline.get_statistics()
    logger.info(f"Pipeline statistics: {json.dumps(stats, indent=2)}")


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["text", "structured"],
        default="text",
        help="Type of data to preprocess"
    )

    args = parser.parse_args()

    import asyncio

    if args.type == "text":
        asyncio.run(preprocess_text_data(args.input, args.output))
    else:
        asyncio.run(clean_structured_data(args.input, args.output))


if __name__ == "__main__":
    main()
