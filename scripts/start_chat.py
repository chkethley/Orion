"""Start the Orion Chat interface."""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Start the chat server."""
    logger.info("="*60)
    logger.info("🚀 Starting Orion AGI Chat Interface")
    logger.info("="*60)
    logger.info("")
    logger.info("Chat interface will be available at:")
    logger.info("  👉 http://localhost:8002")
    logger.info("")
    logger.info("API documentation available at:")
    logger.info("  📚 http://localhost:8002/docs")
    logger.info("")
    logger.info("Available models:")
    logger.info("  🧠 AGI Agent - Full reasoning and learning")
    logger.info("  🎯 Reasoning Network - Neural reasoning")
    logger.info("  💬 Simple Responder - Rule-based chat")
    logger.info("")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("="*60)

    # Start server
    uvicorn.run(
        "src.api.chat_api:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
