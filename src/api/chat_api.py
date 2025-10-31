"""Chat backend with AGI model integration."""
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, AsyncIterator
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
import uuid

from src.agents.agi_agent import AGIAgent
from src.core.base import Goal, Message
from src.memory.memory_system import MemoryType

logger = logging.getLogger(__name__)

app = FastAPI(title="Orion Chat API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class ChatMessage(BaseModel):
    """Chat message model."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str
    session_id: Optional[str] = None
    model: str = "agi_agent"
    parameters: Optional[Dict[str, Any]] = {}


class ChatResponse(BaseModel):
    """Chat response model."""
    message: str
    session_id: str
    model: str
    timestamp: str
    metadata: Dict[str, Any]


# Chat session management
class ChatSession:
    """Manages a chat session."""

    def __init__(self, session_id: str):
        """Initialize chat session."""
        self.session_id = session_id
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.now()
        self.agent: Optional[AGIAgent] = None
        self.current_model = "agi_agent"

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add message to session."""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        return message

    def get_context(self, limit: int = 10) -> List[ChatMessage]:
        """Get recent conversation context."""
        return self.messages[-limit:]

    def initialize_agent(self):
        """Initialize AGI agent for this session."""
        if self.agent is None:
            self.agent = AGIAgent(
                agent_id=f"chat_agent_{self.session_id}",
                name=f"Chat Agent {self.session_id[:8]}",
                capabilities=["reasoning", "learning", "planning"],
                config={
                    "short_term_capacity": 20,
                    "working_capacity": 10
                }
            )
            logger.info(f"Initialized agent for session {self.session_id}")


# Session storage
sessions: Dict[str, ChatSession] = {}


def get_or_create_session(session_id: Optional[str] = None) -> ChatSession:
    """Get existing session or create new one."""
    if session_id and session_id in sessions:
        return sessions[session_id]

    new_session_id = session_id or str(uuid.uuid4())
    session = ChatSession(new_session_id)
    sessions[new_session_id] = session

    logger.info(f"Created new chat session: {new_session_id}")
    return session


# Chat processors for different models
class ChatProcessor:
    """Base chat processor."""

    async def process(
        self,
        message: str,
        session: ChatSession,
        parameters: Dict[str, Any]
    ) -> str:
        """Process chat message and return response."""
        raise NotImplementedError


class AGIAgentProcessor(ChatProcessor):
    """AGI Agent processor."""

    async def process(
        self,
        message: str,
        session: ChatSession,
        parameters: Dict[str, Any]
    ) -> str:
        """Process message using AGI agent."""
        # Initialize agent if needed
        session.initialize_agent()
        agent = session.agent

        # Store user message in agent's memory
        agent.memory_system.store(
            content=message,
            memory_type=MemoryType.SHORT_TERM,
            importance=0.7,
            tags=["user_input", "chat"]
        )

        # Set goal based on message
        goal = Goal(
            id=f"chat_goal_{datetime.now().timestamp()}",
            description=f"Respond to: {message}",
            priority=1,
            status="active",
            created_at=datetime.now()
        )
        agent.current_goal = goal

        # Run agent cycle
        environment = {
            "user_message": message,
            "context": [msg.content for msg in session.get_context(5)],
            "mode": "chat"
        }

        result = await agent.run_cycle(environment)

        # Generate response based on agent's reasoning
        response = self._generate_response(result, message)

        return response

    def _generate_response(self, agent_result: Dict[str, Any], user_message: str) -> str:
        """Generate human-readable response from agent result."""
        reasoning = agent_result.get("reasoning", {})
        conclusion = reasoning.get("conclusion", "")

        # Create response
        response_parts = []

        if conclusion:
            response_parts.append(f"Based on my reasoning: {conclusion}")

        # Add some context from the action result
        action_result = agent_result.get("action_result", {})
        if action_result and action_result.get("status") == "completed":
            response_parts.append("I've processed your request successfully.")

        if not response_parts:
            response_parts.append("I've considered your message. How can I help you further?")

        return " ".join(response_parts)


class ReasoningProcessor(ChatProcessor):
    """Reasoning network processor."""

    async def process(
        self,
        message: str,
        session: ChatSession,
        parameters: Dict[str, Any]
    ) -> str:
        """Process message using reasoning network."""
        from src.models.neural_architectures import ReasoningNetwork
        import torch

        # Initialize model if needed
        if not hasattr(session, 'reasoning_model'):
            session.reasoning_model = ReasoningNetwork(
                input_dim=128,
                hidden_dim=256,
                output_dim=64
            )
            session.reasoning_model.eval()

        # Simple encoding (in practice, use proper text encoding)
        input_vec = torch.randn(1, 128)

        with torch.no_grad():
            output = session.reasoning_model(input_vec)

        return f"Reasoning network processed your input. Output dimensionality: {output.shape}"


class SimpleResponseProcessor(ChatProcessor):
    """Simple rule-based processor for testing."""

    async def process(
        self,
        message: str,
        session: ChatSession,
        parameters: Dict[str, Any]
    ) -> str:
        """Simple response generation."""
        message_lower = message.lower()

        # Simple pattern matching
        if any(greeting in message_lower for greeting in ["hello", "hi", "hey"]):
            return "Hello! I'm Orion, an AGI assistant. How can I help you today?"

        elif any(word in message_lower for word in ["help", "what can you do"]):
            return """I'm an advanced AGI system with multiple capabilities:

🧠 Reasoning - Multi-step logical reasoning
🎯 Planning - Goal decomposition and task planning
💾 Memory - Multi-tiered memory system
📚 Learning - Adaptive learning from interactions
🤖 Agents - Autonomous agent coordination

You can ask me questions, give me tasks, or just chat. What would you like to explore?"""

        elif "model" in message_lower or "architecture" in message_lower:
            return """I'm powered by multiple neural architectures:

- Transformer Encoders for language understanding
- Memory Networks for context retention
- Reasoning Networks for logical inference
- World Models for environment prediction
- Generative models (VAE, GAN, Diffusion)
- RL agents (DQN, SAC, PPO, etc.)

Which would you like to know more about?"""

        elif any(word in message_lower for word in ["think", "reason", "analyze"]):
            return f"Let me analyze that... I'm processing your request through my reasoning engine, considering multiple perspectives and applying logical inference. Based on my analysis of '{message}', I can help you explore this further."

        else:
            return f"I understand you said: '{message}'. Let me think about that and provide a thoughtful response. Could you provide more context or specify what you'd like me to do?"


# Model registry
PROCESSORS = {
    "agi_agent": AGIAgentProcessor(),
    "reasoning": ReasoningProcessor(),
    "simple": SimpleResponseProcessor()
}


# API Endpoints
@app.get("/")
async def root():
    """Serve chat interface."""
    html_path = Path(__file__).parent.parent.parent / "web" / "chat.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return {"message": "Chat API is running. Visit /docs for API documentation."}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message."""
    try:
        # Get or create session
        session = get_or_create_session(request.session_id)

        # Add user message
        session.add_message("user", request.message)

        # Get processor
        processor = PROCESSORS.get(request.model, PROCESSORS["simple"])

        # Process message
        response_text = await processor.process(
            request.message,
            session,
            request.parameters
        )

        # Add assistant message
        session.add_message("assistant", response_text, {
            "model": request.model
        })

        return ChatResponse(
            message=response_text,
            session_id=session.session_id,
            model=request.model,
            timestamp=datetime.now().isoformat(),
            metadata={
                "message_count": len(session.messages),
                "model_used": request.model
            }
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/history")
async def get_history(session_id: str, limit: int = 50):
    """Get chat history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    messages = session.messages[-limit:]

    return {
        "session_id": session_id,
        "messages": [msg.dict() for msg in messages],
        "total_messages": len(session.messages)
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "deleted", "session_id": session_id}

    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/models")
async def list_models():
    """List available chat models."""
    return {
        "models": [
            {
                "id": "agi_agent",
                "name": "AGI Agent",
                "description": "Full AGI agent with reasoning, planning, and learning",
                "capabilities": ["reasoning", "memory", "planning", "learning"]
            },
            {
                "id": "reasoning",
                "name": "Reasoning Network",
                "description": "Neural reasoning network for logical inference",
                "capabilities": ["reasoning", "inference"]
            },
            {
                "id": "simple",
                "name": "Simple Responder",
                "description": "Rule-based simple responses for testing",
                "capabilities": ["basic_chat"]
            }
        ]
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()

    session = get_or_create_session(session_id)

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)

            message = message_data.get("message", "")
            model = message_data.get("model", "simple")

            # Add user message
            session.add_message("user", message)

            # Send typing indicator
            await websocket.send_json({
                "type": "typing",
                "status": "processing"
            })

            # Process message
            processor = PROCESSORS.get(model, PROCESSORS["simple"])
            response_text = await processor.process(message, session, {})

            # Add assistant message
            session.add_message("assistant", response_text, {"model": model})

            # Send response
            await websocket.send_json({
                "type": "message",
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat(),
                "model": model
            })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "active_sessions": len(sessions),
        "available_models": list(PROCESSORS.keys())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
