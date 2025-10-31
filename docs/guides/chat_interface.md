

# Orion Chat Interface Guide

The Orion Chat Interface provides an interactive web-based UI for communicating with all the AGI models and systems.

## Quick Start

### Start the Chat Server

```bash
# Method 1: Using the start script
python scripts/start_chat.py

# Method 2: Direct uvicorn
uvicorn src.api.chat_api:app --host 0.0.0.0 --port 8002 --reload
```

The chat interface will be available at: **http://localhost:8002**

## Features

### 🧠 Multiple AI Models

The chat interface supports switching between different AI models:

1. **AGI Agent** - Full AGI system with:
   - Multi-step reasoning
   - Memory retention across conversations
   - Goal-oriented planning
   - Continuous learning from interactions

2. **Reasoning Network** - Neural reasoning for:
   - Logical inference
   - Pattern recognition
   - Problem solving

3. **Simple Responder** - Rule-based system for:
   - Quick responses
   - Pattern matching
   - Testing and debugging

4. **Memory-Augmented** - Memory network with:
   - Explicit context tracking
   - Attention mechanisms
   - Long-term conversation memory

5. **Multi-Modal** - Multi-modal understanding:
   - Text processing
   - Vision integration (when available)
   - Audio processing (when available)

6. **Generative** - Creative responses using:
   - Variational Autoencoders
   - Latent space exploration
   - Novel idea generation

7. **World Model** - Predictive responses:
   - Outcome prediction
   - Action consequence modeling
   - Environment understanding

8. **Ensemble** - Combines multiple models:
   - Synthesizes different perspectives
   - Robust responses
   - Best-of-all-worlds approach

### 💬 Chat Features

- **Real-time messaging** with instant responses
- **Session management** with persistent chat history
- **Model switching** on-the-fly
- **WebSocket support** for real-time streaming
- **Message formatting** with basic markdown
- **Typing indicators** for better UX
- **Auto-scroll** to latest messages
- **Conversation context** maintained across messages

### 🎨 User Interface

- **Modern gradient design** with smooth animations
- **Responsive layout** works on desktop and mobile
- **Message bubbles** with timestamps
- **Avatar indicators** for user/assistant
- **Status indicators** showing connection state
- **Model badge** showing active model
- **Suggestion chips** for quick starts

## API Endpoints

### REST API

#### POST `/chat`

Send a chat message and get a response.

**Request:**
```json
{
  "message": "Hello, how are you?",
  "session_id": "optional-session-id",
  "model": "agi_agent",
  "parameters": {}
}
```

**Response:**
```json
{
  "message": "I'm doing well! How can I help you?",
  "session_id": "session_12345",
  "model": "agi_agent",
  "timestamp": "2024-01-01T12:00:00",
  "metadata": {
    "message_count": 2,
    "model_used": "agi_agent"
  }
}
```

#### GET `/sessions/{session_id}/history`

Get chat history for a session.

**Parameters:**
- `limit`: Maximum number of messages (default: 50)

**Response:**
```json
{
  "session_id": "session_12345",
  "messages": [
    {
      "role": "user",
      "content": "Hello",
      "timestamp": "2024-01-01T12:00:00",
      "metadata": {}
    },
    {
      "role": "assistant",
      "content": "Hi there!",
      "timestamp": "2024-01-01T12:00:01",
      "metadata": {"model": "agi_agent"}
    }
  ],
  "total_messages": 2
}
```

#### DELETE `/sessions/{session_id}`

Delete a chat session.

#### GET `/models`

List all available chat models.

#### GET `/health`

Health check endpoint.

### WebSocket API

#### WS `/ws/{session_id}`

Real-time WebSocket connection for streaming chat.

**Send:**
```json
{
  "message": "Your message here",
  "model": "agi_agent"
}
```

**Receive:**
```json
{
  "type": "message",
  "role": "assistant",
  "content": "Response here",
  "timestamp": "2024-01-01T12:00:00",
  "model": "agi_agent"
}
```

## Usage Examples

### Python Client

```python
import requests

API_BASE = "http://localhost:8002"

# Send a message
response = requests.post(f"{API_BASE}/chat", json={
    "message": "What are your capabilities?",
    "model": "agi_agent"
})

data = response.json()
print(f"Assistant: {data['message']}")
print(f"Session ID: {data['session_id']}")
```

### JavaScript Client

```javascript
// Send message
async function sendMessage(message, model = 'agi_agent') {
    const response = await fetch('http://localhost:8002/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            message: message,
            model: model
        })
    });

    const data = await response.json();
    console.log('Response:', data.message);
    return data;
}

// Use it
sendMessage('Hello, Orion!');
```

### WebSocket Client

```javascript
const ws = new WebSocket('ws://localhost:8002/ws/my_session_id');

ws.onopen = () => {
    ws.send(JSON.stringify({
        message: 'Hello via WebSocket!',
        model: 'agi_agent'
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'message') {
        console.log('Assistant:', data.content);
    }
};
```

## Model Selection Guide

### When to Use Each Model

**AGI Agent** - Use for:
- Complex reasoning tasks
- Multi-step problem solving
- Tasks requiring memory
- Learning from feedback

**Reasoning Network** - Use for:
- Logical inference
- Pattern recognition
- Quick reasoning tasks

**Memory-Augmented** - Use for:
- Long conversations
- Context-dependent tasks
- Referencing previous topics

**Multi-Modal** - Use for:
- Understanding multiple input types
- Rich media processing

**Generative** - Use for:
- Creative tasks
- Novel ideas
- Brainstorming

**World Model** - Use for:
- Predictions
- Planning scenarios
- Understanding consequences

**Ensemble** - Use for:
- Important decisions
- Multiple perspectives
- Robust answers

## Configuration

### Environment Variables

```bash
# Chat API settings
CHAT_API_HOST=0.0.0.0
CHAT_API_PORT=8002
CHAT_API_RELOAD=true

# Session settings
SESSION_TIMEOUT=3600
MAX_SESSIONS=1000

# Model settings
DEFAULT_CHAT_MODEL=agi_agent
ENABLE_AGI_AGENT=true
```

### Custom Processors

You can add custom chat processors:

```python
from src.api.chat_api import ChatProcessor, ChatSession, PROCESSORS

class MyCustomProcessor(ChatProcessor):
    async def process(
        self,
        message: str,
        session: ChatSession,
        parameters: dict
    ) -> str:
        # Your custom logic here
        return f"Processed: {message}"

# Register it
PROCESSORS['custom'] = MyCustomProcessor()
```

## Advanced Features

### Session Persistence

Sessions are stored in memory by default. For production, implement persistent storage:

```python
# Example with Redis
import redis

redis_client = redis.Redis(host='localhost', port=6379)

def save_session(session):
    redis_client.set(
        f"session:{session.session_id}",
        json.dumps(session.to_dict())
    )
```

### Streaming Responses

For long responses, use streaming:

```python
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        # Process in chunks
        for chunk in process_message_chunks(request.message):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Custom UI Themes

Modify the CSS in `web/chat.html`:

```css
/* Dark theme example */
body {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
}

.chat-container {
    background: #0f3460;
}

.message.assistant .message-content {
    background: #1a1a2e;
    color: #e94560;
}
```

## Troubleshooting

### Chat not loading

1. Check server is running: `curl http://localhost:8002/health`
2. Check browser console for errors
3. Clear browser cache and reload

### Messages not sending

1. Check WebSocket connection
2. Verify session ID is valid
3. Check server logs for errors

### Model errors

1. Ensure all dependencies are installed
2. Check model is loaded correctly
3. Try switching to 'simple' model first

### High latency

1. Use lighter models for quick responses
2. Enable model caching
3. Consider using ONNX models
4. Reduce context window size

## Performance Tips

1. **Use appropriate models** - Don't use AGI Agent for simple queries
2. **Limit history** - Keep chat history manageable
3. **Cache responses** - Cache common queries
4. **Async processing** - Use async/await properly
5. **Connection pooling** - Reuse connections

## Security Considerations

1. **Input validation** - Sanitize all user inputs
2. **Rate limiting** - Prevent abuse
3. **Session management** - Implement proper timeouts
4. **CORS** - Configure appropriately for production
5. **Authentication** - Add user authentication for production

## Deployment

### Docker

```bash
# Build image
docker build -t orion-chat -f Dockerfile.chat .

# Run container
docker run -p 8002:8002 orion-chat
```

### Production

```bash
# Use gunicorn for production
gunicorn src.api.chat_api:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8002
```

## Contributing

To add new chat features:

1. Create a new processor in `src/api/enhanced_chat.py`
2. Register it in `PROCESSORS` or `ENHANCED_PROCESSORS`
3. Add UI options in `web/chat.html`
4. Document in this guide
5. Submit a pull request

## Examples Gallery

See `examples/chat/` for:
- Custom processor examples
- Integration examples
- UI customization examples
- Advanced usage patterns

## Support

- GitHub Issues: [Report bugs](https://github.com/your-org/orion-agi/issues)
- Documentation: [Full docs](../README.md)
- API Reference: http://localhost:8002/docs
