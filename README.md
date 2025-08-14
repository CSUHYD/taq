# Home Service Robot - Web Interface

A web-based interface for the home service robot's video question system with real-time video streaming and interactive dialogue.

## Features

- **Real-time Video Streaming**: Live camera feed display in the browser
- **Status Updates**: Real-time status updates (Ready, Thinking, Capturing, etc.)
- **Interactive Dialogue**: Chat interface for robot questions and user responses
- **Task-based Questions**: Robot generates questions based on configured tasks
- **Structured Responses**: Robot provides reasoning and questions separately
- **Conversation History**: Full conversation log with timestamps

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your camera is connected and accessible.

3. Configure the task in `config/prompt_config.json`:
```json
{
  "task_description": "organize the table"
}
```

## Usage

1. Start the web application:
```bash
python web_app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. The interface will show:
   - **Left Panel**: Live video feed with current task and status
   - **Right Panel**: Conversation interface

4. Click "📸 Ask Question" to:
   - Capture current video frame
   - Send to VLM for analysis
   - Generate a task-specific question

5. Respond to robot questions in the chat interface

## Interface Components

### Video Section
- **Live Video Stream**: Real-time camera feed
- **Status Bar**: Current system status with visual indicators
- **Task Display**: Shows current configured task
- **Capture Button**: Triggers image capture and question generation

### Chat Section
- **Conversation Area**: Shows full dialogue history
- **Robot Messages**: Include reasoning and questions
- **User Responses**: Your answers to robot questions
- **Response Input**: Text field for typing responses

### Status Indicators
- 🟢 **Ready**: System ready for new questions
- 🟡 **Thinking**: VLM processing (animated)
- 🟡 **Capturing**: Taking screenshot
- 🔴 **Error**: System error occurred

## Configuration

### Task Configuration
Edit `config/prompt_config.json` to change the robot's task:
```json
{
  "task_description": "clean the kitchen",
  "question_generation": {
    "systext": "...",
    "usertext": "..."
  }
}
```

### Camera Settings
Modify camera settings in `web_app.py`:
```python
# Change camera resolution
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Change camera ID (for multiple cameras)
camera_id = 0  # Try 1, 2, etc. for other cameras
```

## API Endpoints

- `GET /` - Main web interface
- `GET /video_feed` - Video streaming endpoint
- `POST /capture_and_ask` - Capture frame and generate question
- `POST /respond` - Submit user response
- `GET /conversation` - Get conversation history
- `GET /status` - Get current status and task

## WebSocket Events

- `status_update` - Real-time status updates
- `connect/disconnect` - Client connection events

## Troubleshooting

### Camera Issues
- Ensure camera permissions are granted
- Try different camera IDs (0, 1, 2, etc.)
- Check if camera is used by other applications

### VLM Connection Issues
- Verify VLM API endpoint in `vlmCall_ollama.py`
- Check network connectivity to VLM server
- Ensure model is available and loaded

### Performance Issues
- Reduce video resolution for better performance
- Increase frame delay in video streaming
- Check system resources (CPU, memory)

## File Structure

```
taq/
├── web_app.py              # Flask web application
├── vlmCall_ollama.py       # VLM API integration
├── video_question_system.py # Original CLI system
├── config/
│   └── prompt_config.json  # Task and prompt configuration
├── templates/
│   └── index.html         # Web interface
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Development

To modify the interface:
1. **Backend**: Edit `web_app.py` for API changes
2. **Frontend**: Edit `templates/index.html` for UI changes
3. **Prompts**: Edit `config/prompt_config.json` for VLM behavior
4. **Styling**: Modify CSS in `index.html` for appearance changes

The application uses Flask-SocketIO for real-time communication and OpenCV for video processing.