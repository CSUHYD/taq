# Task-Adaptive Questioning (TAQ) System

An intelligent home service robot system that generates contextually relevant questions based on current tasks and visual scene analysis. The robot adapts its questioning strategy to specific household tasks while collaborating with human operators to execute physical actions.

English | [中文](README_zh.md)

## Core Concept

**Task-Adaptive Questioning** enables the robot to:
- 🎯 Generate task-specific questions based on visual scene analysis
- 🔄 Adapt questioning strategy as tasks progress
- 🤝 Collaborate with human operators for physical task execution
- 📚 Learn from conversation history to improve future interactions

## Key Features

### 🤖 Intelligent Questioning System
- **Visual Scene Analysis**: Real-time camera feed analysis with VLM integration
- **Task-Aware Questions**: Context-sensitive question generation based on current household task
- **Adaptive Dialogue**: Questions evolve based on conversation history and user responses
- **Reasoning Display**: Shows robot's analytical process behind each question

### 🏠 Home Service Integration
- **Operator Collaboration**: Clear instructions for human operators to execute physical actions
- **Task Configuration**: JSON-based task and prompt management
- **Session Management**: Persistent conversation history and context awareness
- **Real-time Status**: Live updates during VLM processing and image capture

### 🌐 Web Interface
- **Live Video Stream**: Real-time camera feed with task status overlay
- **Interactive Chat**: Seamless dialogue between user and robot
- **Conversation History**: Complete interaction log with timestamps and reasoning
- **Responsive Design**: Desktop and mobile-friendly interface

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

## How Task-Adaptive Questioning Works

### 1. **Task Context Setup**
Configure the robot's current task in `config/prompt_config.json`:
```json
{
  "task_description": "organize the kitchen counter"
}
```

### 2. **Visual Scene Analysis**
The robot captures video frames and analyzes them in the context of the current task, considering:
- Objects and layout in the scene
- Task requirements and constraints
- Previous conversation context
- User preferences from dialogue history

### 3. **Adaptive Question Generation**
Based on the analysis, the robot generates questions that:
- Are specific to the current task context
- Build upon previous conversation
- Guide toward effective task completion
- Consider user's preferred interaction style

### 4. **Collaborative Execution**
When users respond, the robot:
- Understands user preferences and instructions
- Generates clear operator instructions when physical actions are needed
- Provides contextual follow-up questions
- Adapts future questioning based on user feedback

## Usage

1. Start the TAQ web application:
```bash
python web_app.py
```

2. Access the interface at `http://localhost:5050`

3. **Task-Adaptive Interaction Flow**:
   - 📸 **Ask Question**: Robot analyzes current scene within task context
   - 💬 **Respond**: User provides preferences or instructions  
   - 🤖 **Robot Adapts**: System updates understanding and generates follow-up questions
   - 📋 **Operator Instructions**: Clear guidance for physical task execution
   - 🔄 **Continue**: Process repeats, adapting to evolving task context

4. **View History**: Click "📋 History" to see complete task interaction log

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

## Task-Adaptive Configuration

### Task Adaptation Examples

The system adapts its questioning approach based on the configured task:

**Kitchen Organization Task**:
```json
{
  "task_description": "organize the kitchen counter"
}
```
- Questions focus on food safety, accessibility, and workflow
- Considers cooking patterns and appliance usage
- Adapts to user's cooking style preferences

**Living Room Cleaning Task**:
```json
{
  "task_description": "tidy up the living room"
}
```  
- Questions about furniture arrangement and decoration
- Considers family usage patterns and comfort
- Adapts to aesthetic preferences and functionality

**Workflow Adaptation**:
The robot's questioning strategy evolves as tasks progress:
1. **Initial Assessment**: Broad questions about preferences and priorities
2. **Focused Inquiry**: Specific questions about challenging areas or items
3. **Execution Guidance**: Detailed questions about implementation details
4. **Quality Assurance**: Questions about satisfaction and adjustments

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

## TAQ System Architecture

```
taq/  # Task-Adaptive Questioning System
├── web_app.py              # Web interface and video streaming
├── robot_service.py        # Business logic and task adaptation
├── vlmCall_ollama.py       # Basic VLM API integration
├── video_question_system.py # CLI interface for TAQ
├── config/
│   └── prompt_config.json  # Task-specific prompts and configuration
├── templates/
│   └── index.html         # Responsive web interface
├── test_architecture.py   # Architecture validation tests
├── requirements.txt       # Python dependencies
└── README.md             # TAQ system documentation
```

### Architecture Principles

**Layered Design**:
- **API Layer** (`vlmCall_ollama.py`): Pure VLM API communication
- **Business Logic** (`robot_service.py`): Task adaptation and question generation
- **Application Layer** (`web_app.py`, `video_question_system.py`): User interfaces

**Task Adaptation Core**:
- JSON-based task and prompt configuration
- Context-aware question generation
- Conversation history integration
- Operator collaboration protocols

## Development & Customization

### Extending Task-Adaptive Capabilities

1. **New Task Types**: Add task-specific prompts in `config/prompt_config.json`
2. **Question Strategies**: Modify business logic in `robot_service.py`
3. **UI Enhancements**: Update web interface in `templates/index.html`
4. **Integration**: Add new VLM capabilities via `vlmCall_ollama.py`

### Task Adaptation Research

The TAQ system provides a foundation for research in:
- **Context-Aware Robotics**: How robots adapt behavior to specific tasks
- **Human-Robot Collaboration**: Effective questioning strategies for task completion
- **Conversational AI**: Dialogue systems that learn and adapt over time
- **Vision-Language Integration**: Combining visual perception with natural language understanding

### Example Applications

- **Elderly Care**: Adaptive assistance based on individual needs and preferences
- **Industrial Training**: Context-aware guidance for complex procedures
- **Educational Robotics**: Personalized learning through adaptive questioning
- **Smart Home Systems**: Intelligent assistance that learns household patterns

The system demonstrates how robots can become more effective collaborators by asking the right questions at the right time, adapting their approach based on task context and user preferences.