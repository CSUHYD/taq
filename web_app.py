from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import cv2
import time
from datetime import datetime
from robot_service import RobotService
from vlmCall_ollama import load_prompt_config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

class WebVideoSystem:
    def __init__(self, model="qwen2.5vl:32b", camera_id=0):
        # Load prompt configuration first
        self.prompt_config = load_prompt_config()
        self.language = self.prompt_config.get("language", "en")  # 从配置文件读取语言设置
        
        self.robot_service = RobotService(model, language=self.language)
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self.current_status = "Ready" if self.language == "en" else "就绪"
        self.conversation_history = []
        
        self.current_task = self.prompt_config.get("task_description", "general home assistance")
        self.current_task_en = self.prompt_config.get("task_description_en", "general home assistance")
        self.current_task_zh = self.prompt_config.get("task_description_zh", "通用家庭助理")
        
        # Initialize camera
        self.start_camera()
    
    def set_language(self, language):
        """设置系统语言"""
        if language not in ["en", "zh"]:
            raise ValueError("Language must be 'en' or 'zh'")
        self.language = language
        self.robot_service.set_language(language)
        # 更新当前状态的语言
        if self.current_status == "Ready" or self.current_status == "就绪":
            self.current_status = "Ready" if language == "en" else "就绪"
            socketio.emit('status_update', {'status': self.current_status})
        
    def start_camera(self):
        """Start camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"Unable to open camera {self.camera_id}")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera started successfully")
            return True
            
        except Exception as e:
            print(f"Failed to start camera: {e}")
            return False
    
    def get_frame(self):
        """Get current camera frame"""
        if not self.cap:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        return frame
    
    def capture_and_analyze(self):
        """Capture frame and generate question"""
        try:
            # Update status
            self.current_status = "Capturing image..." if self.language == "en" else "正在捕获图像..."
            socketio.emit('status_update', {'status': self.current_status})
            
            frame = self.get_frame()
            if frame is None:
                return {"error": "Failed to capture frame"}
            
            # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_frames/web_frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            # Update status
            self.current_status = "Thinking..." if self.language == "en" else "思考中..."
            socketio.emit('status_update', {'status': self.current_status})
            
            print(f"Debug - Generating question for task: {self.current_task}")
            
            # Use robot service to generate question
            vlm_response = self.robot_service.generate_question(
                task_description=self.current_task,
                image_path=filename,
                messages_history=self.conversation_history
            )
            
            reasoning = vlm_response.reasoning
            question = vlm_response.question
            
            print(f"Debug - Structured response received")
            print(f"Debug - parsed reasoning: {reasoning}")
            print(f"Debug - parsed question: {question}")
            
            # Add to conversation history
            robot_message = {
                "type": "robot",
                "reasoning": reasoning,
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "image_path": filename
            }
            self.conversation_history.append(robot_message)
            
            # Update status
            self.current_status = "Ready" if self.language == "en" else "就绪"
            socketio.emit('status_update', {'status': self.current_status})
            
            return {
                "reasoning": reasoning,
                "question": question,
                "success": True
            }
            
        except Exception as e:
            self.current_status = "Error occurred" if self.language == "en" else "发生错误"
            socketio.emit('status_update', {'status': self.current_status})
            return {"error": str(e)}
    
    def add_user_response(self, response_text):
        """Add user response to conversation and generate robot reply"""
        try:
            # Add user response to history
            user_message = {
                "type": "user",
                "response": response_text,
                "timestamp": datetime.now().isoformat()
            }
            self.conversation_history.append(user_message)
            
            # Get the last robot message for context
            last_robot_message = None
            for msg in reversed(self.conversation_history):
                if msg.get("type") == "robot":
                    last_robot_message = msg
                    break
            
            if not last_robot_message:
                return {"error": "No previous robot question found"}
            
            # Update status
            self.current_status = "Analyzing response..." if self.language == "en" else "分析回应中..."
            socketio.emit('status_update', {'status': self.current_status})
            
            # Analyze user response and generate robot reply
            analysis = self.robot_service.process_conversation_turn(
                user_response=response_text,
                last_robot_message=last_robot_message,
                current_task=self.current_task,
                conversation_history=self.conversation_history[:-1]  # Exclude current user message
            )
            
            # Add robot response to conversation
            robot_response = {
                "type": "robot_response",
                "understanding": analysis.understanding,
                "operator_instructions": analysis.operator_instructions,
                "robot_reply": analysis.robot_reply,
                "timestamp": datetime.now().isoformat()
            }
            self.conversation_history.append(robot_response)
            
            # Update status
            self.current_status = "Ready" if self.language == "en" else "就绪"
            socketio.emit('status_update', {'status': self.current_status})
            
            print(f"Debug - User response analysis:")
            print(f"Understanding: {analysis.understanding}")
            print(f"Operator instructions: {analysis.operator_instructions}")
            print(f"Robot reply: {analysis.robot_reply}")
            
            return {
                "success": True,
                "understanding": analysis.understanding,
                "operator_instructions": analysis.operator_instructions,
                "robot_reply": analysis.robot_reply
            }
            
        except Exception as e:
            self.current_status = "Error occurred" if self.language == "en" else "发生错误"
            socketio.emit('status_update', {'status': self.current_status})
            return {"error": str(e)}
    
    def restart_conversation(self):
        """Clear conversation history and restart"""
        self.conversation_history = []
        self.current_status = "Ready"
        print("Conversation history cleared")
        return {"success": True, "message": "Conversation restarted"}

# Global system instance
video_system = WebVideoSystem()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            frame = video_system.get_frame()
            if frame is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_and_ask', methods=['POST'])
def capture_and_ask():
    """Capture frame and generate question"""
    result = video_system.capture_and_analyze()
    return jsonify(result)

@app.route('/respond', methods=['POST'])
def respond():
    """Handle user response"""
    data = request.get_json()
    response_text = data.get('response', '')
    
    result = video_system.add_user_response(response_text)
    return jsonify(result)

@app.route('/conversation', methods=['GET'])
def get_conversation():
    """Get conversation history"""
    return jsonify(video_system.conversation_history)

@app.route('/status', methods=['GET'])
def get_status():
    """Get current status"""
    current_task_display = video_system.current_task_zh if video_system.language == "zh" else video_system.current_task_en
    return jsonify({
        "status": video_system.current_status,
        "task": current_task_display
    })

@app.route('/restart', methods=['POST'])
def restart_conversation():
    """Restart conversation"""
    result = video_system.restart_conversation()
    socketio.emit('conversation_restarted', {'message': 'Conversation restarted'})
    return jsonify(result)

@app.route('/set_language', methods=['POST'])
def set_language():
    """Set system language"""
    data = request.get_json()
    language = data.get('language', 'en')
    
    try:
        video_system.set_language(language)
        return jsonify({'success': True, 'language': language})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status_update', {'status': video_system.current_status})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    print("Starting web application...")
    print(f"Current task: {video_system.current_task}")
    socketio.run(app, host='0.0.0.0', port=5050, debug=True)