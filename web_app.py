from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import cv2
import zmq
import numpy as np
import struct
import time
import threading
from datetime import datetime
from robot_service import RobotService
from vlmCall_ollama import load_prompt_config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

class WebVideoSystem:
    def __init__(self, model="qwen2.5vl:32b", camera_id=0, use_zmq_source=False, zmq_server_address="192.168.123.164", zmq_port=5555):
        # Load prompt configuration (English only)
        self.prompt_config = load_prompt_config()
        
        self.robot_service = RobotService(model)
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self.current_status = "Ready"
        self.conversation_history = []
        
        self.current_task = self.prompt_config.get("task_description", "general home assistance")
        
        # Video source configuration
        self.use_zmq_source = use_zmq_source
        self.zmq_server_address = zmq_server_address
        self.zmq_port = zmq_port
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Initialize video source
        if self.use_zmq_source:
            self.start_zmq_receiver()
        else:
            self.start_camera()
        
        # Start initial logging session
        session_id = self.robot_service.start_logging_session(self.current_task)
        self.robot_service.logger.current_custom_session_id = None
        print(f"Started initial logging session: {session_id}")
        
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
    
    def start_zmq_receiver(self):
        """Start ZMQ image receiver in a separate thread"""
        self.zmq_running = True
        self.zmq_thread = threading.Thread(target=self._zmq_receive_loop, daemon=True)
        self.zmq_thread.start()
        print(f"ZMQ receiver started, connecting to {self.zmq_server_address}:{self.zmq_port}")
    
    def _zmq_receive_loop(self):
        """ZMQ receive loop running in separate thread"""
        try:
            # Set up ZeroMQ context and socket
            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            socket.connect(f"tcp://{self.zmq_server_address}:{self.zmq_port}")
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
            
            print("ZMQ image receiver connected and waiting for data...")
            
            while self.zmq_running:
                try:
                    # Receive message
                    message = socket.recv()
                    
                    # Decode image (assuming no header for simplicity, like image_client.py without Unit_Test)
                    np_img = np.frombuffer(message, dtype=np.uint8)
                    current_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                    
                    if current_image is not None:
                        # Resize image like in image_client.py
                        height, width = current_image.shape[:2]
                        resized_image = cv2.resize(current_image, (width // 2, height // 2))
                        
                        # Update current frame thread-safely
                        with self.frame_lock:
                            self.current_frame = resized_image.copy()
                    
                except zmq.Again:
                    # Timeout, continue loop
                    continue
                except Exception as e:
                    print(f"Error in ZMQ receive loop: {e}")
                    time.sleep(0.1)
            
            socket.close()
            context.term()
            print("ZMQ receiver stopped")
            
        except Exception as e:
            print(f"Failed to start ZMQ receiver: {e}")
    
    def get_frame(self):
        """Get current frame from either camera or ZMQ source"""
        if self.use_zmq_source:
            with self.frame_lock:
                return self.current_frame.copy() if self.current_frame is not None else None
        else:
            if not self.cap:
                return None
                
            ret, frame = self.cap.read()
            if not ret:
                return None
                
            return frame
    
    def capture_and_analyze(self, strategy="user-preference-first", session_id=None):
        """Capture frame and generate question"""
        try:
            # Update status
            self.current_status = "Capturing image..."
            socketio.emit('status_update', {'status': self.current_status})
            
            frame = self.get_frame()
            if frame is None:
                return {"error": "Failed to capture frame"}
            
            # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_frames/web_frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)

            # Using initialized items; no re-scan here

            # Update status
            self.current_status = "Thinking..."
            socketio.emit('status_update', {'status': self.current_status})
            
            # Handle session_id changes
            current_session = self.robot_service.logger.current_session
            if session_id:
                # Check if we need to start a new session with custom ID
                current_custom_id = getattr(self.robot_service.logger, 'current_custom_session_id', None) if current_session else None
                if not current_session or current_custom_id != session_id:
                    # End current session and start new one
                    if current_session:
                        ended_session = self.robot_service.end_logging_session()
                        print(f"Ended session due to session_id change: {ended_session}")
                    new_session = self.robot_service.start_logging_session(self.current_task, session_id)
                    self.robot_service.logger.current_custom_session_id = session_id
                    print(f"Started new session with custom ID: {new_session}")
            elif not current_session:
                # Start auto session if no session exists
                new_session = self.robot_service.start_logging_session(self.current_task)
                self.robot_service.logger.current_custom_session_id = None
                print(f"Started auto session: {new_session}")
            
            print(f"Debug - Generating question for task: {self.current_task}")
            
            # Use robot service to generate question
            vlm_response = self.robot_service.generate_question(
                task_description=self.current_task,
                image_path=filename,
                messages_history=self.conversation_history,
                strategy=strategy
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
            self.current_status = "Ready"
            socketio.emit('status_update', {'status': self.current_status})
            
            return {
                "reasoning": reasoning,
                "question": question,
                "success": True
            }
            
        except Exception as e:
            self.current_status = "Error occurred"
            socketio.emit('status_update', {'status': self.current_status})
            return {"error": str(e)}

    def init_desktop_scan(self):
        """Capture frame and initialize desktop state in RobotService"""
        try:
            self.current_status = "Capturing image..."
            socketio.emit('status_update', {'status': self.current_status})

            frame = self.get_frame()
            if frame is None:
                return {"success": False, "error": "Failed to capture frame"}

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_frames/init_frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)

            self.current_status = "Thinking..."
            socketio.emit('status_update', {'status': self.current_status})

            scan = self.robot_service.initialize_desktop_state(filename)
            self.current_status = "Ready"
            socketio.emit('status_update', {'status': self.current_status})

            if not scan:
                return {"success": False, "error": "Scan failed"}

            # Return a light summary to frontend from RobotService state
            snapshot = self.robot_service.get_desktop_items_snapshot()
            return {"success": True, "items": snapshot.get('items', []), "summary": snapshot.get('summary')}
        except Exception as e:
            self.current_status = "Error occurred"
            socketio.emit('status_update', {'status': self.current_status})
            return {"success": False, "error": str(e)}
    
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
            self.current_status = "Analyzing response..."
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
                "operation": analysis.operation,
                "robot_reply": analysis.robot_reply,
                "timestamp": datetime.now().isoformat()
            }
            self.conversation_history.append(robot_response)
            
            # Update status
            self.current_status = "Ready"
            socketio.emit('status_update', {'status': self.current_status})
            
            print(f"Debug - User response analysis:")
            print(f"Understanding: {analysis.understanding}")
            print(f"Operation: {analysis.operation}")
            print(f"Robot reply: {analysis.robot_reply}")
            
            return {
                "success": True,
                "understanding": analysis.understanding,
                "operation": analysis.operation,
                "robot_reply": analysis.robot_reply
            }
            
        except Exception as e:
            self.current_status = "Error occurred"
            socketio.emit('status_update', {'status': self.current_status})
            return {"error": str(e)}
    
    def restart_conversation(self):
        """Clear conversation history and restart"""
        # End current logging session if exists
        if hasattr(self.robot_service, 'logger') and self.robot_service.logger.current_session:
            session_id = self.robot_service.end_logging_session()
            print(f"Ended logging session: {session_id}")
        
        # Start new logging session (auto session on restart)
        session_id = self.robot_service.start_logging_session(self.current_task)
        self.robot_service.logger.current_custom_session_id = None
        print(f"Started new logging session: {session_id}")
        
        self.conversation_history = []
        self.current_status = "Ready"
        print("Conversation history cleared")
        return {"success": True, "message": "Conversation restarted"}

# Global system instance
# video_system = WebVideoSystem(use_zmq_source=False)  # Set to True to use ZMQ source from image_client.py

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
    data = request.get_json() or {}
    strategy = data.get('strategy', 'user-preference-first')
    session_id = data.get('session_id')
    result = video_system.capture_and_analyze(strategy=strategy, session_id=session_id)
    return jsonify(result)

@app.route('/init_scan', methods=['POST'])
def init_scan():
    """Initialize desktop items state"""
    result = video_system.init_desktop_scan()
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
    return jsonify({
        "status": video_system.current_status,
        "task": video_system.current_task
    })

@app.route('/logging_status', methods=['GET'])
def get_logging_status():
    """Get current logging session status"""
    summary = video_system.robot_service.get_session_summary()
    return jsonify({'logging_session': summary})

@app.route('/desktop_items', methods=['GET'])
def get_desktop_items():
    """Get current desktop items snapshot"""
    try:
        data = video_system.robot_service.get_desktop_items_snapshot()
        return jsonify({"success": True, **data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "items": [], "summary": None})

@app.route('/log_timing', methods=['POST'])
def log_timing():
    """Log timing data from frontend"""
    data = request.get_json() or {}
    timing_data = data.get('timing_data', {})
    
    try:
        video_system.robot_service.log_timing_data(timing_data)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/restart', methods=['POST'])
def restart_conversation():
    """Restart conversation"""
    result = video_system.restart_conversation()
    socketio.emit('conversation_restarted', {'message': 'Conversation restarted'})
    return jsonify(result)

## Removed language switching endpoint (English-only build)

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
    import argparse
    
    parser = argparse.ArgumentParser(description="Web Video Question System")
    parser.add_argument("--use-zmq", action="store_true", help="Use ZMQ source from image_client.py instead of camera")
    parser.add_argument("--zmq-server", default="192.168.123.164", help="ZMQ server address (default: 192.168.123.164)")
    parser.add_argument("--zmq-port", type=int, default=5555, help="ZMQ server port (default: 5555)")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera ID for local camera (default: 0)")
    
    args = parser.parse_args()
    
    # Create video system with specified configuration
    video_system = WebVideoSystem(
        camera_id=args.camera_id,
        use_zmq_source=args.use_zmq,
        zmq_server_address=args.zmq_server,
        zmq_port=args.zmq_port
    )
    
    print("Starting web application...")
    print(f"Current task: {video_system.current_task}")
    print(f"Video source: {'ZMQ from ' + args.zmq_server + ':' + str(args.zmq_port) if args.use_zmq else 'Camera ' + str(args.camera_id)}")
    
    socketio.run(app, host='0.0.0.0', port=5050, debug=True, allow_unsafe_werkzeug=True)
