import cv2
import os
from datetime import datetime
from robot_service import RobotService
from vlmCall_ollama import load_prompt_config


class VideoQuestionSystem:
    def __init__(self, model="qwen2.5vl:32b", camera_id=0):
        """
        Initialize video question system
        Args:
            model: VLM model name
            camera_id: Camera ID, default is 0
        """
        self.robot_service = RobotService(model)
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self.current_task = None
        
        # Load prompt configuration
        self.prompt_config = load_prompt_config()
        
        # Load task from config (now at root level)
        self.current_task = self.prompt_config.get("task_description", "general home assistance")
        
        # Create directory for saving images
        self.save_dir = "captured_frames"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def start_camera(self):
        """Start camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"Unable to open camera {self.camera_id}")
                return False
            
            # Set camera parameters
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera started successfully")
            return True
            
        except Exception as e:
            print(f"Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera stopped")
    
    def capture_frame(self):
        """Capture current frame and save"""
        if not self.cap:
            print("Camera not started")
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            print("Unable to read camera frame")
            return None
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, frame)
        print(f"Image saved: {filepath}")
        
        return filepath
    
    
    def generate_question(self, image_path):
        """
        Generate question based on image and current task
        Args:
            image_path: Image path
        Returns:
            Generated question text
        """
        try:
            print(f"Generating question for task: {self.current_task}")
            vlm_response = self.robot_service.generate_question(
                task_description=self.current_task,
                image_path=image_path
            )
            
            reasoning = vlm_response.reasoning
            question = vlm_response.question
            
            # Display formatted output
            print(f"\nReasoning: {reasoning}")
            print(f"Question: {question}")
            
            return question
            
        except Exception as e:
            print(f"Failed to generate question: {e}")
            return "Unable to generate question, please try again later."
    
    def run(self):
        """Run main loop"""
        if not self.start_camera():
            return
        
        self.is_running = True
        print("\n" + "="*50)
        print("Video Question System Started")
        print(f"Current Task: {self.current_task}")
        print("Instructions:")
        print("- Press SPACE: Capture current frame and generate question")
        print("- Press 'q': Exit system")
        print("="*50)
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Unable to read camera frame")
                    break
                
                # Add instruction text on frame
                cv2.putText(frame, f"Task: {self.current_task[:40]}...", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, "SPACE: ask question | Q: quit", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display video stream
                cv2.imshow("Video Question System", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space key
                    print("\nCapturing image...")
                    image_path = self.capture_frame()
                    
                    if image_path:
                        question = self.generate_question(image_path)
                        print(f"\nGenerated question: {question}\n")
                        
                        
                elif key == ord('q'):  # q key to quit
                    print("\nExiting system...")
                    self.is_running = False
                    
        except KeyboardInterrupt:
            print("\nUser interrupted, exiting...")
            
        finally:
            self.stop_camera()


def main():
    """Main function"""
    print("Initializing Video Question System...")
    
    # Create system instance
    system = VideoQuestionSystem(model="qwen2.5vl:32b")
    
    # Run system
    system.run()


if __name__ == "__main__":
    main()