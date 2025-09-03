import json
import os
import shutil
from datetime import datetime


class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, base_log_dir="logs/experiments"):
        self.base_log_dir = base_log_dir
        self.current_session = None
        self.session_data = {}
        
    def start_session(self, task_description, language="en", custom_session_id=None):
        """开始新的实验会话"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create session_id based on new naming convention: {task}_{session_id}_{time}
        # Clean task description for filename use
        clean_task = task_description.replace(" ", "_").replace("/", "_").replace("\\", "_")[:20]
        
        if custom_session_id:
            # Clean custom session ID for filename use
            clean_custom_id = custom_session_id.replace(" ", "_").replace("/", "_").replace("\\", "_")[:30]
            session_id = f"{clean_task}_{clean_custom_id}_{timestamp}"
        else:
            session_id = f"{clean_task}_auto_{timestamp}"
        
        self.current_session = session_id
        session_dir = os.path.join(self.base_log_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        os.makedirs(os.path.join(session_dir, "images"), exist_ok=True)
        
        self.session_data = {
            "session_id": session_id,
            "custom_session_id": custom_session_id,
            "start_time": datetime.now().isoformat(),
            "task_description": task_description,
            "language": language,
            "interactions": [],
            "timing_data": {
                "task_start_time": None,
                "task_end_time": None,
                "total_task_time_seconds": None,
                "interaction_timings": []
            },
            "session_dir": session_dir
        }
        
        return session_id
    
    def log_question_generation(self, strategy, image_path, vlm_response):
        """记录问题生成（仅记录当次问答，不保存对话历史）"""
        if not self.current_session:
            return
            
        # 复制图像到日志目录
        image_filename = None
        if image_path and os.path.exists(image_path):
            timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]  # 精确到毫秒
            image_filename = f"question_{len(self.session_data['interactions'])}_{timestamp}.jpg"
            image_dest = os.path.join(self.session_data["session_dir"], "images", image_filename)
            try:
                shutil.copy2(image_path, image_dest)
            except Exception as e:
                print(f"Failed to copy image: {e}")
                image_filename = None
        
        interaction = {
            "interaction_id": len(self.session_data["interactions"]),
            "timestamp": datetime.now().isoformat(),
            "type": "question_generation",
            "strategy": strategy,
            "image_file": image_filename,
            "original_image_path": image_path,
            "robot_question": vlm_response.question
        }
        
        self.session_data["interactions"].append(interaction)
        self._save_session_data()
    
    def log_user_response(self, user_response, response_analysis):
        """记录用户回应和机器人分析"""
        if not self.current_session or not self.session_data["interactions"]:
            return
        
        # 获取最后一个交互（应该是问题生成）
        last_interaction = self.session_data["interactions"][-1]
        
        # 添加用户响应到最后一个交互
        last_interaction.update({
            "user_response": user_response,
            "user_response_timestamp": datetime.now().isoformat(),
            "robot_understanding": response_analysis.understanding,
            "robot_operation": response_analysis.operation,
            "robot_reply": response_analysis.robot_reply
        })
        
        self._save_session_data()

    def log_desktop_scan(self, image_path, items: list['DesktopItem'], summary: str | None = None):
        """记录桌面物体扫描结果"""
        if not self.current_session:
            return

        image_filename = None
        if image_path and os.path.exists(image_path):
            timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
            image_filename = f"desktop_scan_{len(self.session_data['interactions'])}_{timestamp}.jpg"
            image_dest = os.path.join(self.session_data["session_dir"], "images", image_filename)
            try:
                shutil.copy2(image_path, image_dest)
            except Exception as e:
                print(f"Failed to copy scan image: {e}")
                image_filename = None

        interaction = {
            "interaction_id": len(self.session_data["interactions"]),
            "timestamp": datetime.now().isoformat(),
            "type": "desktop_scan",
            "image_file": image_filename,
            "original_image_path": image_path,
            "scan": {
                "items": [it.model_dump() if hasattr(it, 'model_dump') else it for it in items],
                "summary": summary
            }
        }
        self.session_data["interactions"].append(interaction)
        self._save_session_data()

    def log_timing_data(self, timing_data):
        """记录任务计时数据"""
        if not self.current_session:
            return
        
        if isinstance(timing_data, dict):
            # Update task timing information
            if "start_time" in timing_data:
                self.session_data["timing_data"]["task_start_time"] = timing_data["start_time"]
            if "end_time" in timing_data:
                self.session_data["timing_data"]["task_end_time"] = timing_data["end_time"]
            if "total_time_seconds" in timing_data:
                self.session_data["timing_data"]["total_task_time_seconds"] = timing_data["total_time_seconds"]
            if "interaction_times" in timing_data:
                self.session_data["timing_data"]["interaction_timings"] = timing_data["interaction_times"]
        
        self._save_session_data()
        print(f"Updated timing data for session {self.current_session}")
    
    def end_session(self):
        """结束当前会话"""
        if not self.current_session:
            return
        
        self.session_data["end_time"] = datetime.now().isoformat()
        self.session_data["total_interactions"] = len(self.session_data["interactions"])
        self._save_session_data()
        
        session_id = self.current_session
        self.current_session = None
        self.session_data = {}
        
        return session_id
    
    def _save_session_data(self):
        """保存会话数据到JSON文件"""
        if not self.current_session:
            return
        
        log_file = os.path.join(self.session_data["session_dir"], "session_log.json")
        try:
            # 创建一个不包含session_dir的副本用于保存
            save_data = {k: v for k, v in self.session_data.items() if k != "session_dir"}
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save session data: {e}")
    
    def log_item_status_change(self, operation_text, updated_items):
        """记录物体状态变化（已操作）"""
        if not self.current_session:
            return
        interaction = {
            "interaction_id": len(self.session_data["interactions"]),
            "timestamp": datetime.now().isoformat(),
            "type": "item_status_change",
            "operation_text": operation_text,
            "updated_items": [
                {"id": it.get("id"), "name": it.get("name"), "operated": it.get("operated", False)}
                for it in updated_items
            ]
        }
        self.session_data["interactions"].append(interaction)
        self._save_session_data()

    def log_preference_summary(self, preferences_summary: dict | None):
        """记录任务结束时的偏好总结"""
        if not self.current_session or not preferences_summary:
            return
        interaction = {
            "interaction_id": len(self.session_data["interactions"]),
            "timestamp": datetime.now().isoformat(),
            "type": "preferences_summary",
            "summary": preferences_summary.get("summary"),
            "key_points": preferences_summary.get("key_points"),
        }
        self.session_data["interactions"].append(interaction)
        self._save_session_data()
    
    def get_session_summary(self):
        """获取当前会话摘要"""
        if not self.current_session:
            return None
        
        return {
            "session_id": self.session_data["session_id"],
            "start_time": self.session_data["start_time"],
            "task_description": self.session_data["task_description"],
            "language": self.session_data["language"],
            "total_interactions": len(self.session_data["interactions"]),
            "last_interaction_time": self.session_data["interactions"][-1]["timestamp"] if self.session_data["interactions"] else None
        }
