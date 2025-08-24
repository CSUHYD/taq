import json
import os
import shutil
from datetime import datetime
from pydantic import BaseModel
from vlmCall_ollama import VLMAPI, load_prompt_config


# Pydantic models for business logic
class VLMResponse(BaseModel):
    reasoning: str
    question: str


class ResponseAnalysis(BaseModel):
    understanding: str  # 对用户回答的理解
    operation: str     # 具体操作描述 (空字符串表示不需要动作)
    robot_reply: str   # 机器人的回复


class DesktopItem(BaseModel):
    name: str
    count: int | None = None
    attributes: str | None = None


class DesktopScan(BaseModel):
    items: list[DesktopItem]
    summary: str | None = None


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
    
    def log_question_generation(self, strategy, image_path, vlm_response, conversation_history=None):
        """记录问题生成"""
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
            "conversation_history": conversation_history,
            "robot_reasoning": vlm_response.reasoning,
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


class RobotService:
    """机器人业务逻辑服务（English only）"""

    def __init__(self, model="qwen2.5vl:32b"):
        self.vlm_api = VLMAPI(model)
        self.config = load_prompt_config()
        self.logger = ExperimentLogger()
    
    def _build_conversation_context(self, messages_history):
        """构建对话历史上下文"""
        conversation_context = ""
        if messages_history:
            for msg in messages_history:
                if msg.get("type") == "robot":
                    if msg.get("reasoning"):
                        conversation_context += f"Robot reasoning: {msg['reasoning']}\n"
                    if msg.get("question"):
                        conversation_context += f"Robot question: {msg['question']}\n"
                elif msg.get("type") == "user":
                    if msg.get("response"):
                        conversation_context += f"User response: {msg['response']}\n"
            conversation_context += "\n"
        return conversation_context
    
    def generate_question(self, 
                         task_description, 
                         image_path=None, 
                         messages_history=None,
                         strategy="user-preference-first"):
        """
        生成基于任务和图像的问题
        
        Args:
            task_description: 当前任务描述
            image_path: 图像路径
            messages_history: 对话历史
            strategy: 提问策略 ("user-preference-first", "parallel-exploration", "direct-querying")
            
        Returns:
            VLMResponse: 包含推理和问题的响应
        """
        # Use config and specified strategy (English only)
        question_config = self.config.get("question_generation", {})
        if not question_config:
            raise ValueError("question_generation configuration not found")
        
        # Get strategy config
        strat_config = question_config.get(strategy, {})
        if not strat_config:
            strat_config = question_config.get("user-preference-first", {})
            if not strat_config:
                raise ValueError(f"Strategy '{strategy}' not found and no default strategy available")

        systext = strat_config.get("systext", "")
        usertext_template = strat_config.get("usertext", "")
        payload_options = strat_config.get("payload_options", {})
        
        # 构建对话历史上下文
        conversation_context = self._build_conversation_context(messages_history)
        
        # 格式化用户文本，包括对话历史
        usertext = usertext_template.format(
            task_description=task_description,
            conversation_context=conversation_context
        )
        
        # 调用基础API
        raw_question = self.vlm_api.vlm_request_with_format(
            systext=systext,
            usertext=usertext,
            format_schema=VLMResponse.model_json_schema(),
            image_path1=image_path,
            options=payload_options
        )
        
        # 解析响应
        try:
            parsed_response = VLMResponse.model_validate_json(raw_question)
            print(f"Question generation successful")
            
            # 记录问题生成日志
            self.logger.log_question_generation(
                strategy=strategy,
                image_path=image_path,
                vlm_response=parsed_response,
                conversation_history=messages_history
            )
            
            return parsed_response
        except Exception as parse_error:
            print(f"Failed to parse question response: {parse_error}")
            print(f"Raw response: {raw_question}")
            fallback_response = VLMResponse(reasoning="", question=raw_question)
            
            # 记录问题生成日志（即使解析失败）
            self.logger.log_question_generation(
                strategy=strategy,
                image_path=image_path,
                vlm_response=fallback_response,
                conversation_history=messages_history
            )
            
            return fallback_response
    
    def analyze_user_response(self,
                            user_response,
                            robot_question,
                            robot_reasoning,
                            current_task,
                            messages_history=None):
        """
        分析用户回答并生成机器人响应
        
        Args:
            user_response: 用户回答
            robot_question: 机器人之前的问题
            robot_reasoning: 机器人之前的推理
            current_task: 当前任务
            messages_history: 对话历史
            
        Returns:
            ResponseAnalysis: 包含理解、动作和回复的分析结果
        """
        # English config
        ra_config = self.config.get("response_analysis", {})
        if not ra_config:
            raise ValueError("response_analysis configuration not found")

        systext = ra_config.get("systext", "")
        usertext_template = ra_config.get("usertext", "")
        payload_options = ra_config.get("payload_options", {})
        
        # 构建对话历史上下文
        conversation_context = self._build_conversation_context(messages_history)
        
        # 格式化用户文本
        usertext = usertext_template.format(
            current_task=current_task,
            conversation_context=conversation_context,
            robot_question=robot_question,
            robot_reasoning=robot_reasoning,
            user_response=user_response
        )
        
        # 调用基础API
        raw_response = self.vlm_api.vlm_request_with_format(
            systext=systext,
            usertext=usertext,
            format_schema=ResponseAnalysis.model_json_schema(),
            options=payload_options
        )
        
        # 解析响应
        try:
            parsed_response = ResponseAnalysis.model_validate_json(raw_response)
            print(f"Response analysis successful")
            
            # 记录用户回应日志
            self.logger.log_user_response(user_response, parsed_response)
            
            return parsed_response
        except Exception as parse_error:
            print(f"Failed to parse response analysis: {parse_error}")
            print(f"Raw response: {raw_response}")
            
            # English-only fallback
            fallback_response = ResponseAnalysis(
                understanding=f"User said: {user_response}",
                operation="",
                robot_reply="I understand your response. Let me continue with the task."
            )
            
            # 记录用户回应日志（即使解析失败）
            self.logger.log_user_response(user_response, fallback_response)
            
            return fallback_response
    
    def process_conversation_turn(self,
                                user_response,
                                last_robot_message,
                                current_task,
                                conversation_history):
        """
        处理一轮完整的对话交互
        
        Args:
            user_response: 用户回答
            last_robot_message: 上一个机器人消息
            current_task: 当前任务
            conversation_history: 完整对话历史
            
        Returns:
            ResponseAnalysis: 分析结果
        """
        return self.analyze_user_response(
            user_response=user_response,
            robot_question=last_robot_message.get("question", ""),
            robot_reasoning=last_robot_message.get("reasoning", ""),
            current_task=current_task,
            messages_history=conversation_history
        )

    def enumerate_desktop_items(self, image_path):
        """Enumerate visible desktop items (English only)."""
        if not image_path:
            return None

        systext = (
            "You are a vision assistant. Identify only items visible on a desk/table surface."
            " Return concise object names and optional counts. Do not hallucinate."
        )
        usertext = (
            "Analyze the image and list distinct desktop items."
            " Group similar items and provide counts when clear; omit if unsure."
            " Use short nouns."
        )

        try:
            raw = self.vlm_api.vlm_request_with_format(
                systext=systext,
                usertext=usertext,
                format_schema=DesktopScan.model_json_schema(),
                image_path1=image_path,
                options={"temperature": 0.2, "num_predict": 300}
            )
            parsed = DesktopScan.model_validate_json(raw)
            return parsed
        except Exception as e:
            print(f"Desktop item enumeration parse failed: {e}")
            print(f"Raw response: {raw if 'raw' in locals() else ''}")
            return None
    
    def start_logging_session(self, task_description, custom_session_id=None):
        """开始日志记录会话（English only）"""
        return self.logger.start_session(task_description, "en", custom_session_id)
    
    def end_logging_session(self):
        """结束日志记录会话"""
        return self.logger.end_session()
    
    def get_session_summary(self):
        """获取当前会话摘要"""
        return self.logger.get_session_summary()
    
    def log_timing_data(self, timing_data):
        """记录任务计时数据"""
        return self.logger.log_timing_data(timing_data)
