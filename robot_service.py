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


class RobotService:
    """机器人业务逻辑服务"""
    
    def __init__(self, model="qwen2.5vl:32b", language="en"):
        self.vlm_api = VLMAPI(model)
        self.language = language  # 默认英文，可选 "en" 或 "zh"
        self.config = load_prompt_config(language)
    
    def set_language(self, language):
        """设置对话语言"""
        if language not in ["en", "zh"]:
            raise ValueError("Language must be 'en' or 'zh'")
        self.language = language
        # Reload config for new language
        self.config = load_prompt_config(language)
    
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
        # 使用当前语言的配置和指定策略
        question_config = self.config.get("question_generation", {})
        if not question_config:
            raise ValueError("question_generation configuration not found")
        
        # 获取指定策略的配置
        lang_config = question_config.get(strategy, {})
        if not lang_config:
            # 如果指定策略不存在，使用默认策略
            lang_config = question_config.get("user-preference-first", {})
            if not lang_config:
                raise ValueError(f"Strategy '{strategy}' not found and no default strategy available")
        
        systext = lang_config.get("systext", "")
        usertext_template = lang_config.get("usertext", "")
        payload_options = lang_config.get("payload_options", {})
        
        # 格式化用户文本
        usertext = usertext_template.format(task_description=task_description)
        
        # 构建对话历史上下文
        conversation_context = self._build_conversation_context(messages_history)
        if conversation_context:
            usertext = usertext.replace("{conversation_context}", conversation_context)
        
        # 调用基础API
        raw_response = self.vlm_api.vlm_request_with_format(
            systext=systext,
            usertext=usertext,
            format_schema=VLMResponse.model_json_schema(),
            image_path1=image_path,
            options=payload_options
        )
        
        # 解析响应
        try:
            parsed_response = VLMResponse.model_validate_json(raw_response)
            print(f"Question generation successful")
            return parsed_response
        except Exception as parse_error:
            print(f"Failed to parse question response: {parse_error}")
            print(f"Raw response: {raw_response}")
            return VLMResponse(reasoning="", question=raw_response)
    
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
        # 使用当前语言的配置
        lang_config = self.config.get("response_analysis", {})
        if not lang_config:
            raise ValueError("response_analysis configuration not found")
        
        systext = lang_config.get("systext", "")
        usertext_template = lang_config.get("usertext", "")
        payload_options = lang_config.get("payload_options", {})
        
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
            return parsed_response
        except Exception as parse_error:
            print(f"Failed to parse response analysis: {parse_error}")
            print(f"Raw response: {raw_response}")
            
            # 根据语言提供错误回复
            if self.language == "zh":
                return ResponseAnalysis(
                    understanding=f"用户说: {user_response}",
                    operation="",
                    robot_reply="我理解您的回应。让我继续完成任务。"
                )
            else:
                return ResponseAnalysis(
                    understanding=f"User said: {user_response}",
                    operation="",
                    robot_reply="I understand your response. Let me continue with the task."
                )
    
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