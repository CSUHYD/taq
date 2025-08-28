import json
import os
import re
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
    operated_item_ids: list[str] | None = None  # 需要标记为已操作的物体ID列表
    operated_item_names: list[str] | None = None  # 可选: 名称列表（当没有ID时）


class OperationSelection(BaseModel):
    selected_item_ids: list[str]
    clarification_question: str | None = None

class AmbiguityCheck(BaseModel):
    ambiguous: bool
    ambiguity_info: str | None = None

class UserPreferences(BaseModel):
    organization_principles: list[str] | None = None
    object_placement_preferences: list[str] | None = None
    constraints_forbidden: list[str] | None = None

class ActResult(BaseModel):
    operation: str
    operated_item_ids: list[str] | None = None
    user_reply: str | None = None

class DesktopItem(BaseModel):
    id: str | None = None
    name: str
    attributes: str | None = None
    operated: bool | None = False

class DesktopScanResult(BaseModel):
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
        self.items: list[DesktopItem] = []
        self.relevant_items: list[dict] = []
        self.conversation_history: list[dict] = []
        self.user_preferences: list[UserPreferences] = []


    def generate_question(self, 
                          items,
                          task_description,
                          ambiguity_info: str = "",
                          strategy: str = "direct-querying"):
        """
        基于任务、物体列表和已知歧义信息生成问题
        
        Args:
            items: 物体列表（dict 或 DesktopItem），仅用于规划，不向用户展示ID
            task_description: 当前任务描述
            ambiguity_info: 已知歧义/不确定信息（初始化为空字符串）
            strategy: 提问策略 ("user-preference-first", "parallel-exploration", "direct-querying")
            
        Returns:
            VLMResponse: 包含推理和问题的响应
        """
        question_config = self.config.get("question_generation")
        if not question_config:
            raise ValueError("question_generation configuration not found")
        strat_key = strategy if strategy in question_config else "direct-querying"
        strat_config = question_config.get(strat_key, {}) or {}
        if 'systext' not in strat_config or 'usertext' not in strat_config:
            raise ValueError(f"question_generation prompt missing for strategy: {strat_key}")
        systext = strat_config["systext"]
        usertext_template = strat_config["usertext"]
        payload_options = strat_config.get("payload_options", {})
        self.current_strategy = strat_key
        
        # Build items block by embedding full items JSON (includes all attributes)
        if not items:
            raise ValueError("No items provided. Please initialize desktop scan first.")
        def _to_dict(it):
            if hasattr(it, 'model_dump'):
                return it.model_dump()
            if isinstance(it, dict):
                return it
            # Fallback minimal mapping
            return {
                'id': getattr(it, 'id', None),
                'name': getattr(it, 'name', None),
                'attributes': getattr(it, 'attributes', None),
                'operated': getattr(it, 'operated', None),
            }
        items_block = json.dumps([_to_dict(it) for it in items], ensure_ascii=False)
        usertext = usertext_template.format(
            task_description=task_description,
            items_block=items_block,
            ambiguity_info=ambiguity_info or "(none)"
        )
        
        # 调用基础API
        raw_question = self.vlm_api.vlm_request_with_format(
            systext=systext,
            usertext=usertext,
            format_schema=VLMResponse.model_json_schema(),
            options=payload_options
        )
        
        # 解析响应
        parsed_response = VLMResponse.model_validate_json(raw_question)
        print(f"Question generation successful")
        
        # 记录问题生成日志
        self.logger.log_question_generation(
            strategy=strategy,
            image_path=None,
            vlm_response=parsed_response
        )
        # Also persist to backend conversation
        self.append_robot_question(
            reasoning=parsed_response.reasoning,
            question=parsed_response.question,
            image_path=None,
        )
        
        return parsed_response

    def ground_objects(self, latest_qa: dict, items):
        """Ground conversation to items using VLM and update self.relevant_items.

        Args:
            latest_qa: dict with keys 'robot' and 'user' (from get_latest_qa())
            items: list of DesktopItem or item dicts

        Returns:
            list[dict]: the updated self.relevant_items (DesktopItem-shaped dicts). None on parse failure.
        """
        if not items:
            return []

        og_cfg = self.config.get("object_grounding")
        if not og_cfg or 'systext' not in og_cfg or 'usertext' not in og_cfg:
            raise ValueError("object_grounding prompt missing in config")
        systext = og_cfg['systext']

        robot = (latest_qa or {}).get("robot") or {}
        user = (latest_qa or {}).get("user") or {}
        robot_question = robot.get("question") or ""
        robot_reasoning = robot.get("reasoning") or ""
        user_response = user.get("response") or ""

        def _to_dict(it):
            if hasattr(it, 'model_dump'):
                return it.model_dump()
            if isinstance(it, dict):
                return it
            return {
                'id': getattr(it, 'id', None),
                'name': getattr(it, 'name', None),
                'attributes': getattr(it, 'attributes', None),
                'operated': getattr(it, 'operated', None),
            }

        # Build surface-matched candidate items from Q&A to guide VLM
        qa_text = f"{robot_question} {user_response}".lower()
        def _name_tokens(nm: str) -> list[str]:
            import re as _re
            return [t for t in _re.split(r"[^a-zA-Z0-9]+", (nm or "").lower()) if len(t) > 2]
        candidates = []
        for it in [ _to_dict(x) for x in items ]:
            nm = (it.get('name') or '').strip()
            if not nm:
                continue
            toks = _name_tokens(nm)
            if any(tok in qa_text for tok in toks):
                candidates.append(nm)
        candidates_block = json.dumps(sorted(list(set(candidates))), ensure_ascii=False)

        items_block = json.dumps([_to_dict(it) for it in items], ensure_ascii=False)
        usertext_template = og_cfg['usertext']
        usertext = usertext_template.format(
            items_block=items_block,
            robot_question=robot_question,
            robot_reasoning=robot_reasoning,
            user_response=user_response,
            candidates_block=candidates_block,
        )

        options = og_cfg.get("payload_options", {"temperature": 0.1, "num_predict": 140})
        # Expect an array of DesktopItem objects
        array_schema = {"type": "array", "items": DesktopItem.model_json_schema()}
        raw = self.vlm_api.vlm_request_with_format(
            systext=systext,
            usertext=usertext,
            format_schema=array_schema,
            options=options,
        )
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                data = []
        except Exception as e:
            print(f"Failed to parse object grounding result: {e}")
            print(f"Raw grounding response: {raw}")
            return None

        # Merge parsed items directly into self.relevant_items (de-dup by id/name)
        existing_ids = {d.get('id') for d in self.relevant_items if d.get('id')}
        existing_names = {d.get('name') for d in self.relevant_items}
        for obj in data:
            try:
                item = DesktopItem.model_validate(obj)
                d = item.model_dump()
                iid = (d.get('id') or '').strip() if d.get('id') else None
                nm = (d.get('name') or '').strip() if d.get('name') else None
                if (iid and iid in existing_ids) or (not iid and nm in existing_names):
                    continue
                self.relevant_items.append(d)
                if iid:
                    existing_ids.add(iid)
                if nm:
                    existing_names.add(nm)
            except Exception:
                continue

        return list(self.relevant_items)

    def preference_parser(self,
                          task_description: str,
                          latest_qa: dict | None = None,
                          items: list | None = None) -> UserPreferences | None:
        """Extract personalized user preferences from the latest Q&A using VLM.

        Returns a UserPreferences object on success and persists it to backend state
        and conversation history; returns None on parsing failure.
        """
        latest_qa = latest_qa or self.get_latest_qa()
        items_source = items
        if items_source is None:
            items_source = self.get_relevant_items() or self.get_desktop_items_snapshot().get('items', [])

        def _to_dict(it):
            if hasattr(it, 'model_dump'):
                return it.model_dump()
            if isinstance(it, dict):
                return it
            return {
                'id': getattr(it, 'id', None),
                'name': getattr(it, 'name', None),
                'attributes': getattr(it, 'attributes', None),
                'operated': getattr(it, 'operated', None),
            }
        items_block = json.dumps([_to_dict(it) for it in (items_source or [])], ensure_ascii=False)

        robot = (latest_qa or {}).get('robot') or {}
        user = (latest_qa or {}).get('user') or {}
        robot_question = robot.get('question') or ''
        robot_reasoning = robot.get('reasoning') or ''
        user_response = user.get('response') or ''

        pp_cfg = self.config.get('preference_parser')
        if not pp_cfg or 'systext' not in pp_cfg or 'usertext' not in pp_cfg:
            raise ValueError("preference_parser prompt missing in config")
        systext = pp_cfg['systext']
        usertext_template = pp_cfg['usertext']
        options = pp_cfg.get('payload_options', {"temperature": 0.2, "num_predict": 220})

        usertext = usertext_template.format(
            task=task_description or self.config.get('task_description', ''),
            items_block=items_block,
            robot_question=robot_question,
            robot_reasoning=robot_reasoning,
            user_response=user_response,
        )

        raw = self.vlm_api.vlm_request_with_format(
            systext=systext,
            usertext=usertext,
            format_schema=UserPreferences.model_json_schema(),
            options=options,
        )
        try:
            prefs = UserPreferences.model_validate_json(raw)
        except Exception as e:
            print(f"Failed to parse user preferences: {e}")
            print(f"Raw preferences response: {raw}")
            return None

        # Persist and log to conversation history
        self.conversation_history.append({
            "type": "preferences",
            "preferences": prefs.model_dump(),
            "timestamp": datetime.now().isoformat(),
        })
        return prefs
    
    def ambiguity_checker(self,
                          task_description: str,
                          latest_qa: dict | None = None,
                          items: list | None = None,
                          user_prefs: 'UserPreferences | dict | None' = None,
                          strategy: str = "direct-querying") -> str | bool:
        """Estimate if current target items have enough info to execute predicate actions.

        Inputs:
          - latest_qa: dict with keys 'robot' and 'user'. If None, uses get_latest_qa().
          - items: list of item dicts or DesktopItem; defaults to self.relevant_items if present,
                   otherwise falls back to current self.items snapshot.
          - current_task: optional task description for context.

        Returns:
          - False if information is sufficient (no ambiguity)
          - str describing ambiguity info (suggested clarifications) if insufficient
        """
        # Resolve context
        latest_qa = latest_qa or self.get_latest_qa()
        items_source = items
        if items_source is None or len(items_source) == 0:
            items_source = self.relevant_items if self.relevant_items else self.get_desktop_items_snapshot().get('items', [])

        # Normalize items to dicts
        def _to_dict(it):
            if hasattr(it, 'model_dump'):
                return it.model_dump()
            if isinstance(it, dict):
                return it
            return {
                'id': getattr(it, 'id', None),
                'name': getattr(it, 'name', None),
                'attributes': getattr(it, 'attributes', None),
                'operated': getattr(it, 'operated', None),
            }
        items_block = json.dumps([_to_dict(it) for it in (items_source or [])], ensure_ascii=False)
        # Preferences block
        if user_prefs is None:
            # fall back to stored preferences or last persisted
            if isinstance(self.user_preferences, list) and len(self.user_preferences) > 0:
                user_prefs = self.user_preferences[-1]
            else:
                for msg in reversed(self.conversation_history):
                    if msg.get('type') == 'preferences':
                        user_prefs = msg.get('preferences')
                        break
        if hasattr(user_prefs, 'model_dump'):
            preferences_block = json.dumps(user_prefs.model_dump(), ensure_ascii=False)
        elif isinstance(user_prefs, dict):
            preferences_block = json.dumps(user_prefs, ensure_ascii=False)
        else:
            preferences_block = json.dumps({}, ensure_ascii=False)

        robot = (latest_qa or {}).get('robot') or {}
        user = (latest_qa or {}).get('user') or {}
        robot_question = robot.get('question') or ''
        robot_reasoning = robot.get('reasoning') or ''
        user_response = user.get('response') or ''

        ac_root = self.config.get('ambiguity_checker')
        if not isinstance(ac_root, dict):
            raise ValueError("ambiguity_checker prompt missing in config")
        ac_cfg = ac_root.get(strategy)
        if not ac_cfg or 'systext' not in ac_cfg or 'usertext' not in ac_cfg:
            raise ValueError(f"ambiguity_checker prompt missing for strategy: {strategy}")
        systext = ac_cfg['systext']
        usertext_template = ac_cfg['usertext']
        payload_options = ac_cfg.get('payload_options', {"temperature": 0.2, "num_predict": 160})

        usertext = usertext_template.format(
            task=task_description or self.config.get('task_description', ''),
            items_block=items_block,
            preferences_block=preferences_block,
            robot_question=robot_question,
            robot_reasoning=robot_reasoning,
            user_response=user_response,
        )

        raw = self.vlm_api.vlm_request_with_format(
            systext=systext,
            usertext=usertext,
            format_schema=AmbiguityCheck.model_json_schema(),
            options=payload_options,
        )

        result = AmbiguityCheck.model_validate_json(raw)
        if result.ambiguous:
            return (result.ambiguity_info or '').strip() or "Ambiguity exists, but details were not provided."
        return False


    def robot_response(self, user_response: str) -> dict:
        """Generate a simple robot reply to user's response.

        Current behavior:
        - Append the user response to backend conversation.
        - Run object grounding against current items using the latest Q&A.
        - Return a concise reply plus the current session's focused (relevant) items.

        Returns a dict: { 'robot_reply': str, 'relevant_items': list[dict] }
        """
        # Record user response
        self.append_user_response(user_response)

        # Build latest Q&A and ground relevant objects
        latest = self.get_latest_qa()
        grounded = self.ground_objects(latest, self.items or [])

        # Parse/update user preferences before ambiguity check
        prefs = self.preference_parser(
            task_description=self.config.get('task_description', ''),
            latest_qa=latest,
            items=self.get_relevant_items() or self.items,
        )
        self.user_preferences.append(prefs)

        # Run ambiguity check to guide next Ask if needed, with preferences context
        ambiguity_info = self.ambiguity_checker(
            task_description=self.config.get('task_description', ''),
            latest_qa=latest,
            items=self.get_relevant_items() or self.items,
            user_prefs=prefs,
            strategy=getattr(self, 'current_strategy', 'direct-querying'),
        )

        # Optionally plan action when unambiguous
        planned_operation = ""
        act_user_reply = None
        if ambiguity_info is False:
            try:
                act = self.plan_action(
                    latest_qa=latest,
                    items=self.get_relevant_items() or self.items,
                    user_prefs=prefs,
                    task_description=self.config.get('task_description', ''),
                )
                if act and getattr(act, 'operation', None):
                    planned_operation = act.operation
                    act_user_reply = getattr(act, "user_reply", None)
                    # Mark operated items in self.items
                    updated = []
                    ids = set((act.operated_item_ids or []))
                    norm = set()
                    for iid in ids:
                        if isinstance(iid, str):
                            s = iid.strip()
                            if s.startswith('#'):
                                s = s[1:]
                            norm.add(s.lower())
                    if norm:
                        for it in self.items:
                            if getattr(it, 'id', None) and it.id.lower() in norm and not getattr(it, 'operated', False):
                                it.operated = True
                                updated.append({"id": it.id, "name": it.name, "operated": True})
                        if updated:
                            self.logger.log_item_status_change(planned_operation, updated)
            except Exception as e:
                print(f"Act planning failed: {e}")

        # Serialize full preferences history (list) for the response
        try:
            pref_list = [p.model_dump() if hasattr(p, 'model_dump') else p for p in (self.user_preferences or [])]
        except Exception:
            pref_list = []

        # Ensure a user-visible reply string exists
        # Priority:
        # 1) Model-provided user reply from action planner
        # 2) Clarification guidance when ambiguous
        # 3) Short confirmation when an operation is planned
        # 4) Generic acknowledgement fallback
        if act_user_reply and isinstance(act_user_reply, str) and act_user_reply.strip():
            reply_text = act_user_reply.strip()
        elif isinstance(ambiguity_info, str) and ambiguity_info.strip():
            reply_text = f"I need a bit more information to proceed: {ambiguity_info.strip()}"
        elif planned_operation and isinstance(planned_operation, str) and planned_operation.strip():
            reply_text = f"Got it. I will proceed: {planned_operation.strip()}"
        else:
            reply_text = "Thanks, I noted your response."

        result = {
            "robot_reply": reply_text,
            "operation": planned_operation,
            "relevant_objects": self.get_relevant_items(),
            "ambiguity": ambiguity_info,
            "preferences": pref_list,
        }
        # Persist to conversation history so history modal can show it
        self.conversation_history.append({
            "type": "robot_response",
            "operation": result["operation"],
            "robot_reply": result["robot_reply"],
            "ambiguity": result["ambiguity"],
            "relevant_objects": result["relevant_objects"],
            "preferences": result.get("preferences"),
            "timestamp": datetime.now().isoformat(),
        })
        return result

    def plan_action(self,
                    latest_qa: dict,
                    items,
                    user_prefs: UserPreferences | dict | None,
                    task_description: str) -> ActResult | None:
        """Plan predicate action using current context and preferences.

        Returns ActResult with operation text and operated_item_ids (raw IDs as in items).
        """
        ap_cfg = self.config.get('act_planner')
        if not ap_cfg or 'systext' not in ap_cfg or 'usertext' not in ap_cfg:
            raise ValueError("act_planner prompt missing in config")

        # Normalize items to dicts
        def _to_dict(it):
            if hasattr(it, 'model_dump'):
                return it.model_dump()
            if isinstance(it, dict):
                return it
            return {
                'id': getattr(it, 'id', None),
                'name': getattr(it, 'name', None),
                'attributes': getattr(it, 'attributes', None),
                'operated': getattr(it, 'operated', None),
            }
        items_block = json.dumps([_to_dict(it) for it in (items or [])], ensure_ascii=False)

        # Preferences block
        if hasattr(user_prefs, 'model_dump'):
            preferences_block = json.dumps(user_prefs.model_dump(), ensure_ascii=False)
        elif isinstance(user_prefs, dict):
            preferences_block = json.dumps(user_prefs, ensure_ascii=False)
        else:
            preferences_block = json.dumps({}, ensure_ascii=False)

        robot = (latest_qa or {}).get('robot') or {}
        user = (latest_qa or {}).get('user') or {}
        robot_question = robot.get('question') or ''
        robot_reasoning = robot.get('reasoning') or ''
        user_response = user.get('response') or ''

        systext = ap_cfg['systext']
        usertext_template = ap_cfg['usertext']
        options = ap_cfg.get('payload_options', {"temperature": 0.2, "num_predict": 160})

        usertext = usertext_template.format(
            task=task_description or self.config.get('task_description', ''),
            items_block=items_block,
            preferences_block=preferences_block,
            robot_question=robot_question,
            robot_reasoning=robot_reasoning,
            user_response=user_response,
        )

        raw = self.vlm_api.vlm_request_with_format(
            systext=systext,
            usertext=usertext,
            format_schema=ActResult.model_json_schema(),
            options=options,
        )
        try:
            return ActResult.model_validate_json(raw)
        except Exception as e:
            print(f"Failed to parse act plan: {e}")
            print(f"Raw act response: {raw}")
            return None

    # Conversation management (backend)
    def append_robot_question(self, reasoning: str, question: str, image_path: str | None = None):
        self.conversation_history.append({
            "type": "robot",
            "reasoning": reasoning,
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
        })

    def append_user_response(self, response_text: str):
        self.conversation_history.append({
            "type": "user",
            "response": response_text,
            "timestamp": datetime.now().isoformat(),
        })

    def get_latest_qa(self) -> dict:
        """Return latest robot question and the next user response if available."""
        last_robot = None
        last_user_after = None
        for msg in reversed(self.conversation_history):
            if last_robot is None:
                if msg.get("type") == "user":
                    last_user_after = msg if last_user_after is None else last_user_after
                elif msg.get("type") == "robot":
                    last_robot = msg
                    break
        return {"robot": last_robot, "user": last_user_after}

    def get_conversation(self) -> list[dict]:
        return list(self.conversation_history)

    def clear_conversation(self):
        self.conversation_history = []

    def get_relevant_items(self) -> list[dict]:
        return list(self.relevant_items)

    def enumerate_desktop_items(self, image_path):
        """Enumerate visible desktop items and return DesktopScanResult."""
        if not image_path:
            return None

        ds_cfg = self.config.get("desktop_scan")
        if not ds_cfg or 'systext' not in ds_cfg or 'usertext' not in ds_cfg:
            raise ValueError("desktop_scan prompt missing in config")
        systext = ds_cfg['systext']
        usertext = ds_cfg['usertext']
        options = ds_cfg.get("payload_options", {"temperature": 0.2, "num_predict": 300})

        try:
            raw = self.vlm_api.vlm_request_with_format(
                systext=systext,
                usertext=usertext,
                format_schema=DesktopScanResult.model_json_schema(),
                image_path1=image_path,
                options=options,
            )
            parsed = DesktopScanResult.model_validate_json(raw)
            return parsed
        except Exception as e:
            print(f"Desktop item enumeration parse failed: {e}")
            print(f"Raw response: {raw if 'raw' in locals() else ''}")
            return None

    def initialize_desktop_state(self, image_path):
        """Run desktop enumeration, store to member, and log the result."""
        scan = self.enumerate_desktop_items(image_path)
        if scan is not None:
            new_items: list[DesktopItem] = []
            for idx, item in enumerate(scan.items, start=1):
                if not hasattr(item, 'operated') or item.operated is None:
                    item.operated = False
                if not getattr(item, 'id', None):
                    base = re.sub(r'[^a-zA-Z0-9]+', '-', (item.name or '').strip().lower()).strip('-')
                    if not base:
                        base = 'item'
                    item.id = f"itm-{idx}-{base}"
                new_items.append(item)
            self.items = new_items
            try:
                self.logger.log_desktop_scan(image_path, self.items, getattr(scan, 'summary', None))
            except Exception as e:
                print(f"Failed to log desktop scan: {e}")
        return scan

    
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

    def get_desktop_items_snapshot(self):
        if not self.items:
            return {"items": [], "summary": None}
        items = [
            {
                "id": getattr(it, 'id', None),
                "name": it.name,
                "attributes": getattr(it, 'attributes', None),
                "operated": getattr(it, 'operated', False),
            }
            for it in self.items
        ]
        return {"items": items, "summary": None}
