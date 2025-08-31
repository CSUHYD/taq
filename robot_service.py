import json
import os
import re
import shutil
from datetime import datetime
from pydantic import BaseModel
from vlmCall_ollama import VLMAPI, load_prompt_config


# Pydantic models for business logic
class VLMResponse(BaseModel):
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

# Preferences summarization
class PreferenceSummary(BaseModel):
    summary: str
    key_points: list[str] | None = None


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
        
        # Build items block by embedding items JSON. For direct-querying, prefer unoperated items.
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
        # If direct-querying, filter to unoperated items when available
        filtered_items = items
        tmp = []
        for it in items:
            d = _to_dict(it)
            if not bool(d.get('operated')):
                tmp.append(d)
        if tmp:
            filtered_items = tmp

        items_block = json.dumps([_to_dict(it) for it in filtered_items], ensure_ascii=False)
        usertext = usertext_template.format(
            task_description=task_description,
            items_block=items_block,
            ambiguity_info=ambiguity_info or "(none)",
            user_preferences = self.user_preferences,
            conversation_history = self.conversation_history
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
        systext = og_cfg['systext']

        robot = (latest_qa or {}).get("robot") or {}
        user = (latest_qa or {}).get("user") or {}
        robot_question = robot.get("question") or ""
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


        items_block = json.dumps([_to_dict(it) for it in items], ensure_ascii=False)
        usertext_template = og_cfg['usertext']
        usertext = usertext_template.format(
            items_block=items_block,
            robot_question=robot_question,
            user_response=user_response,
            conversation_history=self.conversation_history,
        )

        options = og_cfg.get("payload_options", {"temperature": 0.1, "num_predict": 140})
        # Expect an array of item ID strings
        array_schema = {"type": "array", "items": {"type": "string"}}
        raw = self.vlm_api.vlm_request_with_format(
            systext=systext,
            usertext=usertext,
            format_schema=array_schema,
            options=options,
        )
        data = json.loads(raw)
        # Normalize helper
        def _norm(v):
            if v is None:
                return None
            s = str(v).strip()
            return s or None

        # Build id -> item dict from provided items
        id_to_item: dict[str, dict] = {}
        for it in items or []:
            dsrc = _to_dict(it)
            nid = _norm(dsrc.get('id'))
            if nid:
                id_to_item[nid] = dsrc

        # Merge matched items by ID into self.relevant_items (de-dup by ID)
        existing_ids = {(_norm(d.get('id')) or '') for d in self.relevant_items if _norm(d.get('id'))}

        if not isinstance(data, list):
            data = []
        for rid in data:
            try:
                iid = _norm(rid)
                if not iid or iid in existing_ids:
                    continue
                src = id_to_item.get(iid)
                if not src:
                    continue  # ignore unknown IDs
                d = dict(src)
                d['id'] = iid  # ensure normalized id
                self.relevant_items.append(d)
                existing_ids.add(iid)
            except Exception:
                continue

        return list(self.relevant_items)

    def preference_parser(self,
                          task_description: str,
                          latest_qa: dict | None = None,
                          conversation_history: list | None = None,
                          preference: list | dict | None = None) -> UserPreferences | None:
        """Extract personalized user preferences from the latest Q&A using VLM.

        Returns a UserPreferences object on success and persists it to backend state
        and conversation history; returns None on parsing failure.
        """
        latest_qa = latest_qa or self.get_latest_qa()
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
        
        robot = (latest_qa or {}).get('robot') or {}
        user = (latest_qa or {}).get('user') or {}
        robot_question = robot.get('question') or ''
        user_response = user.get('response') or ''

        pp_root = self.config.get('preference_parser')
        if not isinstance(pp_root, dict):
            raise ValueError("preference_parser prompt missing in config")
        strategy = getattr(self, 'current_strategy', 'direct-querying')
        pp_cfg = pp_root.get(strategy)
        if not pp_cfg or 'systext' not in pp_cfg or 'usertext' not in pp_cfg:
            # fallback to any flat legacy block if present
            if 'systext' in pp_root and 'usertext' in pp_root:
                pp_cfg = pp_root
            else:
                raise ValueError(f"preference_parser prompt missing for strategy: {strategy}")
        systext = pp_cfg['systext']
        usertext_template = pp_cfg['usertext']
        options = pp_cfg.get('payload_options', {"temperature": 0.2, "num_predict": 220})

        usertext = usertext_template.format(
            task=task_description or self.config.get('task_description', ''),
            preferences_block = preference,
            items_block = self.relevant_items,
            robot_question=robot_question,
            user_response=user_response,
            conversation_history=conversation_history,
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

        # Note: No post-filtering on organization_principles here to avoid
        # over-constraining model outputs. Prompt enforces high-level guidance.

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
                          relevant_items: list | None = None,
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
        items_block = json.dumps([_to_dict(it) for it in (relevant_items or [])], ensure_ascii=False)
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
            user_response=user_response,
            conversation_history=self.conversation_history,
        )

        raw = self.vlm_api.vlm_request_with_format(
            systext=systext,
            usertext=usertext,
            format_schema=AmbiguityCheck.model_json_schema(),
            options=payload_options,
        )

        result = AmbiguityCheck.model_validate_json(raw)
        return result


    def summarize_preferences(self) -> dict | None:
        """Summarize accumulated user preferences via VLM.

        Returns a dict { 'summary': str, 'key_points': list[str] | None } or None on failure.
        """
        # Collect all preferences seen
        prefs_list = []
        try:
            if isinstance(self.user_preferences, list) and self.user_preferences:
                for p in self.user_preferences:
                    if hasattr(p, 'model_dump'):
                        prefs_list.append(p.model_dump())
                    elif isinstance(p, dict):
                        prefs_list.append(p)
            else:
                for msg in self.conversation_history:
                    if msg.get('type') == 'preferences' and isinstance(msg.get('preferences'), (dict, list)):
                        val = msg.get('preferences')
                        if isinstance(val, dict):
                            prefs_list.append(val)
                        else:
                            prefs_list.extend([v for v in val if isinstance(v, dict)])
        except Exception:
            prefs_list = []

        if not prefs_list:
            return { 'summary': 'No explicit user preferences were provided.', 'key_points': None }

        cfg = self.config.get('preference_summary')
        if not cfg or 'systext' not in cfg or 'usertext' not in cfg:
            # Fallback: simple textual merge
            flat_points = []
            for p in prefs_list:
                for key in ('organization_principles', 'object_placement_preferences', 'constraints_forbidden'):
                    arr = p.get(key)
                    if isinstance(arr, list):
                        flat_points.extend([str(x) for x in arr if isinstance(x, (str, int, float))])
            merged = "; ".join(dict.fromkeys([x.strip() for x in flat_points if str(x).strip()])) or 'None'
            return { 'summary': f'User preferences summary: {merged}', 'key_points': None }

        import json as _json
        systext = cfg['systext']
        usertext = cfg['usertext'].format(
            task=self.config.get('task_description', ''),
            preferences_block=_json.dumps(prefs_list, ensure_ascii=False)
        )
        options = cfg.get('payload_options', { 'temperature': 0.2, 'num_predict': 180 })

        # Expect a structured result
        schema = PreferenceSummary.model_json_schema()
        raw = self.vlm_api.vlm_request_with_format(
            systext=systext,
            usertext=usertext,
            format_schema=schema,
            options=options,
        )
        try:
            parsed = PreferenceSummary.model_validate_json(raw)
            return { 'summary': parsed.summary, 'key_points': parsed.key_points }
        except Exception as e:
            print(f"Failed to parse preference summary: {e}")
            print(f"Raw summary response: {raw}")
            return None
        
        
    def robot_response(self, user_response: str) -> dict:
        """Generate robot reply from user's response with clean, readable flow."""
        # 1) Record user response and get latest Q&A
        self.append_user_response(user_response)
        latest = self.get_latest_qa()

        # 2) Ground relevant objects to current conversation
        relevant_items = self.ground_objects(latest, self.items or [])
        # 3) Update preferences and check ambiguity
        prefs = self.preference_parser(
            task_description=self.config.get('task_description', ''),
            latest_qa=latest,
            conversation_history = self.conversation_history,
            preference = self.user_preferences
        )
        self.user_preferences.append(prefs)

        ambiguity_block = self.ambiguity_checker(
            task_description=self.config.get('task_description', ''),
            latest_qa=latest,
            relevant_items=relevant_items,
            user_prefs=prefs,
            strategy=getattr(self, 'current_strategy', 'direct-querying'),
        )

        # 4) Plan action when not ambiguous and mark operated items
        planned_operation, act_user_reply = self._maybe_plan_action_if_unambiguous(
            ambiguity=ambiguity_block.ambiguous,
            relevant_items=relevant_items,
            latest=latest,
            prefs=prefs,
        )

        # 5) Prepare response fields
        pref_list = self._serialize_preferences_list()
        all_operated = self._all_items_operated()

        # Base reply from action/ambiguity context
        reply_text = self._compose_base_reply(
            act_user_reply=act_user_reply,
            ambiguity_info=ambiguity_block.ambiguity_info,
            planned_operation=planned_operation,
        )

        # Completion branch: summarize preferences and override reply
        preferences_summary = None
        if all_operated:
            preferences_summary = self.summarize_preferences() or None
            reply_text = self._compose_completion_reply(preferences_summary)

        # 6) Build result and persist
        result = {
            "robot_reply": reply_text,
            "operation": planned_operation,
            "relevant_objects": self.get_relevant_items(),
            "ambiguity": ambiguity_block.ambiguity_info,
            "preferences": pref_list,
            "completed": all_operated,
            "preferences_summary": preferences_summary,
        }
        self.conversation_history.append({
            "type": "robot_response",
            "operation": result["operation"],
            "robot_reply": result["robot_reply"],
            "ambiguity": result["ambiguity"],
            "relevant_objects": result["relevant_objects"],
            "preferences": result.get("preferences"),
            "completed": result.get("completed", False),
            "preferences_summary": result.get("preferences_summary"),
            "timestamp": datetime.now().isoformat(),
        })
        return result

    # ---------- Internal helpers for robot_response ----------
    def _serialize_preferences_list(self) -> list:
        try:
            return [p.model_dump() if hasattr(p, 'model_dump') else p for p in (self.user_preferences or [])]
        except Exception:
            return []

    def _all_items_operated(self) -> bool:
        return bool(self.items) and all(bool(getattr(it, 'operated', False)) for it in self.items)

    def _maybe_plan_action_if_unambiguous(self, ambiguity, relevant_items, latest, prefs):
        planned_operation = ""
        act_user_reply = None
        if ambiguity is False:
            try:
                act = self.plan_action(
                    latest_qa=latest,
                    relevant_items=relevant_items,
                    user_prefs=prefs,
                    task_description=self.config.get('task_description', ''),
                )
                if act and getattr(act, 'operation', None):
                    planned_operation = act.operation
                    act_user_reply = getattr(act, "user_reply", None)
                    updated = self._mark_operated_items_from_act(act.operated_item_ids or [])
                    if updated:
                        # Log with the known planned operation text
                        self.logger.log_item_status_change(planned_operation, updated)
            except Exception as e:
                print(f"Act planning failed: {e}")
        return planned_operation, act_user_reply

    def _mark_operated_items_from_act(self, operated_item_ids: list[str]) -> list[dict]:
        """Mark items as operated using only operated_item_ids from the planner.

        Returns a list of updated item dicts for logging.
        """
        updated: list[dict] = []
        # Normalize IDs coming from planner output
        norm: set[str] = set()
        for iid in set(operated_item_ids or []):
            if isinstance(iid, str):
                s = iid.strip()
                if s.startswith('#'):
                    s = s[1:]
                if s:
                    norm.add(s.lower())
        if not norm:
            return updated
        for it in self.items:
            if getattr(it, 'id', None) and it.id.lower() in norm and not getattr(it, 'operated', False):
                it.operated = True
                updated.append({"id": it.id, "name": it.name, "operated": True})
        # Prune executed items from the current relevant_items focus list
        self._prune_relevant_items_by_ids(norm)
        return updated

    def _prune_relevant_items_by_ids(self, id_set: set[str]):
        if not isinstance(self.relevant_items, list) or not self.relevant_items:
            return
        kept = []
        for d in self.relevant_items:
            try:
                did = (d.get('id') or '').strip().lower() if isinstance(d, dict) else ''
            except Exception:
                did = ''
            if did and did in id_set:
                continue
            kept.append(d)
        self.relevant_items = kept

    def _compose_base_reply(self, *, act_user_reply, ambiguity_info, planned_operation) -> str:
        if act_user_reply and isinstance(act_user_reply, str) and act_user_reply.strip():
            return act_user_reply.strip()
        if isinstance(ambiguity_info, str) and ambiguity_info.strip():
            return f"I need a bit more information to proceed: {ambiguity_info.strip()}"
        if planned_operation and isinstance(planned_operation, str) and planned_operation.strip():
            return f"Got it. I will proceed: {planned_operation.strip()}"
        return "Thanks, I noted your response."

    def _compose_completion_reply(self, preferences_summary: dict | None) -> str:
        closing = "Great — all items are completed."
        if preferences_summary and preferences_summary.get('summary'):
            return f"{closing} Preference summary: {preferences_summary['summary']}"
        return closing


    def plan_action(self,
                    latest_qa: dict,
                    relevant_items,
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
        items_block = json.dumps([_to_dict(it) for it in (relevant_items or [])], ensure_ascii=False)

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
        user_response = user.get('response') or ''

        systext = ap_cfg['systext']
        usertext_template = ap_cfg['usertext']
        options = ap_cfg.get('payload_options', {"temperature": 0.2, "num_predict": 160})

        usertext = usertext_template.format(
            task=task_description or self.config.get('task_description', ''),
            items_block=items_block,
            preferences_block=preferences_block,
            robot_question=robot_question,
            user_response=user_response,
            conversation_history=self.conversation_history
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
    def append_robot_question(self, question: str, image_path: str | None = None):
        self.conversation_history.append({
            "type": "robot",
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
        # Also reset parsed user preferences on restart
        self.user_preferences = []

    def get_relevant_items(self) -> list[dict]:
        return list(self.relevant_items)

    def _derive_item_type(self, name: str | None, attributes: str | None = None) -> str:
        """Derive a type slug from item name/attributes for ID generation."""
        text = (name or '').strip().lower()
        if not text and attributes:
            text = (attributes or '').strip().lower()
        tokens = re.split(r"[^a-zA-Z0-9]+", text)
        tokens = [t for t in tokens if len(t) > 1]
        base = tokens[0] if tokens else 'item'
        return re.sub(r'[^a-zA-Z0-9]+', '-', base)

    def _normalize_and_assign_ids(self, raw_items: list[DesktopItem | dict]) -> list[DesktopItem]:
        """Normalize items to DesktopItem, ensure flags, and assign IDs as {type}-{编号}."""
        normalized: list[DesktopItem] = []
        type_counts: dict[str, int] = {}
        for obj in raw_items or []:
            try:
                item = obj if isinstance(obj, DesktopItem) else DesktopItem.model_validate(obj)
            except Exception:
                continue
            if not hasattr(item, 'operated') or item.operated is None:
                item.operated = False
            typ = self._derive_item_type(getattr(item, 'name', None), getattr(item, 'attributes', None))
            type_counts[typ] = type_counts.get(typ, 0) + 1
            item.id = f"{typ}-{type_counts[typ]}"
            normalized.append(item)
        return normalized

    def parse_desktop_items(self, image_path) -> list[DesktopItem]:
        """Enumerate and return DesktopItem list with IDs using the rule: 物体类型-编号.

        - Calls VLM desktop_scan
        - Assigns IDs per type and stores result in self.items
        - Logs the scan via ExperimentLogger
        """
        if not image_path:
            return []

        ds_cfg = self.config.get("desktop_scan")
        if not ds_cfg or 'systext' not in ds_cfg or 'usertext' not in ds_cfg:
            raise ValueError("desktop_scan prompt missing in config")
        systext = ds_cfg['systext']
        usertext = ds_cfg['usertext']
        options = ds_cfg.get("payload_options", {"temperature": 0.2, "num_predict": 300})

        try:
            array_schema = {"type": "array", "items": DesktopItem.model_json_schema()}
            raw = self.vlm_api.vlm_request_with_format(
                systext=systext,
                usertext=usertext,
                format_schema=array_schema,
                image_path1=image_path,
                options=options,
            )
            import json as _json
            def _strip_code_fences(s: str) -> str:
                t = (s or "").strip()
                if t.startswith("```") and t.endswith("```"):
                    t = t.strip('`').strip()
                # common case: starts with ```json
                if t.startswith("```json"):
                    t = t[len("```json"):].strip()
                if t.endswith("```"):
                    t = t[:-3].strip()
                return t

            def _extract_json_array(s: str) -> str | None:
                """Attempt to extract a valid JSON array from messy text.

                Strategies:
                - If we can find a matched [...] via bracket counting, return it.
                - Otherwise, cut from first '[' to last '}' and close with ']' (fix trailing comma).
                """
                if not s:
                    return None
                start = s.find('[')
                if start == -1:
                    return None
                # Try bracket matching first
                depth = 0
                i = start
                while i < len(s):
                    ch = s[i]
                    if ch == '[':
                        depth += 1
                    elif ch == ']':
                        depth -= 1
                        if depth == 0:
                            return s[start:i+1]
                    i += 1
                # No matching closing bracket; salvage by cutting at last '}'
                last_obj_end = s.rfind('}')
                if last_obj_end == -1:
                    return None
                candidate = s[start:last_obj_end+1]
                # Remove trailing comma before the last object if present
                # e.g., "...}\n,\n  garbage" -> remove the comma
                # Trim whitespace at the end to inspect preceding char
                trimmed = candidate.rstrip()
                # If the very end is '}', ensure the char before it is not a comma
                # Also handle case like '},\n  }' which isn't valid; we'll only handle final trailing comma
                # Here we simply ensure the substring ends with '}' and not '},'
                if trimmed.endswith('},'):
                    trimmed = trimmed[:-2] + '}'
                candidate = trimmed
                # Close the array
                candidate = candidate + ']'  # we already have the starting '[' from 'start'
                return candidate

            txt = _strip_code_fences(raw)
            try:
                data = _json.loads(txt)
            except Exception:
                # Attempt array extraction if extra text surrounds JSON
                arr = _extract_json_array(txt)
                if arr is None:
                    raise
                data = _json.loads(arr)
            if not isinstance(data, list):
                data = []
            parsed_items: list[DesktopItem] = []
            for obj in data:
                try:
                    parsed_items.append(DesktopItem.model_validate(obj))
                except Exception:
                    continue
            new_items = self._normalize_and_assign_ids(parsed_items)
        except Exception as e:
            print(f"Desktop item parsing failed: {e}")
            print(f"Raw response: {raw if 'raw' in locals() else ''}")
            new_items = []

        # Keep only movable items (defensive; primary filtering via prompt)
        self.items = self._filter_movable_items(new_items)
        # New scan represents fresh instance detection; clear stale grounded items
        self.relevant_items = []
        try:
            self.logger.log_desktop_scan(image_path, self.items, None)
        except Exception as e:
            print(f"Failed to log desktop scan: {e}")
        return list(self.items)

    def _filter_movable_items(self, items: list[DesktopItem]) -> list[DesktopItem]:
        """Heuristically filter out immovable/background structures from scan results.

        Rules (approximate):
        - Exclude furniture/fixtures and surfaces: desk, table, wall, floor, ceiling, window, door
        - Exclude storage structures: cabinet, shelf/shelves, drawer/drawers, countertop
        - Otherwise keep (keyboards, mice, bottles, books, phones, laptops, etc.)
        """
        if not items:
            return []
        blocked = {
            "desk", "table", "wall", "floor", "ceiling", "window", "door",
            "cabinet", "shelf", "shelves", "drawer", "drawers", "counter", "countertop",
        }
        kept: list[DesktopItem] = []
        for it in items:
            nm = (getattr(it, 'name', None) or '').strip().lower()
            toks = re.split(r"[^a-z0-9]+", nm)
            if any(tok in blocked for tok in toks if tok):
                continue
            kept.append(it)
        return kept

    
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
