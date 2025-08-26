from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from vlmCall_ollama import VLMAPI
from .utils import build_conversation_context


class AskOutput(BaseModel):
    reasoning: str
    question: str


class AskModule:
    """Ask module that supports multiple questioning strategies.

    Strategies examples: 'direct-querying', 'user-preference-first', 'parallel-exploration'.
    Falls back to 'direct-querying' when strategy is unknown.
    """

    def __init__(self, vlm_api: VLMAPI, prompt_config: Dict[str, Any]):
        self.vlm_api = vlm_api
        self.prompt_config = prompt_config or {}
        self.strategy: str = "direct-querying"

    def set_strategy(self, strategy: str):
        self.strategy = strategy or "direct-querying"

    def ask(
        self,
        task_description: str,
        items_block: str,
        messages_history: Optional[List[Dict[str, Any]]] = None,
        image_path: Optional[str] = None,
        override_strategy: Optional[str] = None,
    ) -> AskOutput:
        cfg = (self.prompt_config.get("question_generation", {}) or {})
        strat_name = override_strategy or self.strategy or "direct-querying"
        strat_cfg = cfg.get(strat_name) or cfg.get("direct-querying") or {}

        systext = strat_cfg.get("systext", "")
        usertext_template = strat_cfg.get("usertext", "")
        options = strat_cfg.get("payload_options", {})

        context = build_conversation_context(messages_history)
        usertext = usertext_template.format(
            task_description=task_description,
            conversation_context=context,
            items_block=items_block or ""
        )

        raw = self.vlm_api.vlm_request_with_format(
            systext=systext,
            usertext=usertext,
            format_schema=AskOutput.model_json_schema(),
            image_path1=image_path,
            options=options,
        )
        try:
            return AskOutput.model_validate_json(raw)
        except Exception:
            # If model didn't follow schema, wrap raw as question
            return AskOutput(reasoning="", question=str(raw))

