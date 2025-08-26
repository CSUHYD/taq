from __future__ import annotations
import re
from typing import Optional, Dict, Any, List, Callable
from pydantic import BaseModel

from vlmCall_ollama import VLMAPI
from .utils import build_conversation_context


class UserPreferences(BaseModel):
    user_preferences: dict[str, str] | None = None


class AmbiguityCheckResult(BaseModel):
    ambiguous: bool
    reason: str


class ThinkModule:
    """Think module: preference extraction and ambiguity judgement."""

    def __init__(self, vlm_api: VLMAPI, prompt_config: Dict[str, Any]):
        self.vlm_api = vlm_api
        self.prompt_config = prompt_config or {}

    def _extract_preferences(self, user_text: str) -> dict[str, str]:
        prefs: dict[str, str] = {}
        if not user_text:
            return prefs
        txt = user_text.strip()
        # Very light heuristics to capture simple preferences
        # e.g., "I prefer X", "I like Y", "Don't put A on B" -> store negation
        m = re.findall(r"\bI\s+prefer\s+([^.;!]+)", txt, flags=re.IGNORECASE)
        if m:
            prefs["prefer"] = "; ".join(s.strip() for s in m)
        m = re.findall(r"\bI\s+like\s+([^.;!]+)", txt, flags=re.IGNORECASE)
        if m:
            prefs["like"] = "; ".join(s.strip() for s in m)
        m = re.findall(r"\bdo\s*not\s*\b|\bdon't\b", txt, flags=re.IGNORECASE)
        if m:
            prefs["avoid"] = "found negative constraints"
        return prefs

    def extract_user_preferences(
        self,
        user_response: str,
        current_task: str,
        messages_history: Optional[List[Dict[str, Any]]] = None,
    ) -> UserPreferences:
        cfg = self.prompt_config.get("preference_extraction", {}) or {}
        systext = cfg.get("systext", "")
        options = cfg.get("payload_options", {"temperature": 0.2, "num_predict": 160})
        usertext_template = cfg.get("usertext", "")

        context = build_conversation_context(messages_history)
        usertext = usertext_template.format(
            current_task=current_task,
            conversation_context=context,
            latest_user_response=(user_response or ""),
        )

        raw = self.vlm_api.vlm_request_with_format(
            systext=systext,
            usertext=usertext,
            format_schema=UserPreferences.model_json_schema(),
            options=options,
        )
        try:
            parsed = UserPreferences.model_validate_json(raw)
        except Exception:
            parsed = UserPreferences(user_preferences=None)

        # Heuristic augmentation
        heur = self._extract_preferences(user_response)
        if heur:
            base = parsed.user_preferences or {}
            base.update(heur)
            parsed.user_preferences = base
        return parsed

    def ambiguous(
        self,
        user_response: str,
        robot_question: str,
        robot_reasoning: str,
        current_task: str,
        candidate_operation: str | None,
        messages_history: Optional[List[Dict[str, Any]]] = None,
    ) -> AmbiguityCheckResult:
        """Use VLM to determine if current conversation is ambiguous for executing a concrete predicate action.

        Returns AmbiguityCheckResult with yes/no and a short reason.
        """
        cfg = self.prompt_config.get("ambiguity_check", {}) or {}
        systext = cfg.get("systext", (
            "You judge if the conversation and the proposed action have enough information "
            "to execute a concrete predicate-style operation (like place(a,b))."
            " Answer in JSON."
        ))
        options = cfg.get("payload_options", {"temperature": 0.2, "num_predict": 120})
        usertext_template = cfg.get("usertext", (
            "Current task: {current_task}\n\nConversation:\n{conversation_context}\n\nCandidate operation (if any): {candidate_operation}\n\nDecide:\n- ambiguous: true/false (true if more info is required to act)\n- reason: short reason explaining what is missing\nReturn JSON only."
        ))

        # Use full conversation context; template no longer includes separate fields
        context = build_conversation_context(messages_history)
        usertext = usertext_template.format(
            current_task=current_task,
            conversation_context=context,
            candidate_operation=candidate_operation or "",
        )

        raw = self.vlm_api.vlm_request_with_format(
            systext=systext,
            usertext=usertext,
            format_schema=AmbiguityCheckResult.model_json_schema(),
            options=options,
        )
        try:
            return AmbiguityCheckResult.model_validate_json(raw)
        except Exception:
            # Conservative fallback: if we cannot parse, claim ambiguous
            return AmbiguityCheckResult(ambiguous=True, reason="Parsing failure; request clarification.")
