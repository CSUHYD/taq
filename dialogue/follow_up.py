from typing import Optional, Dict, Any


class FollowUpModule:
    """Generate follow-up questions based on ambiguity reasons and context."""

    def suggest(
        self,
        reason: str | None,
        last_robot_question: str | None,
        conversation_context: str | None,
    ) -> str:
        r = (reason or "").strip()
        if not r:
            # Generic fallback
            return "Could you clarify which item and where to place it?"

        # Minimal templated follow-ups to avoid extra VLM calls
        if "which item" in r.lower() or "item" in r.lower():
            return (
                "Which specific item do you mean? If possible, describe its color, "
                "size, or position on the desk."
            )
        if "where" in r.lower() or "location" in r.lower():
            return "Where should I place it? Please specify the target location."
        if "how" in r.lower() or "method" in r.lower():
            return "How would you like me to handle it?"

        # Default: prepend a short clarification cue
        return f"To clarify: {r}"

