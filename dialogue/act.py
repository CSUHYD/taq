import re
from typing import Optional, Dict, Any
from pydantic import BaseModel


class ActPlan(BaseModel):
    sources: list[str]  # objects to act on (IDs or names)
    action: str         # verb/predicate
    target: Optional[str] = None  # location or destination object


class ActModule:
    """Turn operation text into a structured action plan.

    Accepts predicate style operations like place(cup, shelf) or move(#itm-1-cup, drawer)
    and also tries simple natural language patterns like "put the cup on the shelf".
    """

    _func_call = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_-]*)\s*\((.*)\)\s*$")

    def plan_from_operation(self, operation_text: str | None) -> Optional[ActPlan]:
        if not operation_text:
            return None
        op = operation_text.strip()
        if not op:
            return None

        # 1) predicate-style: verb(arg1, arg2)
        m = self._func_call.match(op)
        if m:
            verb = m.group(1)
            args = [a.strip() for a in m.group(2).split(',') if a.strip()]
            sources: list[str] = []
            target: str | None = None
            if len(args) >= 1:
                sources = [args[0]]
            if len(args) >= 2:
                target = args[1]
            return ActPlan(sources=sources, action=verb, target=target)

        # 2) basic NL patterns
        low = op.lower()
        # put X on Y / place X on Y / move X to Y
        for verb, preposition in (
            ("put", " on "),
            ("place", " on "),
            ("move", " to "),
        ):
            if low.startswith(verb + " ") and preposition in low:
                # naive split: verb X preposition Y
                try:
                    after_verb = op[len(verb):].strip()
                    parts = after_verb.split(preposition.strip(), 1)
                    if len(parts) == 2:
                        src, tgt = parts[0].strip(), parts[1].strip()
                        return ActPlan(sources=[src], action=verb, target=tgt)
                except Exception:
                    pass

        # Fallback: unknown format, expose as raw action verb 'do'
        return ActPlan(sources=[op], action="do", target=None)

