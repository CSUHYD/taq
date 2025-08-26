from typing import List, Dict, Any


def build_conversation_context(messages_history: List[Dict[str, Any]] | None) -> str:
    """Build a linearized conversation context string for prompting.

    Accepts a list of message dicts from the web layer with keys such as
    type: 'robot' | 'user'
    For robots, may include 'reasoning' and 'question'; for users, 'response'.
    """
    conversation_context = ""
    if not messages_history:
        return conversation_context

    for msg in messages_history:
        mtype = msg.get("type")
        if mtype == "robot":
            if msg.get("reasoning"):
                conversation_context += f"Robot reasoning: {msg['reasoning']}\n"
            if msg.get("question"):
                conversation_context += f"Robot question: {msg['question']}\n"
        elif mtype == "user":
            if msg.get("response"):
                conversation_context += f"User response: {msg['response']}\n"
    if conversation_context:
        conversation_context += "\n"
    return conversation_context


def build_conversation_context_excluding_last_robot(
    messages_history: List[Dict[str, Any]] | None,
) -> str:
    """Build context but drop the most recent 'robot' message to avoid redundancy
    when the last robot question/reasoning is provided separately in the prompt.
    """
    if not messages_history:
        return ""
    idx_last_robot = -1
    for i in range(len(messages_history) - 1, -1, -1):
        if messages_history[i].get("type") == "robot":
            idx_last_robot = i
            break
    trimmed = messages_history[:idx_last_robot] if idx_last_robot >= 0 else messages_history
    return build_conversation_context(trimmed)

