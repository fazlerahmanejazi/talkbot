from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

from .llm_manager import LLMManager
from .logging import RichLogger
from .schedule_tool import list_slots, book_slot

_llm_manager = LLMManager()

def plan(last_user: str, ctx: Optional[Dict[str, Any]] = None) -> Tuple[str, str, float]:
    """
    LLM-first planner with per-session memory (sid).
    Returns (reply_text, channel, confidence).
    """
    sid = str((ctx or {}).get("sid", "default"))
    
    print(f"[{RichLogger._format_time()}] {RichLogger.planning_start(last_user)} [sid={sid[:8]}]")

    kind, data = _llm_manager.decide(sid, last_user)
    if kind == "say":
        print(f"[{RichLogger._format_time()}] {RichLogger.planning_response(data['text'], 'voice', 0.90)}")
        return data["text"], "voice", 0.9

    tool_name: str = data["name"]
    tool_args: Dict[str, Any] = data.get("args", {})
    tool_call_id: str = data.get("tool_call_id", "tool")
    
    print(f"[{RichLogger._format_time()}] {RichLogger.planning_tool(tool_name, tool_args)}")

    if tool_name == "list_slots":
        limit = int(tool_args.get("limit", 6))
        slots = [s.raw for s in list_slots(limit=limit)]
        tool_result = {"slots": slots}
        reply = _llm_manager.speak_after_tool(sid, last_user, tool_name, tool_call_id, tool_args, tool_result)
        print(f"[{RichLogger._format_time()}] 📅 Calendar Query: {len(slots)} slots found -> '{reply}' [calendar] (0.96)")
        return reply, "calendar", 0.96

    if tool_name == "book_slot_by_text":
        req = str(tool_args.get("request_text", last_user))
        ok, msg = book_slot(req)
        tool_result = {"ok": ok, "message": msg}
        reply = _llm_manager.speak_after_tool(sid, last_user, tool_name, tool_call_id, tool_args, tool_result)
        status = "✅" if ok else "❌"
        print(f"[{RichLogger._format_time()}] {status} Booking: '{req}' -> '{reply}' [calendar] ({0.99 if ok else 0.8})")
        return reply, "calendar", (0.99 if ok else 0.8)

    print(f"[{RichLogger._format_time()}] ❓ Unknown Tool: {tool_name}")
    return "I can show times or book one. Which do you prefer?", "voice", 0.7
