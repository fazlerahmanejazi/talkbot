from __future__ import annotations
import json, time, traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict

from .logging import RichLogger
from .settings import settings

# --- OpenAI SDK v1.x ---
_openai_client = None

try:
    from openai import OpenAI  # type: ignore
    _openai_client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
except Exception as e:
    print(f"[LLM] OpenAI SDK import failed: {e}")
    _openai_client = None


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: Dict[str, Any]


TOOLS_SPEC: List[ToolDef] = [
    ToolDef(
        name="list_slots",
        description="List currently available appointment slots for the user to choose from.",
        parameters={
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10}},
            "required": [],
        },
    ),
    ToolDef(
        name="book_slot_by_text",
        description=(
            "Book a slot by matching the user's described date/time (day names, times, ranges) "
            "to an available slot. On success, removes the slot from availability."
        ),
        parameters={
            "type": "object",
            "properties": {"request_text": {"type": "string", "description": "user’s wording of the desired slot"}},
            "required": ["request_text"],
        },
    ),
]

SYSTEM_PROMPT = (
    "You are a concise, helpful scheduling assistant in a voice product.\n"
    "- Converse naturally, ask for missing details, then schedule.\n"
    "- If the user asks to see available times, call list_slots.\n"
    "- If the user proposes a time (even partial), call book_slot_by_text.\n"
    "- After list_slots returns, present ALL available times in a natural, conversational way.\n"
    "  Example: 'I have Tuesday at 2 PM, Wednesday at 11 AM, Thursday at 4:30 PM, and Friday at 9 AM'.\n"
    "  Always mention every available time slot, not just 3. Use 'and' for the last item.\n"
    "  Avoid bullet points, dashes, or formatting that sounds robotic when spoken.\n"
    "- If the user mentioned a specific day, only mention times for that day.\n"
    "- If the user confirms with 'yes', book the last suggested time using book_slot_by_text.\n"
    "- Keep spoken replies concise but complete. Use natural, conversational language.\n"
    "- Do NOT expose internal tools; just speak to the user.\n"
)

def _openai_tools_payload() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in TOOLS_SPEC
    ]


class LLM:
    """
    Stateful per-session LLM:
      - Keeps minimal conversation history per session id (sid).
      - Stores assistant tool-call messages and tool results in order.
      - Enables references like “Yes” to resolve to the last proposed time.
    """

    def __init__(self):
        self.model = settings.llm_model
        self.histories: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.max_history = 16  # cap to keep context small
        self.enabled = _openai_client is not None
        
        if settings.llm_log:
            if self.enabled:
                print(f"[{RichLogger._format_time()}] 🤖 LLM: enabled=True model={self.model}")
            else:
                why = "no OPENAI_API_KEY" if not settings.openai_api_key else "SDK not installed"
                print(f"[{RichLogger._format_time()}] 🤖 LLM: enabled=False ({why}) model={self.model}")

    def _ensure_system(self, sid: str):
        if not self.histories[sid] or self.histories[sid][0].get("role") != "system":
            self.histories[sid] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _trim(self, sid: str):
        h = self.histories[sid]
        if len(h) > self.max_history:
            # Keep system + last N-1
            self.histories[sid] = [h[0]] + h[-(self.max_history - 1):]

    # ------------------ Decision turn (user -> say/tool) ------------------
    def decide(self, sid: str, user_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Returns (kind, data)
          - kind == 'say'  → data = {'text': final_reply}
          - kind == 'tool' → data = {'name': str, 'args': dict, 'tool_call_id': str}
        """
        if not self.enabled:
            return "say", {"text": "I'm having trouble connecting. Please try again."}
        
        return self._decide_v1(sid, user_text)

    def _decide_v1(self, sid: str, user_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        try:
            self._ensure_system(sid)
            self.histories[sid].append({"role": "user", "content": user_text})
            if settings.llm_log:
                print(f"[{RichLogger._format_time()}] 🤖 LLM (v1) → user={user_text!r}")

            t0 = time.time()
            resp = _openai_client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model,
                messages=self.histories[sid],
                tools=_openai_tools_payload(),
                tool_choice="auto",
                temperature=0.2,
                max_tokens=80,
            )
            dt = (time.time() - t0) * 1000
            choice = resp.choices[0]
            if settings.llm_log:
                print(f"[{RichLogger._format_time()}] 🤖 LLM (v1) ← finish={choice.finish_reason} dt={dt:.0f}ms")

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                # Save the assistant tool-call message in history
                tool_calls_payload = []
                for tc in choice.message.tool_calls:
                    tool_calls_payload.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    })
                assistant_tool_msg = {"role": "assistant", "tool_calls": tool_calls_payload}
                self.histories[sid].append(assistant_tool_msg)

                # For now we execute only the first tool (single step)
                tc0 = choice.message.tool_calls[0]
                try:
                    args = json.loads(tc0.function.arguments or "{}")
                except Exception:
                    args = {}
                if settings.llm_log:
                    print(f"[LLM] (v1) ← tool={tc0.function.name} args={args}")
                self._trim(sid)
                return "tool", {"name": tc0.function.name, "args": args, "tool_call_id": tc0.id}

            # No tool — speak
            text = (choice.message.content or "").strip() or "Can you share the day and time?"
            if settings.llm_log:
                print(f"[LLM] (v1) ← say={text!r}")
            self.histories[sid].append({"role": "assistant", "content": text})
            self._trim(sid)
            return "say", {"text": text}
        except Exception as e:
            print("[LLM ERROR] decide v1 failed:", repr(e))
            traceback.print_exc()
            self.histories[sid].append({"role": "assistant", "content": "I had trouble thinking. Want to see available times?"})
            self._trim(sid)
            return "say", {"text": "I had trouble thinking. Want to see available times?"}


    # ------------------ Second pass (tool_result -> utterance) ------------------
    def speak_after_tool(
        self,
        sid: str,
        user_text: str,
        tool_name: str,
        tool_call_id: str,
        tool_args: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> str:
        if not self.enabled:
            return "I'm having trouble processing that. Please try again."
        
        return self._speak_after_tool_v1(sid, user_text, tool_name, tool_call_id, tool_args, tool_result)

    def _speak_after_tool_v1(
        self,
        sid: str,
        user_text: str,
        tool_name: str,
        tool_call_id: str,
        tool_args: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> str:
        """
        Correct v1 tool-handshake with stateful history:
          ... assistant(tool_calls=...) -> tool(...) -> assistant(content)
        """
        try:
            self._ensure_system(sid)
            # Append the tool result
            self.histories[sid].append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": json.dumps(tool_result),
            })
            if settings.llm_log:
                preview = json.dumps(tool_result)[:400]
                print(f"[LLM] (v1) → tool_result({tool_name}) {preview}...")

            t0 = time.time()
            resp = _openai_client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model,
                messages=self.histories[sid],
                temperature=0.2,
                max_tokens=80,
            )
            dt = (time.time() - t0) * 1000
            out = (resp.choices[0].message.content or "").strip() or "Okay."
            if settings.llm_log:
                print(f"[LLM] (v1) ← say_after_tool dt={dt:.0f}ms text={out!r}")
            self.histories[sid].append({"role": "assistant", "content": out})
            self._trim(sid)
            return out
        except Exception as e:
            print("[LLM ERROR] speak_after_tool v1 failed:", repr(e))
            traceback.print_exc()
            return "I'm having trouble processing that. Please try again."

