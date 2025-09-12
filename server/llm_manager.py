"""
Simplified LLM management system.
Handles OpenAI LLM backend only.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from .llm import LLM as OpenAILLM
from .settings import settings
from .logging import RichLogger


class LLMManager:
    """
    Simplified LLM manager that handles OpenAI LLM only.
    """
    
    def __init__(self):
        self.openai_llm = OpenAILLM()
        
        if settings.llm_log:
            self._log_initialization()
    
    def _log_initialization(self):
        """Log the initialization status."""
        if self.openai_llm.enabled:
            print(f"[{RichLogger._format_time()}] 🤖 LLM Manager: ✅ Ready model={settings.llm_model}")
        else:
            print(f"[{RichLogger._format_time()}] 🤖 LLM Manager: ❌ OpenAI not available")
    
    def is_available(self) -> bool:
        """Check if LLM backend is available."""
        return self.openai_llm.enabled
    
    def decide(self, sid: str, user_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Make a decision using the OpenAI LLM backend."""
        return self.openai_llm.decide(sid, user_text)
    
    def speak_after_tool(
        self,
        sid: str,
        user_text: str,
        tool_name: str,
        tool_call_id: str,
        tool_args: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> str:
        """Generate response after tool execution."""
        return self.openai_llm.speak_after_tool(sid, user_text, tool_name, tool_call_id, tool_args, tool_result)
    
