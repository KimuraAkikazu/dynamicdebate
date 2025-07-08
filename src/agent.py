"""エージェントクラス"""
from __future__ import annotations

import re
from collections import deque
from typing import Any, Deque, List, Optional

from . import prompts
from .llm_handler import LLMHandler


class Agent:
    def __init__(self, name: str, persona: str, llm_handler: LLMHandler):
        self.name = name
        self.persona = persona
        self.llm_handler = llm_handler
        self.utterance_queue: Deque[str] = deque()

    # ---------------- 行動計画 ---------------- #
    def plan_action(
        self,
        history: str,
        last_event: str,
        topic: str,
        turn: int,
        max_turn: int,
    ) -> dict[str, Any]:
        prompt = prompts.PLAN_ACTION_PROMPT_TEMPLATE.format(
            persona=self.persona,
            topic=topic,
            history=history,
            last_event=last_event,
        )
        return self.llm_handler.generate_action(prompt, turn, max_turn)

    # ---------------- 発言準備 ---------------- #
    def decide_to_speak(
        self,
        history: str,
        topic: str,
        thought: str,
        turn: int,
        max_turn: int,
    ) -> None:
        utterance_prompt = prompts.GENERATE_UTTERANCE_PROMPT_TEMPLATE.format(
            persona=self.persona,
            topic=topic,
            history=history,
            thought=thought,
        )
        full_text = self.llm_handler.generate_utterance(utterance_prompt, turn, max_turn)
        self.utterance_queue.extend(self._chunk_utterance(full_text))

    # ---------------- チャンク化 ---------------- #
    @staticmethod
    def _chunk_utterance(text: str) -> List[str]:
        """
        「。」・「！」・「？」・「、」で分割し、
        区切り記号は直前のチャンクの末尾に残す。
        """
        import re

        parts = re.split(r"([。！？、])", text)  # ← 「、」も対象に追加
        chunks, buf = [], ""
        for p in parts:
            if not p:
                continue
            buf += p
            if p in "。！？、":
                chunks.append(buf)
                buf = ""
        if buf:
            chunks.append(buf)
        return chunks
    
    # ---------------- チャンク取得 ---------------- #
    def get_next_chunk(self) -> Optional[str]:
        return self.utterance_queue.popleft() if self.utterance_queue else None
