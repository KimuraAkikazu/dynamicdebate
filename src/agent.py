"""エージェントクラス"""
from __future__ import annotations

import re
from collections import deque
from typing import Any, Deque, List, Optional

from . import prompts
from .llm_handler import LLMHandler

THOUGHT_WINDOW = 20  # 思考履歴を何件まで渡すか


class Agent:
    def __init__(self, name: str, persona: str, llm_handler: LLMHandler):
        self.name = name
        self.persona = persona
        self.llm_handler = llm_handler
        self.utterance_queue: Deque[str] = deque()
        self.thought_history: List[str] = []

    # ---------------- 行動計画 ---------------- #
    def plan_action(
        self,
        history: str,
        last_event: str,
        topic: str,
        turn: int,
        max_turn: int,
        *,
        silence: bool = False,
    ) -> dict[str, Any]:
        recent_thoughts = "\n".join(self.thought_history[-THOUGHT_WINDOW:])
        template = (
            prompts.SILENCE_PLAN_PROMPT_TEMPLATE
            if silence
            else prompts.PLAN_ACTION_PROMPT_TEMPLATE
        )
        prompt = template.format(
            persona=self.persona,
            topic=topic,
            history=history,
            last_event=last_event,
            thought_history=recent_thoughts,
            turns_left=max_turn - turn,
        )

        action_plan = self.llm_handler.generate_action(
            prompt, turn, max_turn, agent_name=self.name
        )

        # 思考履歴に追加
        if isinstance(action_plan, dict) and "thought" in action_plan:
            self.thought_history.append(action_plan["thought"])

        return action_plan

    # ---------------- 発言準備 ---------------- #
    def decide_to_speak(
        self,
        history: str,
        topic: str,
        thought: str,
        turn: int,
        max_turn: int,
    ) -> None:
        self.utterance_queue.clear()

        utterance_prompt = prompts.GENERATE_UTTERANCE_PROMPT_TEMPLATE.format(
            persona=self.persona,
            topic=topic,
            history=history,
            thought=thought,
            turn=turn,
            max_turn=max_turn,
            turns_left=max_turn - turn,
        ).strip()

        full_text = self.llm_handler.generate_utterance(
            utterance_prompt, turn, max_turn, agent_name=self.name
        )

        if self.llm_handler.logger:
            self.llm_handler.logger.log_generated(
                agent_name=self.name,
                turn=turn,
                full_text=full_text,
            )

        self.utterance_queue.extend(self._chunk_utterance(full_text))

    # ---------------- チャンク化 ---------------- #
    @staticmethod
    def _chunk_utterance(text: str) -> List[str]:
        parts = re.split(r"([、。！？])", text)
        chunks, buf = [], ""
        for p in parts:
            if not p:
                continue
            buf += p
            if p in "、。！？":
                chunks.append(buf)
                buf = ""
        if buf:
            chunks.append(buf)
        return chunks

    # ---------------- チャンク取得 ---------------- #
    def get_next_chunk(self) -> Optional[str]:
        return self.utterance_queue.popleft() if self.utterance_queue else None