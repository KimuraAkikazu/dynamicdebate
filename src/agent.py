"""Agent class for fixed-order ablation (no interrupts, no action-planning)."""
from __future__ import annotations

from typing import Any, List, Tuple

from .llm_handler import LLMHandler


class Agent:
    def __init__(self, name: str, persona: str, llm_handler: LLMHandler):
        self.name = name
        self.persona = persona
        self.llm_handler = llm_handler

        # runtime state
        self.thought_history: List[Tuple[int, str]] = []
        self.initial_answer: dict[str, str] = {}
        self.initial_answer_str: str = ""
        self.all_initial_answers_str: str = ""  # 全員分

    # ──────────────────── 初回回答 ──────────────────── #
    def generate_initial_answer(self, topic: str) -> None:
        self.initial_answer = self.llm_handler.generate_initial_answer(
            topic, agent_name=self.name, persona=self.persona
        )
        self.initial_answer_str = (
            f"Answer: {self.initial_answer.get('answer','')}, "
            f"Reason: {self.initial_answer.get('reason','')}"
        )

    # ──────────────────── 最終回答 ──────────────────── #
    def generate_final_answer(self, topic: str, debate_history: str) -> dict[str, str]:
        return self.llm_handler.generate_final_answer(
            topic,
            self.initial_answer_str,
            debate_history,
            agent_name=self.name,
            persona=self.persona,
        )

    # ──────────────────── 発言（SPEAKER） ──────────────────── #
    def produce_speech(
        self,
        *,
        topic: str,
        turn_log: str,
        turn: int,
        turns_left_for_agent: int,
        max_turn: int,
    ) -> str:
        utterance, raw = self.llm_handler.generate_speaker_utterance(
            agent_name=self.name,
            persona=self.persona,
            topic=topic,
            turn_log=turn_log,
            initial_answers_all=self.all_initial_answers_str,
            turn=turn,
            turns_left_for_agent=turns_left_for_agent,
            max_turn=max_turn,
        )
        # ログ（モデル生出力）
        if self.llm_handler.logger:
            self.llm_handler.logger.log_generated(
                agent_name=self.name, turn=turn, full_text=raw, phase="speaker_generated"
            )
        return utterance

    # ──────────────────── 思考（LISTENER） ──────────────────── #
    def think_only(
        self, *, topic: str, turn_log: str, turn: int, max_turn: int
    ) -> str:
        thought, raw = self.llm_handler.generate_listener_thought(
            agent_name=self.name,
            persona=self.persona,
            topic=topic,
            turn_log=turn_log,
            initial_answers_all=self.all_initial_answers_str,
            turn=turn,
            max_turn=max_turn,
        )
        self.thought_history.append((turn, thought))
        if self.llm_handler.logger:
            self.llm_handler.logger.log_generated(
                agent_name=self.name, turn=turn, full_text=raw, phase="listener_generated"
            )
        return thought
