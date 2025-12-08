"""Agent class (turn-wise history & purpose aware, round-robin debate)"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import prompts
from .llm_handler import LLMHandler


class Agent:
    def __init__(self, name: str, persona: str, llm_handler: LLMHandler):
        self.name = name
        self.persona = persona
        self.llm_handler = llm_handler

        # role
        self.role: str = "normal"  # "normal" or "adversary"
        self.adversary_target: Optional[str] = None  # 例: "A"

        # peers
        self.peer_names: List[str] = []

        # runtime state
        # thought_history: List[(turn, thought, current_answer, consensus)]
        self.thought_history: List[Tuple[int, str, str, bool]] = []
        self.initial_answer: Dict[str, str] = {}
        self.initial_answer_str: str = ""
        self.all_initial_answers_str: str = ""  # 全員分

    # ──────────────────── role setters ──────────────────── #
    def set_adversary(self, target_answer: str) -> None:
        self.role = "adversary"
        self.adversary_target = (target_answer or "").strip().upper() or "A"

    def set_peers(self, all_names: Sequence[str]) -> None:
        self.peer_names = [n for n in all_names if n != self.name]

    # ──────────────────── System Prompt ──────────────────── #
    def _build_system_prompt(self) -> str:
        peer1 = self.peer_names[0] if len(self.peer_names) > 0 else "Peer1"
        peer2 = self.peer_names[1] if len(self.peer_names) > 1 else "Peer2"
        return prompts.SYSTEM_PROMPT.format(
            name=self.name,
            peer1=peer1,
            peer2=peer2,
        )

    # ──────────────────── 初回回答 ──────────────────── #
    def generate_initial_answer(
        self,
        topic: str,
        max_turn: int,
        peer_names: Sequence[str],
    ) -> None:
        # peer_names をセット（念のため）
        self.set_peers(peer_names)
        system_prompt = self._build_system_prompt()

        if self.role == "adversary" and self.adversary_target:
            self.initial_answer = self.llm_handler.generate_adversary_initial_answer(
                topic=topic,
                target_answer=self.adversary_target,
                system_prompt=system_prompt,
                agent_name=self.name,
            )
        else:
            self.initial_answer = self.llm_handler.generate_initial_answer(
                topic=topic,
                system_prompt=system_prompt,
                agent_name=self.name,
            )

        self.initial_answer_str = (
            f"answer: {self.initial_answer.get('answer', '')}, "
            f"reason: {self.initial_answer.get('reason', '')}"
        )

    # ──────────────────── 最終回答 ──────────────────── #
    def generate_final_answer(
        self,
        topic: str,
        debate_history: str,
    ) -> Dict[str, str]:
        system_prompt = self._build_system_prompt()

        if self.role == "adversary" and self.adversary_target:
            return self.llm_handler.generate_adversary_final_answer(
                topic=topic,
                initial_answer_str=self.initial_answer_str,
                debate_history=debate_history,
                target_answer=self.adversary_target,
                system_prompt=system_prompt,
                agent_name=self.name,
            )

        return self.llm_handler.generate_final_answer(
            topic=topic,
            initial_answer_str=self.initial_answer_str,
            debate_history=debate_history,
            system_prompt=system_prompt,
            agent_name=self.name,
        )

    # ──────────────────── 発言（スピーカー） ──────────────────── #
    def produce_speech(
        self,
        *,
        topic: str,
        turn_log: str,
        turn: int,
        turns_left_for_agent: int,
        max_turn: int,
    ) -> str:
        system_prompt = self._build_system_prompt()
        
        if self.role == "adversary" and self.adversary_target:
            utterance, raw_text = self.llm_handler.generate_adversary_speaker_utterance(
                agent_name=self.name,
                system_prompt=system_prompt,
                topic=topic,
                turn_log=turn_log,
                initial_answers_all=self.all_initial_answers_str,
                turn=turn,
                turns_left_for_agent=turns_left_for_agent,
                max_turn=max_turn,
            )
        else:
            utterance, raw_text = self.llm_handler.generate_speaker_utterance(
                agent_name=self.name,
                system_prompt=system_prompt,
                topic=topic,
                turn_log=turn_log,
                initial_answers_all=self.all_initial_answers_str,
                turn=turn,
                turns_left_for_agent=turns_left_for_agent,
                max_turn=max_turn,
            )

        # thought_history は listener のターンで更新されるのでここでは触らない
        if self.llm_handler.logger:
            self.llm_handler.logger.log_generated(
                agent_name=self.name,
                turn=turn,
                full_text=raw_text,
                phase="speaker_utterance_generated",
            )

        return utterance

    # ──────────────────── リスナー（thought のみ） ──────────────────── #
    def think_only(
        self,
        *,
        topic: str,
        turn_log: str,
        turn: int,
        max_turn: int,
    ) -> Dict[str, Any]:
        system_prompt = self._build_system_prompt()
        thought, current_answer, consensus, raw_text = (
            self.llm_handler.generate_listener_thought(
                agent_name=self.name,
                system_prompt=system_prompt,
                topic=topic,
                turn_log=turn_log,
                initial_answers_all=self.all_initial_answers_str,
                turn=turn,
                max_turn=max_turn,
            )
        )

        self.thought_history.append((turn, thought, current_answer, consensus))

        if self.llm_handler.logger:
            self.llm_handler.logger.log_generated(
                agent_name=self.name,
                turn=turn,
                full_text=raw_text,
                phase="listener_thought_generated",
            )

        return {
            "thought": thought,
            "current_answer": current_answer,
            "consensus": consensus,
        }
