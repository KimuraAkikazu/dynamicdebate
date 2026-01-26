"""Agent class (turn-wise history & purpose aware, round-robin debate)"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import prompts
from .llm_handler import LLMHandler


class Agent:
    def __init__(self, name: str, persona: str, llm_handler: LLMHandler, thought_window: int = 5):
        self.name = name
        self.persona = persona
        self.llm_handler = llm_handler
        self.thought_window = thought_window

        # role
        self.role: str = "normal"  # "normal" or "adversary"
        self.adversary_target: Optional[str] = None  # 例: "A"

        # peers
        self.peer_names: List[str] = []

        # runtime state
        # thought_history: List[(turn, thought, current_answer)]
        self.thought_history: List[Tuple[int, str, str]] = []
        self.initial_answer: Dict[str, str] = {}
        self.initial_answer_str: str = ""
        self.all_initial_answers_str: str = ""  # 全員分

    # ──────────────────── role setters ──────────────────── #
    def set_adversary(self, target_answer: str) -> None:
        self.role = "adversary"
        self.adversary_target = (target_answer or "").strip().upper() or "A"

    def set_peers(self, all_names: Sequence[str]) -> None:
        self.peer_names = [n for n in all_names if n != self.name]

    # ──────────────────── Helper: Get Latest Thought ──────────────────── #
    def _get_latest_thoughts_str(self) -> str:
        """
        thought_history から直近の thought_window 分の thought を取得して文字列で返す。
        履歴がない場合は 'None' を返す。
        Format:
          Turn X: thought...
          Turn Y: thought...
        """
        if not self.thought_history:
            return "None"
        
        # 直近 N 個を取得
        recent = self.thought_history[-self.thought_window:]
        
        lines = []
        for (turn, thought, _) in recent:
            lines.append(f"Turn {turn}: {thought}")
            
        return "\n".join(lines)

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

    def set_initial_answer(self, *, answer: str, reason: str) -> None:
        self.initial_answer = {"answer": answer, "reason": reason}
        self.initial_answer_str = f"answer: {answer}, reason: {reason}"

    # ──────────────────── 最終回答 ──────────────────── #
    def generate_final_answer(
        self,
        topic: str,
        debate_history: str,
    ) -> Dict[str, str]:
        system_prompt = self._build_system_prompt()
        latest_thoughts = self._get_latest_thoughts_str()

        if self.role == "adversary" and self.adversary_target:
            return self.llm_handler.generate_adversary_final_answer(
                topic=topic,
                initial_answer_str=self.all_initial_answers_str,
                debate_history=debate_history,
                target_answer=self.adversary_target,
                latest_thoughts=latest_thoughts,
                system_prompt=system_prompt,
                agent_name=self.name,
            )

        return self.llm_handler.generate_final_answer(
            topic=topic,
            initial_answer_str=self.all_initial_answers_str,
            debate_history=debate_history,
            latest_thoughts=latest_thoughts,
            system_prompt=system_prompt,
            agent_name=self.name,
        )

    # ──────────────────── 発言（スピーカー） ──────────────────── #
    def produce_speech(
        self,
        *,
        topic: str,
        turn_log: str,
        last_event: str,
        turn: int,
        token_budget: int,
        tokens_left: int,
    ) -> str:
        system_prompt = self._build_system_prompt()
        latest_thoughts = self._get_latest_thoughts_str()
        
        if self.role == "adversary" and self.adversary_target:
            utterance, raw_text = self.llm_handler.generate_adversary_speaker_utterance(
                agent_name=self.name,
                system_prompt=system_prompt,
                topic=topic,
                turn_log=turn_log,
                last_event=last_event,
                initial_answers_all=self.all_initial_answers_str,
                turn=turn,
                token_budget=token_budget,
                tokens_left=tokens_left,
                latest_thoughts=latest_thoughts,
            )
        else:
            utterance, raw_text = self.llm_handler.generate_speaker_utterance(
                agent_name=self.name,
                system_prompt=system_prompt,
                topic=topic,
                turn_log=turn_log,
                last_event=last_event,
                initial_answers_all=self.all_initial_answers_str,
                turn=turn,
                token_budget=token_budget,
                tokens_left=tokens_left,
                latest_thoughts=latest_thoughts,
            )

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
        last_event: str,
        turn: int,
        token_budget: int,
        tokens_left: int,
    ) -> Dict[str, Any]:
        system_prompt = self._build_system_prompt()
        latest_thoughts = self._get_latest_thoughts_str()

        thought, current_answer, raw_text = (
            self.llm_handler.generate_listener_thought(
                agent_name=self.name,
                system_prompt=system_prompt,
                topic=topic,
                turn_log=turn_log,
                last_event=last_event,
                initial_answers_all=self.all_initial_answers_str,
                turn=turn,
                token_budget=token_budget,
                tokens_left=tokens_left,
                latest_thoughts=latest_thoughts,
            )
        )

        self.thought_history.append((turn, thought, current_answer))

        if self.llm_handler.logger:
            self.llm_handler.logger.log_generated(
                agent_name=self.name,
                turn=turn,
                full_text=raw_text,
                phase="listener_thought_generated",
            )

        return {
            "thought": thought,
            "answer": current_answer,
        }
