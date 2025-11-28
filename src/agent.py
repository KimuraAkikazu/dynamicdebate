"""Agent class (turn-wise history & purpose aware)"""
from __future__ import annotations

import re
from collections import deque
from typing import Any, Deque, List, Optional, Sequence

from . import prompts
from .llm_handler import LLMHandler


class Agent:
    def __init__(self, name: str, persona: str, llm_handler: LLMHandler):
        self.name = name
        self.persona = persona
        self.llm_handler = llm_handler

        # role
        self.role: str = "normal"  # "normal" or "adversary"
        self.adversary_target: Optional[str] = None  # e.g., "A"

        # runtime state
        self.utterance_queue: Deque[str] = deque()
        self.thought_history: List[tuple[int, str]] = []
        self.initial_answer: dict[str, str] = {}
        self.initial_answer_str: str = ""
        self.all_initial_answers_str: str = ""  # 全員分

    # ---- role setters -------------------------------------------------
    def set_adversary(self, target_answer: str) -> None:
        self.role = "adversary"
        self.adversary_target = (target_answer or "").strip().upper() or "A"

    # ──────────────────── 初回回答 ──────────────────── #
    def generate_initial_answer(self, topic: str, max_turn: int, peer_names: Sequence[str]) -> None:
        if self.role == "adversary" and self.adversary_target:
            self.initial_answer = self.llm_handler.generate_adversary_initial_answer(
                topic, target_answer=self.adversary_target, agent_name=self.name, persona=self.persona, max_turn=max_turn, peer_names=peer_names
            )
        else:
            self.initial_answer = self.llm_handler.generate_initial_answer(
                topic, agent_name=self.name, persona=self.persona, max_turn=max_turn, peer_names=peer_names
            )
        self.initial_answer_str = (
            f"answer: {self.initial_answer.get('answer','')}, "
            f"reason: {self.initial_answer.get('reason','')}"
        )

    # ──────────────────── 最終回答 ──────────────────── #
    def generate_final_answer(self, topic: str, debate_history: str, latest_thoughts: str, max_turn: int, peer_names: Sequence[str]) -> dict[str, str]:
        if self.role == "adversary" and self.adversary_target:
            return self.llm_handler.generate_adversary_final_answer(
                topic,
                self.all_initial_answers_str,
                debate_history,
                latest_thoughts,
                target_answer=self.adversary_target,
                agent_name=self.name,
                persona=self.persona,
                max_turn=max_turn,
                peer_names=peer_names,
            )
        return self.llm_handler.generate_final_answer(
            topic,
            self.all_initial_answers_str,
            debate_history,
            latest_thoughts,
            agent_name=self.name,
            persona=self.persona,
            max_turn=max_turn,
            peer_names=peer_names,
        )

    # ───────────────────── Action planning ───────────────────── #
    def plan_action(
        self,
        turn_log: str,
        last_event: str,
        topic: str,
        turn: int,
        max_turn: int,
        *,
        silence: bool,
        peer_names: Sequence[str],
        latest_thoughts: str,
    ) -> dict[str, Any]:
        if self.role == "adversary" and self.adversary_target:
            template = (
                prompts.ADVERSARY_SILENCE_PLAN_PROMPT_TEMPLATE
                if silence
                else prompts.ADVERSARY_PLAN_ACTION_PROMPT_TEMPLATE
            )
            if turn == 0:
                last_event = "Let's start the discussion now."
            prompt = template.format(
                name=self.name,
                turn_log=turn_log,
                last_event=last_event,
                turns_left=max_turn - turn,
                max_turn=max_turn,
                turn=turn,
                initial_answer=self.all_initial_answers_str,
                topic=topic,
                latest_thoughts=latest_thoughts,
                target_answer=self.adversary_target,
            )
            action_plan = self.llm_handler.generate_action_adversary(
                prompt,
                turn=turn,
                max_turn=max_turn,
                agent_name=self.name,
                persona=self.persona,
                topic=topic,
                peer_names=peer_names,
                target_answer=self.adversary_target,
            )
        else:
            template = (
                prompts.SILENCE_PLAN_PROMPT_TEMPLATE
                if silence
                else prompts.PLAN_ACTION_PROMPT_TEMPLATE
            )
            if turn == 0:
                last_event = "Let's start the discussion now."
            prompt = template.format(
                name=self.name,
                turn_log=turn_log,
                last_event=last_event,
                turns_left=max_turn - turn,
                max_turn=max_turn,
                turn=turn,                    # plan用プロンプトに {turn} を渡す
                initial_answer=self.all_initial_answers_str,
                topic=topic,
                latest_thoughts=latest_thoughts,  # ★ 自分の最新 thought のみ
            )
            action_plan = self.llm_handler.generate_action(
                prompt,
                turn=turn,
                max_turn=max_turn,
                agent_name=self.name,
                persona=self.persona,
                topic=topic,
                peer_names=peer_names,
            )

        if isinstance(action_plan, dict) and "thought" in action_plan:
            self.thought_history.append((turn, action_plan["thought"]))
        return action_plan

    # ───────────────────── Prepare utterance ─────────────────── #
    def decide_to_speak(
        self,
        event_type: str,
        turn_log: str,
        topic: str,
        thought: str,
        purpose: str,
        turn: int,
        max_turn: int,
        *,
        peer_names: Sequence[str],
        latest_thoughts: str,
    ) -> None:
        self.utterance_queue.clear()

        if self.role == "adversary" and self.adversary_target:
            utterance_prompt = prompts.ADVERSARY_GENERATE_UTTERANCE_PROMPT_TEMPLATE.format(
                event_type=event_type,
                topic=topic,
                turn_log=turn_log,
                thought=thought,
                purpose=purpose,
                turns_left=max_turn - turn,
                name=self.name,
                turn=turn,
                initial_answer=self.all_initial_answers_str,
                max_turn=max_turn,
                latest_thoughts=latest_thoughts,
                target_answer=self.adversary_target,
            ).strip()
            result = self.llm_handler.generate_utterance_adversary(
                utterance_prompt,
                turn=turn,
                max_turn=max_turn,
                agent_name=self.name,
                persona=self.persona,
                topic=topic,
                peer_names=peer_names,
                target_answer=self.adversary_target,
            )
        else:
            utterance_prompt = prompts.GENERATE_UTTERANCE_PROMPT_TEMPLATE.format(
                event_type=event_type,
                topic=topic,
                turn_log=turn_log,
                thought=thought,
                purpose=purpose,
                turns_left=max_turn - turn,
                name=self.name,
                turn=turn,                    # plan用プロンプトに {turn} を渡す
                initial_answer=self.all_initial_answers_str,
                max_turn=max_turn,
                latest_thoughts=latest_thoughts,  # ★ 自分の最新 thought のみ
            ).strip()
            result = self.llm_handler.generate_utterance(
                utterance_prompt,
                turn=turn,
                max_turn=max_turn,
                agent_name=self.name,
                persona=self.persona,
                topic=topic,
                peer_names=peer_names,
            )

        if isinstance(result, tuple):
            utterance_text, raw_text = result
        else:
            utterance_text = raw_text = result  # type: ignore

        # ログにはモデルの生出力を保存
        if self.llm_handler.logger:
            self.llm_handler.logger.log_generated(
                agent_name=self.name, turn=turn, full_text=raw_text
            )

        # 発話キューには 8トークン単位のチャンクを格納（sbd 文分割→置換）
        self.utterance_queue.extend(self._chunk_utterance(utterance_text))

    # ───────────────────── Chunk utilities ───────────────────── #
    def _chunk_utterance(self, text: str) -> list[str]:
        """
        仕様変更：文単位ではなく **8トークン** 単位でチャンク化。
        モデルの tokenizer/detokenizer により厳密なトークン境界を使用。
        """
        return self.llm_handler.chunk_by_tokens(text)
    
    def get_next_chunk(self) -> Optional[str]:
        return self.utterance_queue.popleft() if self.utterance_queue else None
