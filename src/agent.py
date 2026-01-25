# src/agent.py
"""Agent class (turn-wise history & purpose aware)"""
from __future__ import annotations

import re
from collections import deque
from typing import Any, Deque, List, Optional, Sequence, Tuple, Dict

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

    def reset_role(self) -> None:
        self.role = "normal"
        self.adversary_target = None

    # ──────────────────── 初回回答 ──────────────────── #
    def generate_initial_answer(self, topic: str, max_turn: int, peer_names: Sequence[str]) -> Dict[str, int]:
        """
        戻り値: usage dict
        """
        if self.role == "adversary" and self.adversary_target:
            self.initial_answer, usage = self.llm_handler.generate_adversary_initial_answer(
                topic, target_answer=self.adversary_target, agent_name=self.name, persona=self.persona, max_turn=max_turn, peer_names=peer_names
            )
        else:
            self.initial_answer, usage = self.llm_handler.generate_initial_answer(
                topic, agent_name=self.name, persona=self.persona, max_turn=max_turn, peer_names=peer_names
            )
        self.initial_answer_str = (
            f"answer: {self.initial_answer.get('answer','')}, "
            f"reason: {self.initial_answer.get('reason','')}"
        )
        return usage

    def set_initial_answer(self, *, answer: str, reason: str) -> None:
        self.initial_answer = {"answer": answer, "reason": reason}
        self.initial_answer_str = f"answer: {answer}, reason: {reason}"

    # ──────────────────── 最終回答 ──────────────────── #
    def generate_final_answer(self, topic: str, debate_history: str, latest_thoughts: str, max_turn: int, peer_names: Sequence[str]) -> Tuple[dict[str, str], Dict[str, int]]:
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
        allow_interruption: bool = True,
        token_budget: int | None = None,
        tokens_left: int | None = None,
    ) -> Tuple[dict[str, Any], Dict[str, int]]:
        
        # 敵対者ロジック
        if self.role == "adversary" and self.adversary_target:
            if not allow_interruption:
                # 割り込みなしモード用 (Adversary)
                template = prompts.ADVERSARY_PLAN_ACTION_NO_INTERRUPT_PROMPT_TEMPLATE
            else:
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
                turn=turn,
                initial_answer=self.all_initial_answers_str,
                topic=topic,
                latest_thoughts=latest_thoughts,
                token_budget=token_budget,
                tokens_left=tokens_left,
                target_answer=self.adversary_target,
            )
            action_plan, usage = self.llm_handler.generate_action_adversary(
                prompt,
                turn=turn,
                max_turn=max_turn,
                agent_name=self.name,
                persona=self.persona,
                topic=topic,
                peer_names=peer_names,
                target_answer=self.adversary_target,
            )
        
        # 通常エージェントロジック
        else:
            if not allow_interruption:
                # 割り込みなし専用プロンプト (Normal)
                template = prompts.PLAN_ACTION_NO_INTERRUPT_PROMPT_TEMPLATE
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
                turn=turn,
                initial_answer=self.all_initial_answers_str,
                topic=topic,
                latest_thoughts=latest_thoughts,
                token_budget=token_budget,
                tokens_left=tokens_left,
            )
            action_plan, usage = self.llm_handler.generate_action(
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
        return action_plan, usage

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
        token_budget: int | None = None,
        tokens_left: int | None = None,
    ) -> Dict[str, int]:
        """
        戻り値: usage dict
        """
        self.utterance_queue.clear()

        if self.role == "adversary" and self.adversary_target:
            utterance_prompt = prompts.ADVERSARY_GENERATE_UTTERANCE_PROMPT_TEMPLATE.format(
                event_type=event_type,
                topic=topic,
                turn_log=turn_log,
                thought=thought,
                purpose=purpose,
                name=self.name,
                turn=turn,
                initial_answer=self.all_initial_answers_str,
                latest_thoughts=latest_thoughts,
                target_answer=self.adversary_target,
                token_budget=token_budget,
                tokens_left=tokens_left,
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
                tokens_left=tokens_left,
            )
        else:
            utterance_prompt = prompts.GENERATE_UTTERANCE_PROMPT_TEMPLATE.format(
                event_type=event_type,
                topic=topic,
                turn_log=turn_log,
                thought=thought,
                purpose=purpose,
                name=self.name,
                turn=turn,
                initial_answer=self.all_initial_answers_str,
                latest_thoughts=latest_thoughts,
                token_budget=token_budget,
                tokens_left=tokens_left,
            ).strip()
            result = self.llm_handler.generate_utterance(
                utterance_prompt,
                turn=turn,
                max_turn=max_turn,
                agent_name=self.name,
                persona=self.persona,
                topic=topic,
                peer_names=peer_names,
                tokens_left=tokens_left,
            )

        # result is always (utterance_text, raw_text, usage)
        utterance_text, raw_text, usage = result

        if self.llm_handler.logger:
            self.llm_handler.logger.log_generated(
                agent_name=self.name, turn=turn, full_text=raw_text, token_stats=usage
            )

        self.utterance_queue.extend(self._chunk_utterance(utterance_text))
        return usage

    # ───────────────────── Chunk utilities ───────────────────── #
    def _chunk_utterance(self, text: str) -> list[str]:
        """
        仕様変更：文単位ではなく **8トークン** 単位でチャンク化。
        モデルの tokenizer/detokenizer により厳密なトークン境界を使用。
        """
        return self.llm_handler.chunk_by_tokens(text)
    
    def get_next_chunk(self) -> Optional[str]:
        return self.utterance_queue.popleft() if self.utterance_queue else None
