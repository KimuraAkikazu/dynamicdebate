"""Agent class (turn-wise history & intent aware)"""
from __future__ import annotations

import re
from collections import deque
from typing import Any, Deque, List, Optional, Sequence

from . import prompts
from .llm_handler import LLMHandler
from .sbd import split_into_sentences  # ファイル先頭の import 群に追加


class Agent:
    def __init__(self, name: str, persona: str, llm_handler: LLMHandler):
        self.name = name
        self.persona = persona
        self.llm_handler = llm_handler

        # runtime state
        self.utterance_queue: Deque[str] = deque()
        self.thought_history: List[tuple[int, str]] = []
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
    def generate_final_answer(self, topic: str, debate_history: str, latest_thoughts: str) -> dict[str, str]:
        return self.llm_handler.generate_final_answer(
            topic,
            self.initial_answer_str,
            debate_history,
            latest_thoughts,
            agent_name=self.name,
            persona=self.persona,
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
        template = (
            prompts.SILENCE_PLAN_PROMPT_TEMPLATE
            if silence
            else prompts.PLAN_ACTION_PROMPT_TEMPLATE
        )
        if turn == 0:
            last_event = "Let's start the discussion now."
        prompt = template.format(
            name = self.name,
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
        intent: str,
        turn: int,
        max_turn: int,
        *,
        peer_names: Sequence[str],
        latest_thoughts: str,
    ) -> None:
        self.utterance_queue.clear()
        utterance_prompt = prompts.GENERATE_UTTERANCE_PROMPT_TEMPLATE.format(
            event_type=event_type,
            topic=topic,
            turn_log=turn_log,
            thought=thought,
            intent=intent,
            turns_left=max_turn - turn,
            name=self.name,
            turn=turn,                    # plan用プロンプトに {turn} を渡す
            initial_answer=self.all_initial_answers_str,
            max_turn=max_turn,
            latest_thoughts=latest_thoughts,  # ★ 自分の最新 thought のみ
        ).strip()

        # generate_utterance は埋め込まれた prompt をそのまま使う
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
            # 後方互換：タプルでない場合はそのまま文字列とみなす
            utterance_text = raw_text = result  # type: ignore

        # ログにはモデルの生出力を保存
        if self.llm_handler.logger:
            self.llm_handler.logger.log_generated(
                agent_name=self.name, turn=turn, full_text=raw_text
            )

        # 発話キューには "utterance" フィールドのみを格納
        self.utterance_queue.extend(self._chunk_utterance(utterance_text))

    # ───────────────────── Chunk utilities ───────────────────── #
    @staticmethod
    def _chunk_utterance(text: str) -> list[str]:
        """
        Use spaCy for sentence boundary detection.
        - “Inc.” や “v.”, “U.S.”, 小数 3.14 などの文中ドットでは基本的に分割されない
        - 末尾の . ? ! などでのみ文を切る
        """
        return split_into_sentences(text)

    def get_next_chunk(self) -> Optional[str]:
        return self.utterance_queue.popleft() if self.utterance_queue else None
