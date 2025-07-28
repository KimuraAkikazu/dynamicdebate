"""Agent class (turn‑wise history & intent aware)"""
from __future__ import annotations

import re
from collections import deque
from typing import Any, Deque, List, Optional, Sequence

from . import prompts
from .llm_handler import LLMHandler

THOUGHT_WINDOW = 10  # not used any more but kept for future tweaks


class Agent:
    def __init__(self, name: str, persona: str, llm_handler: LLMHandler):
        self.name = name
        self.persona = persona
        self.llm_handler = llm_handler
        self.utterance_queue: Deque[str] = deque()
        # keep (turn, thought) tuples
        self.thought_history: List[tuple[int, str]] = []

    # ───────────────────── Action planning ─────────────────────
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
    ) -> dict[str, Any]:
        """Ask LLM for the next‑turn action plan."""
        template = (
            prompts.SILENCE_PLAN_PROMPT_TEMPLATE
            if silence
            else prompts.PLAN_ACTION_PROMPT_TEMPLATE
        )
        prompt = template.format(
            turn_log=turn_log,
            last_event=last_event,
            turns_left=max_turn - turn,
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

        # store thought with turn index
        if isinstance(action_plan, dict) and "thought" in action_plan:
            self.thought_history.append((turn, action_plan["thought"]))

        return action_plan

    # ───────────────────── Prepare utterance ───────────────────
    def decide_to_speak(
        self,
        turn_log: str,
        topic: str,
        thought: str,
        intent: str,
        turn: int,
        max_turn: int,
        *,
        peer_names: Sequence[str],
    ) -> None:
        """Generate utterance, chunk it, and queue."""
        self.utterance_queue.clear()

        utterance_prompt = prompts.GENERATE_UTTERANCE_PROMPT_TEMPLATE.format(
            turn_log=turn_log,
            thought=thought,
            intent=intent,
            turns_left=max_turn - turn,
        ).strip()

        full_text = self.llm_handler.generate_utterance(
            utterance_prompt,
            turn=turn,
            max_turn=max_turn,
            agent_name=self.name,
            persona=self.persona,
            topic=topic,
            peer_names=peer_names,
        )

        if self.llm_handler.logger:
            self.llm_handler.logger.log_generated(
                agent_name=self.name, turn=turn, full_text=full_text
            )

        self.utterance_queue.extend(self._chunk_utterance(full_text))

    # ───────────────────── Chunk utilities ─────────────────────
    @staticmethod
    def _chunk_utterance(text: str) -> List[str]:
        parts = re.split(r"([。！？,.?!])", text)
        chunks, buf = [], ""
        for p in parts:
            if not p:
                continue
            buf += p
            if p in ",.。！？!?":
                chunks.append(buf)
                buf = ""
        if buf:
            chunks.append(buf)
        return chunks

    # ───────────────────── Next chunk ──────────────────────────
    def get_next_chunk(self) -> Optional[str]:
        return self.utterance_queue.popleft() if self.utterance_queue else None
