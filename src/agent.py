"""Agent class for fixed-order ablation (System Prompt Aware)."""
from __future__ import annotations

from typing import Any, List, Tuple, Dict
from .llm_handler import LLMHandler
from . import prompts

class Agent:
    def __init__(self, name: str, persona: str, llm_handler: LLMHandler):
        self.name = name
        self.persona = persona
        self.llm_handler = llm_handler

        # runtime state
        self.thought_history: List[Tuple[int, str, str, bool]] = []
        self.initial_answer: dict[str, str] = {}
        self.initial_answer_str: str = ""
        self.all_initial_answers_str: str = ""
        
        # System Prompt 構築用
        self.peers: List[str] = [] 

    def set_peers(self, all_agent_names: List[str]) -> None:
        """マネージャーから呼ばれ、自分以外のエージェント名を記録する"""
        self.peers = [n for n in all_agent_names if n != self.name]

    def _get_system_prompt(self) -> str:
        """
        prompts.SYSTEM_PROMPT に対して {name}, {peer1}, {peer2} を埋め込む。
        3人固定のプロンプトテンプレートに対応するため、リストから割り当てる。
        """
        # テンプレートに合わせて peer1, peer2 を用意
        # エージェントが足りない場合は空文字や "no one" を入れる等の処理
        p1 = self.peers[0] if len(self.peers) > 0 else "no one"
        p2 = self.peers[1] if len(self.peers) > 1 else ""
        
        # prompts.SYSTEM_PROMPT の中身:
        # "You are {name}... debating with {peer1}, {peer2}..."
        # 変数 {peer1}, {peer2} が存在することを前提にフォーマット
        return prompts.SYSTEM_PROMPT.format(
            name=self.name,
            peer1=p1,
            peer2=p2
        )

    # ──────────────────── 初回回答 ──────────────────── #
    def generate_initial_answer(self, topic: str) -> None:
        sys_prompt = self._get_system_prompt()
        self.initial_answer = self.llm_handler.generate_initial_answer(
            topic, 
            system_prompt=sys_prompt, 
            agent_name=self.name
        )
        self.initial_answer_str = (
            f"Answer: {self.initial_answer.get('answer','')}, "
            f"Reason: {self.initial_answer.get('reason','')}"
        )

    # ──────────────────── 最終回答 ──────────────────── #
    def generate_final_answer(self, topic: str, debate_history: str) -> dict[str, str]:
        sys_prompt = self._get_system_prompt()
        return self.llm_handler.generate_final_answer(
            topic,
            self.initial_answer_str,
            debate_history,
            system_prompt=sys_prompt,
            agent_name=self.name,
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
        sys_prompt = self._get_system_prompt()
        utterance, raw = self.llm_handler.generate_speaker_utterance(
            agent_name=self.name,
            system_prompt=sys_prompt,
            topic=topic,
            turn_log=turn_log,
            initial_answers_all=self.all_initial_answers_str,
            turn=turn,
            turns_left_for_agent=turns_left_for_agent,
            max_turn=max_turn,
        )
        if self.llm_handler.logger:
            self.llm_handler.logger.log_generated(
                agent_name=self.name, turn=turn, full_text=raw, phase="speaker_generated"
            )
        return utterance

    # ──────────────────── 思考（LISTENER） ──────────────────── #
    def think_only(
        self, *, topic: str, turn_log: str, turn: int, max_turn: int
    ) -> Dict[str, Any]:
        sys_prompt = self._get_system_prompt()
        thought, current_answer, consensus, raw = self.llm_handler.generate_listener_thought(
            agent_name=self.name,
            system_prompt=sys_prompt,
            topic=topic,
            turn_log=turn_log,
            initial_answers_all=self.all_initial_answers_str,
            turn=turn,
            max_turn=max_turn,
        )
        self.thought_history.append((turn, thought, current_answer, consensus))
        if self.llm_handler.logger:
            self.llm_handler.logger.log_generated(
                agent_name=self.name, turn=turn, full_text=raw, phase="listener_generated"
            )
        return {
            "thought": thought,
            "current_answer": current_answer,
            "consensus": consensus,
        }