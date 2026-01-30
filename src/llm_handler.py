# src/llm_handler.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
from json_repair import repair_json

from llama_cpp import Llama

from . import prompts
from .prompt_logger import PromptLogger

qa_schema: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
    },
    "required": ["reason", "answer"],
    "strict": True,
    "additionalProperties": False,
}


plan_action_schema: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {"type": "string", "maxLength": 500},
        "action": {"type": "string", "enum": ["listen", "speak", "interrupt"]},
        "urgency": {"type": "integer", "minimum": 0, "maximum": 9},
        "purpose": {"type": "string", "maxLength": 50},
        "answer": {"type": "string", "enum": ["A", "B", "C", "D", "none"]},
        "consensus": {"type": "boolean"},
    },
    "required": ["thought", "action", "urgency", "purpose", "answer", "consensus"],
    "additionalProperties": False,
}


class LLMHandler:
    _instance: "LLMHandler" | None = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ──────────────────── 初期化 ──────────────────── #
    def __init__(
        self,
        config: Dict[str, Any],
        *,
        prompt_logger: Optional[PromptLogger] = None,
    ) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        self.logger = prompt_logger
        self.model_path = (
            Path(__file__).resolve().parents[1] / "models" / config["filename"]
        )
        if not self.model_path.exists():
            raise FileNotFoundError(f"モデルが見つかりません: {self.model_path}")

        print(f"[LLMHandler] 🔄 モデル読み込み開始: {self.model_path}")
        self.model = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=config.get("n_gpu_layers", -1),
            n_ctx=config.get("n_ctx", 4096),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 512),
            chat_format="llama-3",
        )
        print("[LLMHandler] ✅ モデル読み込み完了")

    # ──────────────────── 内部ユーティリティ ──────────────────── #
    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```", "", text).strip()
        return text

    @staticmethod
    def _safe_load_json(raw_text: str) -> Dict[str, Any]:
        txt = LLMHandler._strip_code_fence(raw_text)
        repaired = repair_json(txt)
        try:
            return json.loads(repaired)
        except Exception:
            pass
        txt_q = txt.replace("'", '"')
        try:
            return json.loads(txt_q)
        except Exception:
            pass
        first = txt.find("{")
        last = txt.rfind("}")
        if first != -1 and last != -1 and last > first:
            sub = txt[first : last + 1]
            try:
                return json.loads(sub)
            except Exception:
                pass
        return {}

    # ──────────────────── 共通 JSON 生成ユーティリティ ──────────────────── #
    def _generate_json_only(
        self,
        user_prompt: str,
        *,
        agent_name: str,
        persona: str,
        phase: str,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        戻り値: (パース済みJSON, usage辞書)
        usage辞書例: {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        resp = self.model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": (schema or qa_schema)},
            max_tokens=max_tokens,
        )
        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})  # トークン使用量の取得

        parsed: Dict[str, Any] = content if isinstance(content, dict) else self._safe_load_json(str(content))
        parsed.setdefault("answer", "")
        parsed.setdefault("reason", "")

        if self.logger:
            self.logger.log_generated(
                agent_name=agent_name,
                turn=0 if phase == "Initial" else 30,
                full_text=str(content),
                phase="initial_generated" if phase == "Initial" else "final_generated",
                token_stats=usage,
            )
        return parsed, usage

    # ====================== 通常: 初回/最終 ====================== #
    def generate_initial_answer(
        self, topic: str, *, agent_name: str, persona: str, max_turn: int, peer_names: Sequence[str]
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        
        system_prompt = self._build_system_prompt(
        name=agent_name,
        peer_names=peer_names,
        persona=persona,
        max_turn=max_turn,
        )
        prompt = prompts.INITIAL_ANSWER_PROMPT_TEMPLATE.format(topic=topic, name=agent_name, persona=persona)
        if self.logger:
            self.logger.log(
                agent_name=agent_name,
                turn=0,
                system_prompt=system_prompt,
                user_prompt=prompt,
                phase="initial_prompt",
            )
        return self._generate_json_only(
            prompt, agent_name=agent_name, persona=persona, phase="Initial"
        )

    def generate_final_answer(
        self,
        topic: str,
        initial_answer_str: str,
        debate_history: str,
        latest_thoughts: str,
        *,
        agent_name: str,
        persona: str,
        max_turn: int,
        peer_names: Sequence[str],
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        system_prompt = self._build_system_prompt(
        name=agent_name,
        peer_names=peer_names,
        persona=persona,
        max_turn=max_turn,
        )
        prompt = prompts.FINAL_ANSWER_PROMPT_TEMPLATE.format(
            topic=topic,
            initial_answer=initial_answer_str,
            debate_history=debate_history,
            latest_thoughts=latest_thoughts,
            name=agent_name,
            persona=persona,
        )
        if self.logger:
            self.logger.log(
                agent_name=agent_name,
                turn=30,
                system_prompt=system_prompt,
                user_prompt=prompt,
                phase="final_prompt",
            )
        return self._generate_json_only(
            prompt, agent_name=agent_name, persona=persona, phase="Final"
        )

    # ====================== adversary: 初回/最終 ====================== #
    def generate_adversary_initial_answer(
        self, topic: str, *, target_answer: str, agent_name: str, persona: str, max_turn: int, peer_names: Sequence[str]
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        prompt = prompts.ADVERSARY_INITIAL_ANSWER_PROMPT_TEMPLATE.format(
            topic=topic, name=agent_name, target_answer=target_answer
        )
        sys = prompts.ADVERSARY_SYSTEM_PROMPT.format(
            name = agent_name,
            peer1=peer_names[0] if len(peer_names) >= 1 else "Another agent",
            peer2=peer_names[1] if len(peer_names) >= 2 else "Another agent",
            max_turn=max_turn,
        )
        return self._generate_json_only(
            prompt, agent_name=agent_name, persona=persona, phase="Initial", system_prompt=sys
        )

    def generate_adversary_final_answer(
        self,
        topic: str,
        initial_answer_str: str,
        debate_history: str,
        latest_thoughts: str,
        *,
        target_answer: str,
        agent_name: str,
        persona: str,
        max_turn: int,
        peer_names: Sequence[str],
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        prompt = prompts.ADVERSARY_FINAL_ANSWER_PROMPT_TEMPLATE.format(
            topic=topic,
            initial_answer=initial_answer_str,
            debate_history=debate_history,
            latest_thoughts=latest_thoughts,
            name=agent_name,
            target_answer=target_answer,
        )
        sys = prompts.ADVERSARY_SYSTEM_PROMPT.format(
            name = agent_name,
            peer1=peer_names[0] if len(peer_names) >= 1 else "Another agent",
            peer2=peer_names[1] if len(peer_names) >= 2 else "Another agent",
            max_turn=max_turn,
        )
        return self._generate_json_only(
            prompt, agent_name=agent_name, persona=persona, phase="Final", system_prompt=sys
        )

    # ======================  内部: system prompt ====================== #
    def _build_system_prompt(
        self,
        *,
        name: str,
        peer_names: Sequence[str],
        persona: str,
        max_turn: int,
    ) -> str:
        p1 = peer_names[0] if len(peer_names) >= 1 else "Another agent"
        p2 = peer_names[1] if len(peer_names) >= 2 else "Another agent"
        return prompts.SYSTEM_PROMPT.format(
            name=name,
            persona=persona,
            peer1=p1,
            peer2=p2,
            max_turn=max_turn,
        )

    def _build_adversary_system_prompt(
        self,
        *,
        peer_names: Sequence[str],
        target_answer: str,
        agent_name: str,
        max_turn: int,
    ) -> str:
        p1 = peer_names[0] if len(peer_names) >= 1 else "Another agent"
        p2 = peer_names[1] if len(peer_names) >= 2 else "Another agent"
        return prompts.ADVERSARY_SYSTEM_PROMPT.format(
            name = agent_name,
            peer1=p1,
            peer2=p2,
            target_answer=target_answer,
            max_turn=max_turn,
        )

    # ======================  行動計画 / 発話生成 ====================== #
    def generate_action(
        self,
        user_prompt: str,
        *,
        turn: int,
        max_turn: int,
        agent_name: str,
        persona: str,
        topic: str,
        peer_names: Sequence[str],
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        system_prompt = self._build_system_prompt(
            name=agent_name,
            peer_names=peer_names,
            persona=persona,
            max_turn=max_turn,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.logger:
            self.logger.log(agent_name, "plan", turn, system_prompt, user_prompt)

        resp = self.model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": plan_action_schema},
            max_tokens=1024,
        )
        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        parsed = self._safe_load_json(content)

        parsed.setdefault("consensus", False)

        if self.logger:
            self.logger.log_generated(
                agent_name=agent_name,
                turn=turn,
                full_text=str(content),
                phase="plan_generated",
                token_stats=usage,
            )
        return parsed, usage

    def generate_action_adversary(
        self,
        user_prompt: str,
        *,
        turn: int,
        max_turn: int,
        agent_name: str,
        persona: str,
        topic: str,
        peer_names: Sequence[str],
        target_answer: str,
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        system_prompt = self._build_adversary_system_prompt(
            peer_names=peer_names, target_answer=target_answer, agent_name=agent_name, max_turn=max_turn
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.logger:
            self.logger.log(agent_name, "plan", turn, system_prompt, user_prompt)

        resp = self.model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": plan_action_schema},
            max_tokens=1024,
        )
        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        parsed = self._safe_load_json(content)
        parsed.setdefault("consensus", False)

        if self.logger:
            self.logger.log_generated(
                agent_name=agent_name,
                turn=turn,
                full_text=str(content),
                phase="plan_generated",
                token_stats=usage,
            )
        return parsed, usage

    def generate_utterance(
        self,
        user_prompt: str,
        *,
        turn: int,
        max_turn: int,
        agent_name: str,
        persona: str,
        topic: str,
        peer_names: Sequence[str],
        tokens_left: int,
    ) -> Tuple[str, str, Dict[str, int]]:
        system_prompt = self._build_system_prompt(
            name=agent_name,
            peer_names=peer_names,
            persona=persona,
            max_turn=max_turn,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.logger:
            self.logger.log(agent_name, "utterance", turn, system_prompt, user_prompt)

        resp = self.model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object",
                              "schema": {
                                        "type": "object",
                                        "properties": {
                                            "utterance": {"type": "string",},
                                        },
                                        "required": ["utterance"],
                                        "additionalProperties": False,
                                        }},
            max_tokens=1024,
        )
        raw_text = resp["choices"][0]["message"]["content"].strip()
        usage = resp.get("usage", {})
        parsed = self._safe_load_json(raw_text)
        utterance = parsed.get("utterance")
        utterance_text = utterance.strip() if isinstance(utterance, str) and utterance.strip() else raw_text

        if self.logger:
            self.logger.log_generated(
                agent_name=agent_name,
                turn=turn,
                full_text=raw_text,
                phase="utterance_generated",
                token_stats=usage,
            )
        return utterance_text, raw_text, usage

    def generate_utterance_adversary(
        self,
        user_prompt: str,
        *,
        turn: int,
        max_turn: int,
        agent_name: str,
        persona: str,
        topic: str,
        peer_names: Sequence[str],
        target_answer: str,
        tokens_left: int,
    ) -> Tuple[str, str, Dict[str, int]]:
        system_prompt = self._build_adversary_system_prompt(
            peer_names=peer_names, target_answer=target_answer, agent_name=agent_name, max_turn=max_turn
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.logger:
            self.logger.log(agent_name, "utterance", turn, system_prompt, user_prompt)

        resp = self.model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object",
                              "schema": {
                                        "type": "object",
                                        "properties": {
                                            "utterance": {"type": "string"},
                                        },
                                        "required": ["utterance"],
                                        "additionalProperties": False,
                                        }},
            max_tokens=1024,
        )
        raw_text = resp["choices"][0]["message"]["content"].strip()
        usage = resp.get("usage", {})
        parsed = self._safe_load_json(raw_text)
        utterance = parsed.get("utterance")
        utterance_text = utterance.strip() if isinstance(utterance, str) and utterance.strip() else raw_text

        if self.logger:
            self.logger.log_generated(
                agent_name=agent_name,
                turn=turn,
                full_text=raw_text,
                phase="utterance_generated",
                token_stats=usage,
            )
        return utterance_text, raw_text, usage
