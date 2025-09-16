# src/llm_handler.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from llama_cpp import Llama

from . import prompts
from .prompt_logger import PromptLogger

qa_schema: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},  # ~100 words
        "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
    },
    "required": ["reason", "answer"],
    "strict": True,
    "additionalProperties": False,
}

plan_action_schema: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {"type": "string", "maxLength": 300},
        "action": {"type": "string", "enum": ["listen", "speak", "interrupt"]},
        "urgency": {"type": "integer", "minimum": 0, "maximum": 4},
        "intent": {"type": "string", "maxLength": 50},
        "consensus": {
            "type": "object",
            "properties": {
                "agreed": {"type": "boolean"},
                "answer": {"type": "string", "enum": ["A", "B", "C", "D","none"]},
            },
            "required": ["agreed"],
            "additionalProperties": False,
        },
    },
    "required": ["thought", "action", "urgency", "intent", "consensus"],
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
        """```json ... ``` や ``` ... ``` を除去して戻す"""
        text = text.strip()
        text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```", "", text).strip()
        return text

    @staticmethod
    def _safe_load_json(raw_text: str) -> Dict[str, Any]:
        """
        多少壊れた JSON でも best-effort でパースして dict を返す。
        """
        txt = LLMHandler._strip_code_fence(raw_text)

        # try-as-is
        try:
            return json.loads(txt)
        except Exception:
            pass

        # single quotes → double quotes
        txt_q = txt.replace("'", '"')
        try:
            return json.loads(txt_q)
        except Exception:
            pass

        # substring between first { ... last }
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
    ) -> Dict[str, Any]:
        messages = [
            {"role": "user", "content": user_prompt},
        ]

        resp = self.model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": qa_schema},
            max_tokens=max_tokens,
        )
        content = resp["choices"][0]["message"]["content"]

        if isinstance(content, dict):
            parsed: Dict[str, Any] = content
        else:
            parsed = self._safe_load_json(str(content))

        parsed.setdefault("answer", "")
        parsed.setdefault("reason", "")

        if self.logger:
            self.logger.log_generated(
                agent_name=agent_name,
                turn=0 if phase == "Initial" else 30,
                full_text=str(content),
                phase="initial_generated" if phase == "Initial" else "final_generated",
            )

        return parsed

    # 初回回答
    def generate_initial_answer(
        self, topic: str, *, agent_name: str, persona: str
    ) -> Dict[str, Any]:
        prompt = prompts.INITIAL_ANSWER_PROMPT_TEMPLATE.format(topic=topic, name=agent_name, persona=persona)
        return self._generate_json_only(
            prompt, agent_name=agent_name, persona=persona, phase="Initial"
        )

    # 最終回答
    def generate_final_answer(
        self,
        topic: str,
        initial_answer_str: str,
        debate_history: str,
        *,
        agent_name: str,
        persona: str,
    ) -> Dict[str, Any]:
        prompt = prompts.FINAL_ANSWER_PROMPT_TEMPLATE.format(
            topic=topic,
            initial_answer=initial_answer_str,
            debate_history=debate_history,
            name=agent_name,
            persona=persona,
        )
        return self._generate_json_only(
            prompt, agent_name=agent_name, persona=persona, phase="Final"
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
    ) -> Dict[str, Any]:
        phase = "plan"
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
            self.logger.log(agent_name, phase, turn, system_prompt, user_prompt)

        resp = self.model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": plan_action_schema},
            max_tokens=256,
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = self._safe_load_json(content)

        # ★ 追加: plan の生出力（consensus含む）も JSONL に保存
        if self.logger:
            self.logger.log_generated(
                agent_name=agent_name,
                turn=turn,
                full_text=str(content),
                phase="plan_generated",
            )

        return parsed

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
    ) -> Tuple[str, str]:
        """
        Returns (utterance_text, raw_model_output).
        If the model returns JSON with an "utterance" field, that value is used.
        Otherwise the raw text itself is treated as the utterance.
        """
        phase = "utterance"
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
            self.logger.log(agent_name, phase, turn, system_prompt, user_prompt)

        # 発話生成でも JSON Object を要求
        resp = self.model.create_chat_completion(
            messages=messages, response_format={"type": "json_object"}
        )
        raw_text = resp["choices"][0]["message"]["content"].strip()

        parsed = self._safe_load_json(raw_text)
        utterance = parsed.get("utterance")
        if isinstance(utterance, str) and utterance.strip():
            utterance_text = utterance.strip()
        else:
            # モデルが JSON で返さなかった場合はそのまま発話とみなす
            utterance_text = raw_text

        # 生出力も残す
        if self.logger:
            self.logger.log_generated(
                agent_name=agent_name,
                turn=turn,
                full_text=raw_text,
                phase="utterance_generated",
            )

        return utterance_text, raw_text
