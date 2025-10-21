# src/llm_handler.py (fixed-order ablation version)
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from llama_cpp import Llama

from . import prompts
from .prompt_logger import PromptLogger

# ===== JSON Schemas =====
qa_schema: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "maxLength": 750},  # ~100 words
        "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
    },
    "required": ["reason", "answer"],
    "additionalProperties": False,
}

utterance_schema: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "utterance": {"type": "string"},
    },
    "required": ["utterance"],
    "additionalProperties": False,
}

thought_schema: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {"type": "string",},
    },
    "required": ["thought"],
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
        phase: str,
        response_schema: Dict[str, Any],
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        messages = [{"role": "user", "content": user_prompt}]
        if self.logger:
            # systemは最小構成なので user_prompt のみを保存
            self.logger.log(agent_name, phase, 0, system_prompt="", user_prompt=user_prompt)

        resp = self.model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": response_schema},
            max_tokens=max_tokens,
        )
        content = resp["choices"][0]["message"]["content"]

        parsed = content if isinstance(content, dict) else self._safe_load_json(str(content))

        # 生出力も保存
        if self.logger:
            self.logger.log_generated(agent_name=agent_name, turn=0, full_text=str(content), phase=f"{phase}_generated")

        return parsed

    # ──────────────────── 初回回答 / 最終回答 ──────────────────── #
    def generate_initial_answer(self, topic: str, *, agent_name: str, persona: str) -> Dict[str, Any]:
        prompt = prompts.INITIAL_ANSWER_PROMPT_TEMPLATE.format(topic=topic, name=agent_name, persona=persona)
        parsed = self._generate_json_only(
            prompt, agent_name=agent_name, phase="initial", response_schema=qa_schema
        )
        parsed.setdefault("answer", "")
        parsed.setdefault("reason", "")
        return parsed

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
        parsed = self._generate_json_only(
            prompt, agent_name=agent_name, phase="final", response_schema=qa_schema
        )
        parsed.setdefault("answer", "")
        parsed.setdefault("reason", "")
        return parsed

    # ──────────────────── 固定順序: 発言／思考 ──────────────────── #
    def generate_speaker_utterance(
        self,
        *,
        agent_name: str,
        persona: str,
        topic: str,
        turn_log: str,
        initial_answers_all: str,
        turn: int,
        turns_left_for_agent: int,
        max_turn: int,
    ) -> Tuple[str, str]:
        user_prompt = prompts.SPEAKER_TURN_PROMPT_TEMPLATE.format(
            name=agent_name,
            topic=topic,
            initial_answer=initial_answers_all,
            turn_log=turn_log,
            turn=turn,
            turns_left=turns_left_for_agent,
            max_turn=max_turn,
        )
        parsed = self._generate_json_only(
            user_prompt, agent_name=agent_name, phase="speaker", response_schema=utterance_schema
        )
        utterance = (parsed.get("utterance") or "").strip()
        raw_text = json.dumps(parsed, ensure_ascii=False)
        return utterance, raw_text

    def generate_listener_thought(
        self,
        *,
        agent_name: str,
        persona: str,
        topic: str,
        turn_log: str,
        initial_answers_all: str,
        turn: int,
        max_turn: int,
    ) -> Tuple[str, str]:
        user_prompt = prompts.LISTENER_THINK_PROMPT_TEMPLATE.format(
            name=agent_name,
            topic=topic,
            initial_answer=initial_answers_all,
            turn_log=turn_log,
            turn=turn,
            turns_left="N/A",
            max_turn=max_turn,
        )
        parsed = self._generate_json_only(
            user_prompt, agent_name=agent_name, phase="listener", response_schema=thought_schema
        )
        thought = (parsed.get("thought") or "").strip()
        raw_text = json.dumps(parsed, ensure_ascii=False)
        return thought, raw_text
