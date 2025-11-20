"""LLM Handler for Llama-3.1-8B (System Prompt Aware)"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from llama_cpp import Llama

from . import prompts
from .prompt_logger import PromptLogger

# ===== JSON Schemas (変更なし) =====
qa_schema: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "maxLength": 2000},
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
        "thought": {"type": "string"},
        "current_answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
        "belief_team_consensus": {"type": "boolean"},
    },
    "required": ["thought", "current_answer", "belief_team_consensus"],
    "additionalProperties": False,
}


class LLMHandler:
    _instance: "LLMHandler" | None = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

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
            # モデルパスが見つからない場合は適宜調整してください
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        print(f"[LLMHandler] Loading model: {self.model_path}")
        self.model = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=config.get("n_gpu_layers", -1),
            n_ctx=config.get("n_ctx", 8192),
            temperature=config.get("temperature", 0.3),
            max_tokens=config.get("max_tokens", 1024),
            verbose=False,
        )
        print("[LLMHandler] Model loaded.")

    # ──────────────────── ユーティリティ ──────────────────── #
    @staticmethod
    def _safe_load_json(raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()
        # ```json ... ``` 除去
        text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```\s*$", "", text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # 簡易的な復旧処理（必要に応じて強化）
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except:
                pass
        return {}

    def _generate_json_only(
        self,
        user_prompt: str,
        system_prompt: str,  # 追加: システムプロンプトを受け取る
        *,
        agent_name: str,
        phase: str,
        response_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        
        # Llama-3 形式のメッセージ構築 (System / User 分離)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.logger:
            self.logger.log(
                agent_name, phase, 0, 
                system_prompt=system_prompt, 
                user_prompt=user_prompt
            )

        # JSON Schema 強制 (grammar) を使うとより確実ですが、
        # ここでは json_object モードを使用
        resp = self.model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": response_schema},
        )
        
        content = resp["choices"][0]["message"]["content"]
        parsed = self._safe_load_json(str(content))

        if self.logger:
            self.logger.log_generated(
                agent_name=agent_name, 
                turn=0, 
                full_text=str(content), 
                phase=f"{phase}_generated"
            )

        return parsed

    # ──────────────────── 各フェーズ (System Prompt 対応) ──────────────────── #
    
    def generate_initial_answer(
        self, topic: str, system_prompt: str, agent_name: str
    ) -> Dict[str, Any]:
        user_prompt = prompts.INITIAL_ANSWER_PROMPT_TEMPLATE.format(topic=topic)
        
        parsed = self._generate_json_only(
            user_prompt,
            system_prompt,
            agent_name=agent_name,
            phase="initial",
            response_schema=qa_schema
        )
        parsed.setdefault("answer", "")
        parsed.setdefault("reason", "")
        return parsed

    def generate_final_answer(
        self,
        topic: str,
        initial_answer_str: str,
        debate_history: str,
        system_prompt: str,
        agent_name: str,
    ) -> Dict[str, Any]:
        user_prompt = prompts.FINAL_ANSWER_PROMPT_TEMPLATE.format(
            topic=topic,
            initial_answer=initial_answer_str,
            debate_history=debate_history,
        )
        parsed = self._generate_json_only(
            user_prompt,
            system_prompt,
            agent_name=agent_name,
            phase="final",
            response_schema=qa_schema
        )
        parsed.setdefault("answer", "")
        parsed.setdefault("reason", "")
        return parsed

    def generate_speaker_utterance(
        self,
        *,
        agent_name: str,
        system_prompt: str,
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
            user_prompt,
            system_prompt,
            agent_name=agent_name,
            phase="speaker",
            response_schema=utterance_schema
        )
        utterance = (parsed.get("utterance") or "").strip()
        raw_text = json.dumps(parsed, ensure_ascii=False)
        return utterance, raw_text

    def generate_listener_thought(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        topic: str,
        turn_log: str,
        initial_answers_all: str,
        turn: int,
        max_turn: int,
    ) -> Tuple[str, str, bool, str]:
        # turns_left は Listener には厳密には不要だが prompt にあるなら渡す
        user_prompt = prompts.LISTENER_THINK_PROMPT_TEMPLATE.format(
            topic=topic,
            initial_answer=initial_answers_all,
            turn_log=turn_log,
            turn=turn,
            turns_left="N/A", 
            max_turn=max_turn,
        )
        parsed = self._generate_json_only(
            user_prompt,
            system_prompt,
            agent_name=agent_name,
            phase="listener",
            response_schema=thought_schema
        )
        
        thought = (parsed.get("thought") or "").strip()
        current_answer = (parsed.get("current_answer") or "").strip()
        # booleanのパース揺れ対応
        c_val = parsed.get("belief_team_consensus")
        if isinstance(c_val, str):
            consensus = c_val.lower() == "true"
        else:
            consensus = bool(c_val)

        raw_text = json.dumps(parsed, ensure_ascii=False)
        return thought, current_answer, consensus, raw_text