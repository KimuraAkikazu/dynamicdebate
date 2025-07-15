"""llama-cpp-python をラップするシングルトン LLMHandler"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from llama_cpp import Llama
from . import prompts
from .prompt_logger import PromptLogger


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
        self.model_path = Path(__file__).resolve().parents[1] / "models" / config["filename"]
        if not self.model_path.exists():
            raise FileNotFoundError(f"モデルが見つかりません: {self.model_path}")

        print(f"[LLMHandler] 🔄 モデル読み込み開始: {self.model_path}")
        self.model = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=config.get("n_gpu_layers", -1),
            n_ctx=config.get("n_ctx", 4096),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 512),
            chat_format="llama-3",
        )
        print("[LLMHandler] ✅ モデル読み込み完了")

    # ======================  公開 API  ====================== #
    def _build_system_prompt(self, max_turn: int, turn: int) -> str:
        return prompts.SYSTEM_PROMPT.format(
            max_turn=max_turn,
            turn=turn,
            turns_left=max_turn - turn,
        )

    def generate_action(
        self,
        user_prompt: str,
        turn: int,
        max_turn: int,
        *,
        agent_name: str | None = None,
    ) -> Dict[str, Any]:
        phase = "plan"
        system_prompt = self._build_system_prompt(max_turn, turn)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.logger and agent_name:
            self.logger.log(agent_name, phase, turn, system_prompt, user_prompt)

        try:
            resp = self.model.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = resp["choices"][0]["message"]["content"]
            return json.loads(raw)
        except Exception:
            return {"action": "listen", "thought": "JSON 解析失敗→listen"}

    def generate_utterance(
        self,
        user_prompt: str,
        turn: int,
        max_turn: int,
        *,
        agent_name: str | None = None,
    ) -> str:
        phase = "utterance"
        system_prompt = self._build_system_prompt(max_turn, turn)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.logger and agent_name:
            self.logger.log(agent_name, phase, turn, system_prompt, user_prompt)

        try:
            resp = self.model.create_chat_completion(messages=messages)
            print(resp["choices"][0]["message"]["content"])
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            return "…"
