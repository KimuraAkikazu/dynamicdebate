from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from llama_cpp import Llama

from . import prompts
from .prompt_logger import PromptLogger


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

    # ──────────────────── 共通 JSON 生成ユーティリティ ──────────────────── #
    def _generate_json_only(
        self,
        user_prompt: str,
        *,
        agent_name: str,
        persona: str,
        phase: str,
        max_tokens: int = 128,
    ) -> Dict[str, str]:
        system_prompt = (
            f"You are {agent_name}. {persona}\n"
            "You are debating with other AI agents."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            # print(messages)
            resp = self.model.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
            )
            print(resp)
            raw = resp["choices"][0]["message"]["content"]
            print(f"[{phase}-raw] {agent_name} >>> {raw}")
            return json.loads(raw)
        except Exception:
            # JSON が壊れている場合は {...} を抜き出す
            try:
                json_str = re.search(r"\{.*\}", raw, re.S).group(0)  # type: ignore
                return json.loads(json_str)
            except Exception:
                return {"answer": "", "reasoning": "failed to parse"}

    # 初回回答
    def generate_initial_answer(
        self, topic: str, *, agent_name: str, persona: str
    ) -> Dict[str, str]:
        prompt = prompts.INITIAL_ANSWER_PROMPT_TEMPLATE.format(topic=topic)

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
    ) -> Dict[str, str]:
        prompt = prompts.FINAL_ANSWER_PROMPT_TEMPLATE.format(
            topic=topic,
            initial_answer=initial_answer_str,
            debate_history=debate_history,
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
        topic: str,
        max_turn: int,
        turn: int,
    ) -> str:
        p1 = peer_names[0] if len(peer_names) >= 1 else "Another agent"
        p2 = peer_names[1] if len(peer_names) >= 2 else "Another agent"
        return prompts.SYSTEM_PROMPT.format(
            name=name,
            persona=persona,
            topic=topic,
            peer1=p1,
            peer2=p2,
            max_turn=max_turn,
            turn=turn,
            turns_left=max_turn - turn,
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
            topic=topic,
            max_turn=max_turn,
            turn=turn,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.logger:
            self.logger.log(agent_name, phase, turn, system_prompt, user_prompt)

        try:
            resp = self.model.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=256,
            )
            return json.loads(resp["choices"][0]["message"]["content"])
        except Exception:
            return {"action": "listen", "thought": "JSON parse error → listen"}

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
    ) -> str:
        phase = "utterance"
        system_prompt = self._build_system_prompt(
            name=agent_name,
            peer_names=peer_names,
            persona=persona,
            topic=topic,
            max_turn=max_turn,
            turn=turn,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.logger:
            self.logger.log(agent_name, phase, turn, system_prompt, user_prompt)

        try:
            resp = self.model.create_chat_completion(messages=messages)
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            return "…"