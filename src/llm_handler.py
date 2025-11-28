"""LLM Handler for Llama-3.1-8B (System Prompt Aware)"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from llama_cpp import Llama

from . import prompts
from .prompt_logger import PromptLogger

# ===== JSON Schemas =====
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
        "utterance": {"type": "string", "maxLength": 1000},
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
        """
        壊れた JSON / JSON文字列 / 末尾ゴミ付き すべてを最大限復元する JSON パーサ
        ---------------------------------------------------------
        例：
        "{ \"answer\":\"D\", \"reason\":\"...\"}}"    → OK
        "\"{ \\\"answer\\\":\\\"D\\\" }\""          → 2段階で展開してOK
        foo{ "answer":"B","reason":"x"}bar         → {...} のみ検出して復元
        """

        # -------- 事前クリーニング --------
        text = raw_text.strip()
        text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```\s*$", "", text)

        # ===== ① まずストレートに JSON として読めるか試す =====
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
            # JSON文字列だった場合 → 再パース
            if isinstance(obj, str):
                try:
                    obj2 = json.loads(obj)
                    if isinstance(obj2, dict):
                        return obj2
                except Exception:
                    pass
        except Exception:
            pass

        # ===== ② {と} の対応をカウントして最初に閉じる位置まで抽出 =====
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:       # ←最初に閉じたところ＝完全なJSON
                        candidate = text[start : i+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            pass

        # ===== ③ まだ無理なら JSON文字列の可能性に賭けて最終チャレンジ =====
        m = re.search(r"\"(\{.*\})\"", text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        # ------------ どうしても無理なら空で返す --------------
        return {}


    def _generate_json_only(
        self,
        user_prompt: str,
        system_prompt: str,
        *,
        agent_name: str,
        phase: str,
        response_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.logger:
            self.logger.log(
                agent_name,
                phase,
                0,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

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
                phase=f"{phase}_generated",
            )

        return parsed

    # ──────────────────── 各フェーズ ──────────────────── #

    def generate_initial_answer(
        self,
        topic: str,
        system_prompt: str,
        agent_name: str,
    ) -> Dict[str, Any]:
        user_prompt = prompts.INITIAL_ANSWER_PROMPT_TEMPLATE.format(topic=topic)

        parsed = self._generate_json_only(
            user_prompt,
            system_prompt,
            agent_name=agent_name,
            phase="initial",
            response_schema=qa_schema,
        )
        parsed.setdefault("answer", "")
        parsed.setdefault("reason", "")
        return parsed

    def generate_adversary_initial_answer(
        self,
        topic: str,
        target_answer: str,
        system_prompt: str,
        agent_name: str,
    ) -> Dict[str, Any]:
        """
        敵対エージェント用の初回回答。
        - ADVERSARY_INITIAL_ANSWER_PROMPT_TEMPLATE を使う
        - モデル出力に関わらず answer は target_answer に強制
        """
        user_prompt = prompts.ADVERSARY_INITIAL_ANSWER_PROMPT_TEMPLATE.format(
            topic=topic,
            target_answer=target_answer,
        )

        parsed = self._generate_json_only(
            user_prompt,
            system_prompt,
            agent_name=agent_name,
            phase="initial_adversary",
            response_schema=qa_schema,
        )

        ans = parsed.get("answer")
        if ans not in ["A","B","C","D"]:
            parsed["answer"] = target_answer

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
            response_schema=qa_schema,
        )
        parsed.setdefault("answer", "")
        parsed.setdefault("reason", "")
        return parsed

    def generate_adversary_final_answer(
        self,
        topic: str,
        initial_answer_str: str,
        debate_history: str,
        target_answer: str,
        system_prompt: str,
        agent_name: str,
    ) -> Dict[str, Any]:
        """
        敵対エージェント用の最終回答。
        - 通常の FINAL_ANSWER_PROMPT_TEMPLATE を使うが
        - 最後に answer を target_answer に固定する
        """
        user_prompt = prompts.FINAL_ANSWER_PROMPT_TEMPLATE.format(
            topic=topic,
            initial_answer=initial_answer_str,
            debate_history=debate_history,
        )
        parsed = self._generate_json_only(
            user_prompt,
            system_prompt,
            agent_name=agent_name,
            phase="final_adversary",
            response_schema=qa_schema,
        )

        parsed["answer"] = target_answer  # 強制
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
            response_schema=utterance_schema,
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
            response_schema=thought_schema,
        )

        thought = (parsed.get("thought") or "").strip()
        current_answer = (parsed.get("current_answer") or "").strip()
        c_val = parsed.get("belief_team_consensus")
        if isinstance(c_val, str):
            consensus = c_val.lower() == "true"
        else:
            consensus = bool(c_val)

        raw_text = json.dumps(parsed, ensure_ascii=False)
        return thought, current_answer, consensus, raw_text
