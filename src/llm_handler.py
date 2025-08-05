# src/llm_handler.py
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

    # ──────────────────── 内部ユーティリティ ──────────────────── #
    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """```json ... ``` や ``` ... ``` を除去して戻す"""
        text = text.strip()
        # 先頭の ```json / ``` を除く
        text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
        # 末尾の ``` を除く
        text = re.sub(r"\s*```$", "", text).strip()
        return text

    @staticmethod
    def _safe_load_json(raw_text: str) -> Dict[str, str]:
        """
        多少壊れた JSON でも best-effort でパースして dict を返す。
        必須キーが無ければ空文字列で埋める。
        """
        # コードフェンス削除
        txt = LLMHandler._strip_code_fence(raw_text)

        # そのままチャレンジ
        try:
            return json.loads(txt)
        except Exception:
            pass

        # シングルクォート → ダブルクォート
        txt_q = txt.replace("'", '"')
        try:
            return json.loads(txt_q)
        except Exception:
            pass

        # 最初の { と 最後の } でサブストリング抽出
        first = txt.find("{")
        last = txt.rfind("}")
        if first != -1 and last != -1 and last > first:
            sub = txt[first : last + 1]
            try:
                return json.loads(sub)
            except Exception:
                # サブストリングでも失敗したらあきらめ
                pass

        # ここまで来たらパース不能
        return {
            "answer": "",
            "reasoning": f"failed to parse: {raw_text[:100]}..."  # 先頭100文字だけ保持
        }

    # ──────────────────── 共通 JSON 生成ユーティリティ ──────────────────── #
    def _generate_json_only(
        self,
        user_prompt: str,
        *,
        agent_name: str,
        persona: str,
        phase: str,
        max_tokens: int = 512,
    ) -> Dict[str, str]:
        system_prompt = (
            f"You are {agent_name}. {persona}\n"
            "You are debating with other AI agents."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # llama-cpp-python は response_format を無視することがあるが
        # 付けても害はないので一応付けておく
        resp = self.model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
        )

        content = resp["choices"][0]["message"]["content"]

        # まれに dict で返ってくる場合もある
        if isinstance(content, dict):
            parsed = content
        else:
            parsed = self._safe_load_json(str(content))

        # 念のため必須キーを保証
        parsed.setdefault("reasoning", "")
        parsed.setdefault("answer", "")
        

        if self.logger:
            self.logger.log_generated(
                agent_name=agent_name,
                turn=0 if phase == "Initial" else 30,  # 初回/最終は負数で区別
                full_text=str(content),
            )

        return parsed

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

        resp = self.model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=256,
        )
        return self._safe_load_json(resp["choices"][0]["message"]["content"])

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

        resp = self.model.create_chat_completion(messages=messages)
        return resp["choices"][0]["message"]["content"].strip()
