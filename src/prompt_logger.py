"""プロンプト／生成テキストを JSON Lines で保存するユーティリティ"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class PromptLogger:
    """
    1 行 1 レコード (JSONL) で
      • system / user プロンプト
      • モデルが生成した全文
    の両方を保存する。
    """

    def __init__(self, logs_root: Path) -> None:
        logs_root.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = logs_root / f"prompt_log_{ts}.jsonl"
        self._fp = self.file_path.open("w", encoding="utf-8")

    # ---------- 入力プロンプトを保存 ---------- #
    def log(
        self,
        agent_name: str,
        phase: str,
        turn: int,
        system_prompt: str,
        user_prompt: str,
    ) -> None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "turn": turn,
            "agent": agent_name,
            "phase": phase,               # "plan" / "utterance"
            "system": system_prompt,
            "user": user_prompt,
        }
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fp.flush()

    # ---------- モデル生成全文を保存 ---------- #
    def log_generated(
        self,
        agent_name: str,
        turn: int,
        full_text: str,
    ) -> None:
        """
        チャンク化前の “一括生成テキスト” を保存する。
        """
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "turn": turn,
            "agent": agent_name,
            "phase": "generated_text",
            "content": full_text,
        }
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fp.flush()

    # ---------- 終了処理 ---------- #
    def __del__(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass
