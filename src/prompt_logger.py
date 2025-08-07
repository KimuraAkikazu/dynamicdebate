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
    を保存する。

    - logs_root に `run_YYYYMMDD_HHMMSS` が含まれていれば
      そのディレクトリをそのまま使用。
    - 含まれていなければ自動で run_* を 1 つだけ作成。
      （problem_### など追加のサブフォルダは作らない）
    """

    def __init__(self, logs_root: Path) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # パス要素に run_ が含まれているか
        if any(part.startswith("run_") for part in logs_root.parts):
            run_dir = logs_root
        else:
            run_dir = logs_root / f"run_{ts}"

        run_dir.mkdir(parents=True, exist_ok=True)

        # ファイルは渡された (または作成した) フォルダ直下
        self.file_path = run_dir / f"prompt_log_{ts}.jsonl"
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
        rec = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "turn": turn,
            "agent": agent_name,
            "phase": phase,
            "system": system_prompt,
            "user": user_prompt,
        }
        self._fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fp.flush()

    # ---------- 生成全文 ---------- #
    def log_generated(
        self,
        agent_name: str,
        turn: int,
        full_text: str,
    ) -> None:
        rec = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "turn": turn,
            "agent": agent_name,
            "phase": "generated_text",
            "content": full_text,
        }
        self._fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fp.flush()

    # ---------- 終了 ---------- #
    def __del__(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass
