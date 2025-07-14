"""議論全体を統括する DiscussionManager"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from .agent import Agent


HISTORY_WINDOW = 100  # 直近何行の履歴を渡すか

class DiscussionManager:
    def __init__(self, agents: List[Agent], config: dict[str, Any]):
        self.agents = agents
        self.topic: str = config["discussion"]["topic"]
        self.max_turns: int = config["discussion"]["max_turns"]

        self.history: List[str] = []          # 全発話ログ（テキストのみ）
        self.current_actions: dict[str, Any] = {}
        self.speaker: Optional[Agent] = None  # 現在発言中のエージェント

        logs_dir = Path(__file__).resolve().parents[1] / "logs"
        logs_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = logs_dir / f"discussion_log_{ts}.json"
        self.log_data: List[dict[str, Any]] = []
        self._write_log()  # 空ログを作成しておく

    # ---------------- 公開 API ---------------- #
    def run_discussion(self) -> None:
        print(f"=== 議論開始: {self.topic} ===")
        self._initialize_discussion()
        for turn in range(1, self.max_turns + 1):
            self._run_turn(turn)
        print("=== 議論終了 ===")

    # ---------------- 初期化 ---------------- #
    def _initialize_discussion(self) -> None:
        # ターン0: 全員が行動計画を立てる
        self.current_actions = {
            ag.name: ag.plan_action("", "", self.topic, 0, self.max_turns) for ag in self.agents
        }
        # 最初のスピーカー決定
        self._determine_next_speaker(0)

    # ---------------- 1ターン処理 ---------------- #
    def _run_turn(self, turn: int) -> None:
        event_type, content, speaker_name = "silence", "", None

        # 発話フェーズ ------------------------------------------------------
        if self.speaker:
            chunk = self.speaker.get_next_chunk()
            if chunk:                                # まだ発話が残っている
                event_type, content = "utterance", chunk
                speaker_name = self.speaker.name
                self.history.append(f"{speaker_name}: {chunk}")
                print(f"[Turn {turn}] {speaker_name}: {chunk}")
            else:                                    # 発話を出し切った
                self.speaker = None

        if event_type == "silence":
            print(f"[Turn {turn}] --- 沈黙 ---")

        # 行動計画フェーズ --------------------------------------------------
        self.current_actions.clear()

        last_event = (
            f"沈黙:None:このターン({turn}/{self.max_turns})では誰も発言しませんでした"
            if event_type == "silence"
            else f"{event_type}:{speaker_name}:{content}"
        )

        hist_str = "\n".join(self.history[-HISTORY_WINDOW:])

        for ag in self.agents:
            if ag is self.speaker:
                continue  # 発言者は計画不要
            self.current_actions[ag.name] = ag.plan_action(
                hist_str, last_event, self.topic, turn, self.max_turns
            )

        # 次ターンのスピーカー選定 -----------------------------------------
        self._determine_next_speaker(turn)

        # ログ -------------------------------------------------------------
        self.log_data.append(
            {
                "turn": turn,
                "event_type": event_type,
                "speaker": speaker_name,
                "content": content,
                "agent_actions": [
                    {"agent_name": n, "action_plan": p} for n, p in self.current_actions.items()
                ],
            }
        )
        self._write_log()

    # ---------------- スピーカー選定 ---------------- #
    def _determine_next_speaker(self, current_turn: int) -> None:
        # speak を選択した聞き手を urgency でソート
        candidates = [
            (n, p) for n, p in self.current_actions.items() if p.get("action") == "speak"
        ]
        if not candidates:
            return  # 誰も発言希望なし → 次ターンも沈黙の可能性

        candidates.sort(key=lambda x: x[1].get("urgency", 0.0), reverse=True)
        next_name, next_plan = candidates[0]

        # すでにその人が発言権を持っていれば何もしない
        if self.speaker and self.speaker.name == next_name:
            return

        # 新しいスピーカーに交代
        self.speaker = next(a for a in self.agents if a.name == next_name)
        # 常に新しい全文を生成（キューは Agent 内で clear 済み）
        self.speaker.decide_to_speak(
            "\n".join(self.history[-HISTORY_WINDOW:]),
            self.topic,
            next_plan.get("thought", ""),
            current_turn + 1,
            self.max_turns,
        )
        print(f"[Manager] 👉 スピーカー変更: {self.speaker.name} (urgency={next_plan.get('urgency')})")

    # ---------------- ログ書き込み ---------------- #
    def _write_log(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)
