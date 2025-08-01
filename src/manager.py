"""議論全体を統括する DiscussionManager (per‑agent turn‑wise history & random tie‑break)"""
from __future__ import annotations

import json
import random           # ★ 追加
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from .agent import Agent

HISTORY_WINDOW = 30  # 直近何行の履歴を渡すか


class DiscussionManager:
    def __init__(self, agents: List[Agent], config: dict[str, Any]):
        self.agents = agents
        self.topic: str = config["discussion"]["topic"]
        self.max_turns: int = config["discussion"]["max_turns"]

        self.history: List[Tuple[str, str]] = []  # (speaker, chunk)
        self.current_actions: dict[str, Any] = {}
        self.speaker: Optional[Agent] = None
        self.speaker_interrupt = False
        self.first_chunk = False

        logs_dir = Path(__file__).resolve().parents[1] / "logs"
        logs_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = logs_dir / f"discussion_log_{ts}.json"
        self.log_data: List[dict[str, Any]] = []
        self._write_log()

    # ───────────────────────── 公開 API ───────────────────────── #
    def run_discussion(self) -> None:
        print(f"=== Debate Start: {self.topic} ===")
        self._initialize_discussion()
        for turn in range(1, self.max_turns + 1):
            self._run_turn(turn)
        print("=== Debate End ===")

    # ───────────────────── Turn‑wise log (per agent) ───────────────────── #
    def _build_turn_log(self, agent_name: str, limit: int) -> str:
        """
        Assemble history for *agent_name*:
          • utterance line for every turn
          • (agent_name)(thought): ...  ← only that agent’s thought & only when they were NOT the speaker
        """
        lines: List[str] = []
        for e in self.log_data[-limit:]:
            if e["turn"] == 0:
                continue  # skip planning turn
            # utterance line
            if e["event_type"] in {"utterance", "interrupt"}:
                lines.append(f"{e['speaker']}: {e['content']}")
            elif e["event_type"] == "silence":
                lines.append("Silence")

            # agent's own thought for that turn (if any)
            if e["speaker"] != agent_name:  # speaker自体のターンでは thought を付けない
                for aa in e["agent_actions"]:
                    if aa["agent_name"] == agent_name:
                        th = aa["action_plan"].get("thought", "")
                        if th:
                            lines.append(f"  {agent_name}(thought): {th}")
                        break
        return "\n".join(lines)

    # ───────────────────────── 初期化 ───────────────────────── #
    def _initialize_discussion(self) -> None:
        self.current_actions.clear()
        for ag in self.agents:
            peers = [p.name for p in self.agents if p is not ag]
            turn_log = ""  # 初回は履歴なし
            self.current_actions[ag.name] = ag.plan_action(
                turn_log,
                "",
                self.topic,
                0,
                self.max_turns,
                silence=False,
                peer_names=peers,
            )
        # turn0 をログ
        self.log_data.append(
            {
                "turn": 0,
                "event_type": "plan",
                "speaker": None,
                "content": "",
                "agent_actions": [
                    {"agent_name": n, "action_plan": p}
                    for n, p in self.current_actions.items()
                ],
            }
        )
        self._write_log()
        self._determine_next_speaker(0)

    # ───────────────────── 1ターン処理 ───────────────────── #
    def _run_turn(self, turn: int) -> None:
        event_type, content, speaker_name = "silence", "", None

        # ---------- 発話フェーズ ----------
        if self.speaker:
            chunk = self.speaker.get_next_chunk()
            if chunk:
                event_type = "interrupt" if self.speaker_interrupt else "utterance"
                speaker_name = self.speaker.name
                content = chunk
                if self.history and self.history[-1][0] == speaker_name:
                    print(f"[Turn {turn}] {chunk}")
                else:
                    print(f"[Turn {turn}] {speaker_name}: {chunk}")
                self.history.append((speaker_name, chunk))
            else:
                self.speaker = None

        if event_type == "silence":
            print(f"[Turn {turn}] --- 沈黙 ---")

        # ---------- 行動計画フェーズ ----------
        self.current_actions.clear()
        last_event = (
            f"Silence:None:No one spoke in this turn({turn}/{self.max_turns})"
            if event_type == "silence"
            else f"{event_type}:{speaker_name}:{content}"
        )

        for ag in self.agents:
            if ag is self.speaker:
                continue
            peers = [p.name for p in self.agents if p is not ag]
            turn_log = self._build_turn_log(ag.name, HISTORY_WINDOW)
            self.current_actions[ag.name] = ag.plan_action(
                turn_log,
                last_event,
                self.topic,
                turn,
                self.max_turns,
                silence=(event_type == "silence"),
                peer_names=peers,
            )

        # ---------- 次ターンのスピーカー選定 ----------
        if turn < self.max_turns:
            self._determine_next_speaker(turn)

        # ---------- ログ ----------
        self.log_data.append(
            {
                "turn": turn,
                "event_type": event_type,
                "speaker": speaker_name,
                "content": content,
                "agent_actions": [
                    {"agent_name": n, "action_plan": p}
                    for n, p in self.current_actions.items()
                ],
            }
        )
        self._write_log()

    # ──────────────────── スピーカー選定 ──────────────────── #
    def _determine_next_speaker(self, current_turn: int) -> None:
        # 候補抽出
        candidates: List[tuple[str, dict[str, Any]]] = [
            (n, p)
            for n, p in self.current_actions.items()
            if p.get("action") in {"speak", "interrupt"}
        ]
        if not candidates:
            return

        # 最大 urgency を求め，同値ならランダム
        max_u = max(p.get("urgency", 0) for _, p in candidates)
        top: List[tuple[str, dict[str, Any]]] = [
            (n, p) for n, p in candidates if p.get("urgency", 0) == max_u
        ]
        next_name, next_plan = random.choice(top)  # ★ ランダム tie‑break

        if self.speaker and self.speaker.name == next_name:
            return

        self.speaker_interrupt = (
            self.speaker is not None and self.speaker.utterance_queue
        )

        self.speaker = next(a for a in self.agents if a.name == next_name)
        peers = [a.name for a in self.agents if a is not self.speaker]
        turn_log = self._build_turn_log(self.speaker.name, HISTORY_WINDOW)
        self.speaker.decide_to_speak(
            turn_log,
            self.topic,
            next_plan.get("thought", ""),
            next_plan.get("intent", ""),
            current_turn + 1,
            self.max_turns,
            peer_names=peers,
        )
        mode = "interrupt" if self.speaker_interrupt else "speak"
        print(f"[Manager] 👉 スピーカー変更: {self.speaker.name} ({mode})")

    # ──────────────────── ヘルパ: ログ書き込み ─────────────────── #
    def _write_log(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)
