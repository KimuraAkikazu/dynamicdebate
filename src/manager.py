"""議論全体を統括する DiscussionManager (per-agent turn-wise history & random tie-break)"""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .agent import Agent

HISTORY_WINDOW = 30  # エージェントに渡す履歴行数


class DiscussionManager:
    def __init__(
        self,
        agents: List[Agent],
        config: dict[str, Any],
        *,
        log_dir: Path | None = None,
    ):
        self.agents = agents
        self.topic: str = config["discussion"]["topic"]
        self.max_turns: int = config["discussion"]["max_turns"]

        # ---------- ログ用ディレクトリ ----------
        if log_dir is None:
            root = Path(__file__).resolve().parents[1] / "logs"
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = root / f"run_{run_id}"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.log_path = log_dir / "discussion_log.json"

        # ---------- 実行時状態 ----------
        self.history: List[Tuple[str, str]] = []
        self.current_actions: dict[str, Any] = {}
        self.speaker: Optional[Agent] = None
        self.speaker_interrupt = False
        self.final_answers: dict[str, dict[str, str]] = {}

        self.log_data: List[dict[str, Any]] = []
        self._write_log()

    # ───────────────────────── 公開 API ───────────────────────── #
    def run_discussion(self) -> dict[str, dict[str, str]]:
        print(f"=== Debate Start: {self.topic} ===")
        self._initialize_discussion()
        for turn in range(1, self.max_turns + 1):
            self._run_turn(turn)
        print("=== Debate End ===")
        self._collect_final_answers()
        return self.final_answers  # <<< 呼び出し元へ返す

    # ───────────────────────── 初期化 ───────────────────────── #
    def _initialize_discussion(self) -> None:
        # 1) 初回回答
        for ag in self.agents:
            ag.generate_initial_answer(self.topic)
            print(f"[Init] {ag.name} → {ag.initial_answer_str}")

        # 2) 全初回回答を共有
        all_initial = "\n".join(
            f"{ag.name}: Answer={ag.initial_answer.get('answer','')}, "
            f"Reason={ag.initial_answer.get('reasoning','')}"
            for ag in self.agents
        )
        for ag in self.agents:
            ag.all_initial_answers_str = all_initial

        # 3) ターン0の行動計画
        self.current_actions.clear()
        for ag in self.agents:
            peers = [p.name for p in self.agents if p is not ag]
            self.current_actions[ag.name] = ag.plan_action(
                turn_log="",
                last_event="",
                topic=self.topic,
                turn=0,
                max_turn=self.max_turns,
                silence=False,
                peer_names=peers,
            )

        self.log_data.append(
            {
                "turn": 0,
                "event_type": "plan",
                "speaker": None,
                "content": "",
                "initial_answers": {ag.name: ag.initial_answer for ag in self.agents},
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
            print(f"[Turn {turn}] --- Silence ---")

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
        # ---------- 次スピーカー選定 ----------
        if turn < self.max_turns:
            self._determine_next_speaker(turn)

        
        self._write_log()

    # ──────────────────── Turn-log 生成 ──────────────────── #
    def _build_turn_log(self, agent_name: str, limit: int) -> str:
        lines: List[str] = []
        for e in self.log_data[-limit:]:
            if e["turn"] == 0:
                continue
            if e["event_type"] in {"utterance", "interrupt"}:
                lines.append(f"Turn{e['turn']} {e['speaker']}({e['event_type']}): {e['content']}")
            elif e["event_type"] == "silence":
                lines.append(f"Turn{e['turn']} Silence")
            # thought
            if e["speaker"] != agent_name:
                for aa in e["agent_actions"]:
                    if aa["agent_name"] == agent_name:
                        th = aa["action_plan"].get("thought", "")
                        if th:
                            lines.append(f"Turn{e['turn']}   your thought: {th}")
                        break
        return "\n".join(lines)

    # ──────────────────── 最終回答収集 ──────────────────── #
    def _collect_final_answers(self) -> None:
        print("=== Collecting final answers ===")
        debate_history = "\n".join(
            f"{spk}: {txt}" for spk, txt in self.history[-1000:]
        )
        self.final_answers = {}
        for ag in self.agents:
            ans = ag.generate_final_answer(self.topic, debate_history)
            self.final_answers[ag.name] = ans
            print(f"[FINAL] {ag.name} -> {ans}")

        self.log_data.append(
            {
                "turn": "final",
                "event_type": "final_answers",
                "answers": self.final_answers,
            }
        )
        self._write_log()

    # ──────────────────── スピーカー選定 ──────────────────── #
    def _determine_next_speaker(self, current_turn: int) -> None:
        candidates = [
            (n, p)
            for n, p in self.current_actions.items()
            if p.get("action") in {"speak", "interrupt"}
        ]
        if not candidates:
            return
        max_u = max(p.get("urgency", 0) for _, p in candidates)
        top = [(n, p) for n, p in candidates if p.get("urgency", 0) == max_u]
        next_name, next_plan = random.choice(top)

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
        print(f"[Manager] 👉 Next speaker: {self.speaker.name} ({mode})")

    # ──────────────────── JSON 書込み ──────────────────── #
    def _write_log(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)
