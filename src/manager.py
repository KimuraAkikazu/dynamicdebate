"""議論全体を統括する DiscussionManager（固定順序・割り込みなし・各人3回）"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agent import Agent

HISTORY_WINDOW = 1000  # 発話履歴として渡す行数（十分大きく）


class DiscussionManager:
    def __init__(
        self,
        agents: List[Agent],
        config: Dict[str, Any],
        *,
        log_dir: Path | None = None,
    ):
        self.agents = agents
        self.topic: str = config["discussion"]["topic"]

        # 固定順序設定
        order_cfg = config.get("discussion", {}).get("speaking_order")
        if order_cfg and isinstance(order_cfg, list) and all(isinstance(n, str) for n in order_cfg):
            self.order: List[str] = list(order_cfg)
        else:
            self.order = [a.name for a in agents]

        self.turns_per_agent: int = int(config.get("discussion", {}).get("turns_per_agent", 3))
        self.max_turns: int = len(self.order) * self.turns_per_agent

        # name -> Agent
        self._agent_by_name = {a.name: a for a in self.agents}
        # 全員存在検証
        missing = [n for n in self.order if n not in self._agent_by_name]
        if missing:
            raise ValueError(f"speaking_order に未知のエージェント名があります: {missing}")

        # ---------- ログ用ディレクトリ ----------
        if log_dir is None:
            root = Path(__file__).resolve().parents[1] / "logs"
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = root / f"run_{run_id}"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.log_path = log_dir / "discussion_log.json"

        # ---------- 実行時状態 ----------
        self.history: List[Tuple[str, str]] = []  # [(speaker, utterance)]
        self.final_answers: Dict[str, Dict[str, str]] = {}
        self.log_data: List[Dict[str, Any]] = []

        self._write_log()  # 空配列でファイルを作成

    # ───────────────────────── 公開 API ───────────────────────── #
    def run_discussion(self) -> Dict[str, Dict[str, str]]:
        print(f"=== Debate Start (Fixed Order): {self.topic} ===")
        self._initialize_discussion()

        turn = 0
        # ラウンドロビン：各エージェントが turns_per_agent 回発言
        for r in range(1, self.turns_per_agent + 1):
            for name in self.order:
                turn += 1
                self._run_fixed_turn(turn, speaker_name=name)

        print("=== Debate End ===")
        self._collect_final_answers()
        return self.final_answers

    # ───────────────────────── 初期化 ───────────────────────── #
    def _initialize_discussion(self) -> None:
        # 1) 初回回答
        for ag in self.agents:
            ag.generate_initial_answer(self.topic)
            print(f"[Init] {ag.name} → Answer={ag.initial_answer.get('answer','')}, "
                  f"Reason={ag.initial_answer.get('reason','')}")
        # 2) 全初回回答を共有
        all_initial = "\n".join(
            f"{ag.name}: Answer={ag.initial_answer.get('answer','')}, "
            f"Reason={ag.initial_answer.get('reason','')}"
            for ag in self.agents
        )
        for ag in self.agents:
            ag.all_initial_answers_str = all_initial

        init_record: Dict[str, Any] = {
            "turn": 0,
            "event_type": "init",
            "speaker": None,
            "content": "",
            "initial_answers": {ag.name: ag.initial_answer for ag in self.agents},
            "config": {
                "speaking_order": self.order,
                "turns_per_agent": self.turns_per_agent,
                "max_turns": self.max_turns,
            },
        }
        self.log_data.append(init_record)
        self._write_log()

    # ───────────────────── 固定順序の1ターン処理 ───────────────────── #
    def _run_fixed_turn(self, turn: int, speaker_name: str) -> None:
        speaker = self._agent_by_name[speaker_name]

        # 発言者向け turn_log（直近の発話のみ）
        turn_log_for_speaker = self._build_turn_log(limit=HISTORY_WINDOW)

        # 発言者が1発話
        turns_left_for_agent = self._turns_left_of_agent_after_this_turn(speaker_name, turn)
        utterance = speaker.produce_speech(
            topic=self.topic,
            turn_log=turn_log_for_speaker,
            turn=turn,
            turns_left_for_agent=turns_left_for_agent,
            max_turn=self.max_turns,
        )
        if utterance:
            print(f"[Turn {turn}] {speaker_name}: {utterance}")
            self.history.append((speaker_name, utterance))
        else:
            print(f"[Turn {turn}] {speaker_name}: (empty utterance)")

        # 非発言者の thought を取得
        listener_thoughts: List[Dict[str, str]] = []
        for ag in self.agents:
            if ag is speaker:
                continue
            # 非発言者にも同じ発話履歴を渡す
            thought = ag.think_only(
                topic=self.topic,
                turn_log=self._build_turn_log(limit=HISTORY_WINDOW),
                turn=turn,
                max_turn=self.max_turns,
            )
            listener_thoughts.append({"agent_name": ag.name, "thought": thought})

        # ログ
        record: Dict[str, Any] = {
            "turn": turn,
            "event_type": "utterance",
            "speaker": speaker_name,
            "content": utterance,
            "listener_thoughts": listener_thoughts,
        }
        self.log_data.append(record)
        self._write_log()

    # ──────────────────── Turn-log（発話履歴） ──────────────────── #
    def _build_turn_log(self, limit: int) -> str:
        lines: List[str] = []
        for i, (spk, txt) in enumerate(self.history[-limit:], start=1):
            lines.append(f"{spk}: {txt}")
        return "\n".join(lines)

    def _turns_left_of_agent_after_this_turn(self, agent_name: str, turn: int) -> int:
        """このターン発言後に、そのエージェントが残す発言回数（情報用）"""
        idx_in_order = self.order.index(agent_name)  # 0-based
        # これまでに agent_name が何回スピーカーだったかを数える
        # 1..turn の中で、order順に回っているので計算できる
        per_round = len(self.order)
        spoken_count = 0
        for t in range(1, turn + 1):
            r = (t - 1) // per_round  # 0-based round
            pos = (t - 1) % per_round
            if self.order[pos] == agent_name:
                spoken_count += 1
        total_allowed = self.turns_per_agent
        return max(total_allowed - spoken_count, 0)

    # ──────────────────── 最終回答収集 ──────────────────── #
    def _collect_final_answers(self) -> None:
        print("=== Collecting final answers ===")
        debate_history = "\n".join(f"{spk}: {txt}" for spk, txt in self.history[-1000:])
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

    # ──────────────────── JSON 書込み ──────────────────── #
    def _write_log(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)
