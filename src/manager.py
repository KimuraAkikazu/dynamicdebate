"""議論全体を統括する DiscussionManager (per-agent turn-wise history & random tie-break + early stop + rich logging)"""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agent import Agent

HISTORY_WINDOW = 30  # エージェントに渡す履歴行数


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
        self.max_turns: int = config["discussion"]["max_turns"]

        # ---------- ログ用ディレクトリ ----------
        if log_dir is None:
            root = Path(__file__).resolve().parents[1] / "logs"
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = root / f"run_{run_id}"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir.resolve()
        self.log_path = self.log_dir / "discussion_log.json"

        # ---------- 実行時状態 ----------
        self.history: List[Tuple[str, str]] = []
        self.current_actions: Dict[str, Dict[str, Any]] = {}
        self.speaker: Optional[Agent] = None
        self.speaker_interrupt = False  # 既存フィールドは維持（互換のため未使用）
        self.final_answers: Dict[str, Dict[str, str]] = {}

        # ★ 追加: interrupt を「1回だけ適用」するためのワンショットフラグ
        self._interrupt_once: bool = False

        self.log_data: List[Dict[str, Any]] = []
        # 各エージェントの「直近1ターンの thought」のみ保持
        self.latest_thought_by_agent: Dict[str, str] = {}

        # ---------- 早期終了用 ----------
        self.last_plan_by_agent: Dict[str, Dict[str, Any]] = {}
        self.consensus_streak: int = 0
        self._early_stop_answer: Optional[str] = None
        self.early_cfg: Dict[str, Any] = config.get("discussion", {}).get("early_stop", {})
        self._early_enabled: bool = bool(self.early_cfg.get("enabled", False))
        self._req_consec: int = int(self.early_cfg.get("require_consecutive", 1))
        self._min_turns: int = int(self.early_cfg.get("min_turns", 1))

        self._write_log()  # 空配列でファイルを作成

    # ───────────────────────── 公開 API ───────────────────────── #
    def run_discussion(self) -> Dict[str, Dict[str, str]]:
        print(f"=== Debate Start: {self.topic} ===")
        self._initialize_discussion()
        for turn in range(1, self.max_turns + 1):
            if self._run_turn(turn):
                print("=== Early stop: consensus reached ===")
                break
        print("=== Debate End ===")
        self._collect_final_answers()
        return self.final_answers

    # ───────────────────────── 初期化 ───────────────────────── #
    def _initialize_discussion(self) -> None:
        # 1) 初回回答
        for ag in self.agents:
            ag.generate_initial_answer(self.topic)
            print(f"[Init] {ag.name} → {ag.initial_answer_str}")

        # 2) 全初回回答を共有
        all_initial = "\n".join(
            f"{{Name: {ag.name}, Answer: {ag.initial_answer.get('answer','')}, Reason: {ag.initial_answer.get('reason','') }}}"
            "\n"
            for ag in self.agents
        )
        for ag in self.agents:
            ag.all_initial_answers_str = all_initial

        # 3) ターン0の行動計画
        self.current_actions.clear()
        for ag in self.agents:
            peers = [p.name for p in self.agents if p is not ag]
            self.current_actions[ag.name] = ag.plan_action(
                turn_log="The debate has not yet begun.",
                last_event="Let's start the discussion now.",
                topic=self.topic,
                turn=0,
                max_turn=self.max_turns,
                silence=True,
                peer_names=peers,
                # ★ 自分の最新 thought のみを渡す
                latest_thoughts=self._get_latest_thought_for(ag.name),
            )
            # 初期planを保存
            self.last_plan_by_agent[ag.name] = self.current_actions[ag.name]
            th0 = self.current_actions[ag.name].get("thought")
            if isinstance(th0, str) and th0.strip():
                self.latest_thought_by_agent[ag.name] = th0.strip()

        # 初期ログ行
        init_record: Dict[str, Any] = {
            "turn": 0,
            "event_type": "plan",
            "speaker": None,
            "content": "",
            "initial_answers": {ag.name: ag.initial_answer for ag in self.agents},
            "agent_actions": [
                {"agent_name": n, "action_plan": p}
                for n, p in self.current_actions.items()
            ],
            # ★ 追加: 早期終了の設定を記録
            "early_stop_config": {
                "enabled": self._early_enabled,
                "require_consecutive": self._req_consec,
                "min_turns": self._min_turns,
            },
            # ★ 追加: consensus snapshot
            "consensus_state": self._build_consensus_state_snapshot(),
            "consensus_meta": self._build_consensus_meta_snapshot(),
        }
        self.log_data.append(init_record)
        self._write_log()
        self._determine_next_speaker(0)

    # ───────────────────── 1ターン処理 ───────────────────── #
    def _run_turn(self, turn: int) -> bool:
        event_type, content, speaker_name = "silence", "", None

        # ---------- 発話フェーズ ----------
        if self.speaker:
            chunk = self.speaker.get_next_chunk()
            if chunk:
                # ★ ここでワンショット割り込みを消費する
                event_type = "interrupt" if self._interrupt_once else "utterance"
                self._interrupt_once = False  # 消費（次発話からは通常の utterance）
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
            "No one has spoken this turn"
            if event_type == "silence"
            else f"Turn {turn}({event_type})\n{speaker_name}:{content}"
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
                # ★ 自分の最新 thought のみを渡す
                latest_thoughts=self._get_latest_thought_for(ag.name),
            )
            # 直近planを更新
            self.last_plan_by_agent[ag.name] = self.current_actions[ag.name]
            th0 = self.current_actions[ag.name].get("thought")
            if isinstance(th0, str) and th0.strip():
                self.latest_thought_by_agent[ag.name] = th0.strip()

        # ---------- ログ ----------
        record: Dict[str, Any] = {
            "turn": turn,
            "event_type": event_type,
            "speaker": speaker_name,
            "content": content,
            "agent_actions": [
                {"agent_name": n, "action_plan": p}
                for n, p in self.current_actions.items()
            ],
            # ★ 追加: 各ターンの consensus snapshot
            "consensus_state": self._build_consensus_state_snapshot(),
            "consensus_meta": self._build_consensus_meta_snapshot(),
        }
        self.log_data.append(record)

        # ---------- 次スピーカー選定 ----------
        if turn < self.max_turns:
            self._determine_next_speaker(turn)

        self._write_log()
        # 早期終了判定
        if self._early_stop_check(turn):
            # 早期終了イベントに、スナップショットも残す
            self.log_data.append(
                {
                    "turn": turn,
                    "event_type": "early_stop",
                    "reason": "consensus",
                    "answer": self._early_stop_answer,
                    "streak": self.consensus_streak,
                    "consensus_state": self._build_consensus_state_snapshot(),
                    "consensus_meta": self._build_consensus_meta_snapshot(),
                }
            )
            self._write_log()
            return True
        return False

    # ──────────────────── Turn-log 生成 ──────────────────── #
    def _build_turn_log(self, agent_name: str, limit: int) -> str:
        lines: List[str] = []
        for e in self.log_data[-limit:]:
            if e["turn"] == 0:
                continue
            if e["event_type"] in {"utterance", "interrupt"}:
                lines.append(f"Turn{e['turn']}({e['event_type']})")
                lines.append(f"{e['speaker']}: {e['content']}")
            elif e["event_type"] == "silence":
                lines.append(f"Turn{e['turn']} (Silence): No one spoke this turn.")
        return "\n".join(lines)

    # ★ 追加: 指定エージェント自身の最新 Thought だけを返す
    def _get_latest_thought_for(self, agent_name: str) -> str:
        th = self.latest_thought_by_agent.get(agent_name, "")
        return th.strip() if isinstance(th, str) and th.strip() else "(none)"

    # ──────────────────── consensus スナップショット ──────────────────── #
    def _build_consensus_state_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """
        各エージェントについて、直近 plan の consensus を抜き出して
        { agent_name: { "agreed": bool, "answer": "A|B|C|D" or None } } を返す
        """
        snap: Dict[str, Dict[str, Any]] = {}
        for ag in self.agents:
            plan = self.last_plan_by_agent.get(ag.name, {})
            c = plan.get("consensus", {}) if isinstance(plan, dict) else {}
            agreed = bool(c.get("agreed", False)) if isinstance(c, dict) else False
            answer = c.get("answer") if isinstance(c, dict) else None
            if isinstance(answer, str):
                answer = answer.strip().upper() or None
            snap[ag.name] = {"agreed": agreed, "answer": answer}
        return snap

    def _build_consensus_meta_snapshot(self) -> Dict[str, Any]:
        snap = self._build_consensus_state_snapshot()
        answers = [v["answer"] for v in snap.values() if v["agreed"] and v["answer"]]
        all_agreed = len(answers) == len(self.agents) and len(set(answers)) == 1
        return {
            "all_agreed": all_agreed,
            "answer_if_all": answers[0] if all_agreed else None,
            "streak": self.consensus_streak,
        }

    # ──────────────────── 早期終了判定 ──────────────────── #
    def _early_stop_check(self, turn: int) -> bool:
        if not self._early_enabled:
            return False
        if turn < self._min_turns:
            self.consensus_streak = 0
            return False

        # 全エージェントの最新planが揃っているか
        plans = [self.last_plan_by_agent.get(a.name, {}) for a in self.agents]
        if any(not p for p in plans):
            self.consensus_streak = 0
            return False

        # consensus.agreed==True かつ answer が全員一致
        answers: List[str] = []
        for p in plans:
            c = p.get("consensus", {})
            if not isinstance(c, dict) or not c.get("agreed", False):
                self.consensus_streak = 0
                return False
            ans = (c.get("answer") or "").strip().upper()
            if ans not in {"A", "B", "C", "D"}:
                self.consensus_streak = 0
                return False
            answers.append(ans)

        if len(set(answers)) == 1:
            self.consensus_streak += 1
            if self.consensus_streak >= self._req_consec:
                self._early_stop_answer = answers[0]
                return True
        else:
            self.consensus_streak = 0
        return False

    # ──────────────────── 最終回答収集 ──────────────────── #
    def _collect_final_answers(self) -> None:
        print("=== Collecting final answers ===")
        debate_history = "\n".join(f"{spk}: {txt}" for spk, txt in self.history[-1000:])
        self.final_answers = {}
        if self._early_stop_answer:
            # 早期終了時は合意解答を全員の回答に採用
            for ag in self.agents:
                self.final_answers[ag.name] = {
                    "answer": self._early_stop_answer,
                    "reason": "Group consensus reached before max turns.",
                }
                print(f"[FINAL] {ag.name} -> {self.final_answers[ag.name]}")
            # 収集後のスナップショットも残す
            self.log_data.append(
                {
                    "turn": "final",
                    "event_type": "final_answers",
                    "answers": self.final_answers,
                    "consensus_state": self._build_consensus_state_snapshot(),
                    "consensus_meta": self._build_consensus_meta_snapshot(),
                }
            )
            self._write_log()
        else:
            # 通常フロー
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
            # 同一話者の継続は割り込みではない
            self._interrupt_once = False
            return

        # 旧スピーカーに残りのチャンクがある場合のみ、次の新スピーカー最初の1発話を「interrupt」扱い
        self._interrupt_once = bool(self.speaker and self.speaker.utterance_queue)
        event_type = "interrupt" if self._interrupt_once else "utterance"
        self.speaker = next(a for a in self.agents if a.name == next_name)
        peers = [a.name for a in self.agents if a is not self.speaker]
        turn_log = self._build_turn_log(self.speaker.name, HISTORY_WINDOW)
        self.speaker.decide_to_speak(
            event_type,
            turn_log,
            self.topic,
            next_plan.get("thought", ""),
            next_plan.get("intent", ""),
            current_turn + 1,
            self.max_turns,
            peer_names=peers,
            # ★ 自分の最新 thought のみを渡す
            latest_thoughts=self._get_latest_thought_for(self.speaker.name),
        )
        mode = "interrupt" if self._interrupt_once else "speak"
        print(f"[Manager] 👉 Next speaker: {self.speaker.name} ({mode})")

    # ──────────────────── JSON 書込み ──────────────────── #
    def _write_log(self) -> None:
        # 親ディレクトリを必ず作成してから書き込み（安全策）
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)
