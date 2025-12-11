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
        self.config = config
        self.topic: str = config["discussion"]["topic"]
        self.max_turns: int = config["discussion"]["max_turns"]
        self._thought_window: int = int(config.get("discussion", {}).get("thought_window", 5))
        
        # 割り込み設定
        self.enable_interruption: bool = config.get("discussion", {}).get("enable_interruption", True)

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
        self.speaker_interrupt = False
        self.final_answers: Dict[str, Dict[str, str]] = {}

        self._interrupt_once: bool = False
        self.log_data: List[Dict[str, Any]] = []

        # ---------- 早期終了用 ----------
        self.last_plan_by_agent: Dict[str, Dict[str, Any]] = {}
        self.consensus_streak: int = 0
        self._early_stop_answer: Optional[str] = None
        self.early_cfg: Dict[str, Any] = config.get("discussion", {}).get("early_stop", {})
        self._early_enabled: bool = bool(self.early_cfg.get("enabled", False))
        self._req_consec: int = int(self.early_cfg.get("require_consecutive", 1))
        self._min_turns: int = int(self.early_cfg.get("min_turns", 1))

        self._write_log()

    # ───────────────────────── 公開 API ───────────────────────── #
    def run_discussion(self) -> Dict[str, Dict[str, str]]:
        print(f"=== Debate Start: {self.topic} ===")
        print(f"=== Mode: Interruption {'Enabled' if self.enable_interruption else 'Disabled'} ===")
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
        # まず全員を normal にリセット
        for ag in self.agents:
            ag.reset_role()

        # 0) 敵対者の設定
        adv_cfg = self.config.get("adversary", {})
        if adv_cfg.get("enabled", False):
            # 名前リストの取得
            target_names = adv_cfg.get("agent_names", [])
            # 互換性: agent_names が空なら agent_name を確認
            if not target_names and "agent_name" in adv_cfg:
                val = adv_cfg["agent_name"]
                if val:
                    target_names = [val]
            
            target_strategy = adv_cfg.get("target_strategy", "random_wrong")
            fixed_label = adv_cfg.get("fixed_label", "D")

            # 全敵対者で共通の誤答ターゲットを決定
            final_target = fixed_label
            if target_strategy == "random_wrong":
                final_target = random.choice(["A", "B", "C", "D"])
            
            print(f"[System] Adversary Strategy: {target_strategy}, Target Answer: {final_target}")
            print(f"[System] Targeted Agents: {target_names}")

            for ag in self.agents:
                if ag.name in target_names:
                    ag.set_adversary(final_target)
                    print(f"[System] Agent {ag.name} is set as ADVERSARY.")
                else:
                    print(f"[System] Agent {ag.name} is set as NORMAL.")

        # 1) 初回回答
        for ag in self.agents:
            ag.generate_initial_answer(self.topic, self.max_turns, [p.name for p in self.agents if p is not ag])
            print(f"[Init] {ag.name} → {ag.initial_answer_str}")

        # 2) 全初回回答を共有
        all_initial = "\n".join(
            f"Name: {ag.name},\nAnswer: {ag.initial_answer.get('answer','')},\nreason: {ag.initial_answer.get('reason','') }"
            "\n"
            for ag in self.agents
        )
        for ag in self.agents:
            ag.all_initial_answers_str = all_initial

        # 3) ターン0の行動計画
        self.current_actions.clear()
        for ag in self.agents:
            peers = [p.name for p in self.agents if p is not ag]
            # Turn 0 は全員計画に参加
            self.current_actions[ag.name] = ag.plan_action(
                turn_log="The debate has not yet begun.",
                last_event="Let's start the discussion now.",
                topic=self.topic,
                turn=0,
                max_turn=self.max_turns,
                silence=True,
                peer_names=peers,
                latest_thoughts=self.__format_recent_thoughts(ag.name, current_turn=0),
                allow_interruption=self.enable_interruption
            )
            self.last_plan_by_agent[ag.name] = self.current_actions[ag.name]
            self.__trim_thoughts(ag)

        # 初期ログ
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
            "early_stop_config": {
                "enabled": self._early_enabled,
                "require_consecutive": self._req_consec,
                "min_turns": self._min_turns,
            },
            "roles": {
                ag.name: ag.role
                for ag in self.agents
            },
            "consensus_state": self._build_consensus_state_snapshot(),
            "consensus_meta": self._build_consensus_meta_snapshot(),
        }
        self.log_data.append(init_record)
        self._write_log()
        self._determine_next_speaker(0)

    # ───────────────────── 1ターン処理 ───────────────────── #
    def _run_turn(self, turn: int) -> bool:
        event_type, content, speaker_name = "silence", "", None
        
        # --- 発話処理フェーズ ---
        if not self.enable_interruption:
            # === 割り込みなしモード ===
            if self.speaker:
                chunks = []
                while True:
                    c = self.speaker.get_next_chunk()
                    if c is None:
                        break
                    chunks.append(c)
                full_content = " ".join(chunks)
                if full_content:
                    speaker_name = self.speaker.name
                    content = full_content
                    event_type = "utterance"
                    print(f"[Turn {turn}] {speaker_name}: {content}")
                    self.history.append((speaker_name, content))
                self.speaker = None
            else:
                print(f"[Turn {turn}] --- Silence ---")
                event_type = "silence"
        else:
            # === 割り込みありモード ===
            if self.speaker:
                chunk = self.speaker.get_next_chunk()
                if chunk:
                    event_type = "interrupt" if self._interrupt_once else "utterance"
                    self._interrupt_once = False
                    speaker_name = self.speaker.name
                    content = chunk
                    if self.history and self.history[-1][0] == speaker_name:
                        print(f"[Turn {turn}] {chunk}")
                    else:
                        print(f"[Turn {turn}] {speaker_name}: {chunk}")
                    self.history.append((speaker_name, chunk))
                else:
                    self.speaker = None
            
            if event_type == "silence" and not self.speaker:
                print(f"[Turn {turn}] --- Silence ---")

        # ログ保存用の一時辞書 (Actionはまだ)
        record: Dict[str, Any] = {
            "turn": turn,
            "event_type": event_type,
            "speaker": speaker_name,
            "content": content,
            "agent_actions": [], 
            "consensus_state": {}, 
            "consensus_meta": {}
        }

        # --- 行動計画フェーズ ---
        self.current_actions.clear()
        last_event = (
            "No one has spoken this turn"
            if event_type == "silence"
            else f"Turn {turn}({event_type})\n{speaker_name}:{content}"
        )
        
        for ag in self.agents:
            # 発言者(Speaker)はそのターンでの計画から除外する
            # ただし Silence ターンの場合(speaker_name=None)は全員参加する
            if event_type != "silence" and ag.name == speaker_name:
                continue
                
            peers = [p.name for p in self.agents if p is not ag]
            turn_log = self._build_turn_log(ag.name, HISTORY_WINDOW)
            
            # 割り込みなしモードなら常に silence=True扱い & 割り込み禁止プロンプト
            is_silence_mode = True if not self.enable_interruption else (event_type == "silence")
            allow_int = self.enable_interruption

            self.current_actions[ag.name] = ag.plan_action(
                turn_log,
                last_event,
                self.topic,
                turn,
                self.max_turns,
                silence=is_silence_mode,
                peer_names=peers,
                latest_thoughts=self.__format_recent_thoughts(ag.name, current_turn=turn),
                allow_interruption=allow_int
            )
            self.last_plan_by_agent[ag.name] = self.current_actions[ag.name]
            self.__trim_thoughts(ag)

        # ログ更新
        record["agent_actions"] = [
            {"agent_name": n, "action_plan": p}
            for n, p in self.current_actions.items()
        ]
        record["consensus_state"] = self._build_consensus_state_snapshot()
        record["consensus_meta"] = self._build_consensus_meta_snapshot()
        self.log_data.append(record)
        
        # 次の話者決定
        if turn < self.max_turns:
            self._determine_next_speaker(turn)
        
        self._write_log()

        if self._early_stop_check(turn):
            self.log_data.append({
                "turn": turn,
                "event_type": "early_stop",
                "reason": "consensus",
                "answer": self._early_stop_answer,
                "streak": self.consensus_streak,
                "consensus_state": self._build_consensus_state_snapshot(),
                "consensus_meta": self._build_consensus_meta_snapshot(),
            })
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

    # ──────────────────── consensus スナップショット ──────────────────── #
    def _build_consensus_state_snapshot(self) -> Dict[str, Dict[str, Any]]:
        snap: Dict[str, Dict[str, Any]] = {}
        for ag in self.agents:
            plan = self.last_plan_by_agent.get(ag.name, {}) or {}
            consensus, answer = self._extract_agreement(plan)
            snap[ag.name] = {"consensus": consensus, "answer": answer}
        return snap

    def _build_consensus_meta_snapshot(self) -> Dict[str, Any]:
        snap = self._build_consensus_state_snapshot()
        answers = [v["answer"] for v in snap.values() if v["consensus"] and v["answer"]]
        all_consensus = len(answers) == len(self.agents) and len(set(answers)) == 1
        return {
            "all_consensus": all_consensus,
            "answer_if_all": answers[0] if all_consensus else None,
            "streak": self.consensus_streak,
        }

    def _extract_agreement(self, plan: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        consensus = False
        answer_val: Any = None
        if isinstance(plan, dict) and "consensus" in plan and isinstance(plan.get("consensus"), dict):
            c = plan["consensus"]
            consensus = bool(c.get("consensus", False))
            answer_val = c.get("answer")
        else:
            consensus = bool(plan.get("consensus", False)) if isinstance(plan, dict) else False
            answer_val = plan.get("answer") if isinstance(plan, dict) else None

        if isinstance(answer_val, str):
            ans = answer_val.strip().upper()
            if ans in {"A", "B", "C", "D"}:
                return consensus, ans
        return consensus, None

    # ──────────────────── 早期終了判定 ──────────────────── #
    def _early_stop_check(self, turn: int) -> bool:
        if not self._early_enabled:
            return False
        if turn < self._min_turns:
            self.consensus_streak = 0
            return False

        plans = [self.last_plan_by_agent.get(a.name, {}) for a in self.agents]
        if any(not p for p in plans):
            self.consensus_streak = 0
            return False

        answers: List[str] = []
        for p in plans:
            consensus, ans = self._extract_agreement(p if isinstance(p, dict) else {})
            if not consensus or ans is None:
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
        debate_history = self._build_turn_log("", HISTORY_WINDOW * 10)
        self.final_answers = {}
        if self._early_stop_answer:
            for ag in self.agents:
                self.final_answers[ag.name] = {
                    "answer": self._early_stop_answer,
                    "reason": "Group consensus reached before max turns.",
                }
                print(f"[FINAL] {ag.name} -> {self.final_answers[ag.name]}")
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
            for ag in self.agents:
                ans = ag.generate_final_answer(
                    self.topic,
                    debate_history,
                    latest_thoughts=self.__format_recent_thoughts(ag.name, current_turn=self.max_turns + 1),
                    max_turn=self.max_turns,
                    peer_names=[p.name for p in self.agents if p is not ag],
                )
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

    # ──────────────────── 次スピーカー選定 ──────────────────── #
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
        
        # 修正: 同点の urgency を持つ候補者のリストをシャッフルしてから選択する
        # これにより、リストの先頭にあるエージェント（例: Alex）が優先されるバイアスを防ぐ
        random.shuffle(top)
        next_name, next_plan = random.choice(top)

        if self.speaker and self.speaker.name == next_name:
            self._interrupt_once = False
            return

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
            next_plan.get("purpose", ""),
            current_turn + 1,
            self.max_turns,
            peer_names=peers,
            latest_thoughts=self.__format_recent_thoughts(self.speaker.name, current_turn=current_turn + 1),
        )
        mode = "interrupt" if self._interrupt_once else "speak"
        print(f"[Manager] 👉 Next speaker: {self.speaker.name} ({mode})")

    # ──────────────────── JSON 書込み ──────────────────── #
    def _write_log(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)

    # ──────────────────── thought の整形/保持 ──────────────────── #
    def __trim_thoughts(self, ag: Agent) -> None:
        try:
            k = self._thought_window
            if k <= 0:
                return
            if len(ag.thought_history) > k:
                ag.thought_history[:] = ag.thought_history[-k:]
        except Exception:
            pass

    def __format_recent_thoughts(self, agent_name: str, current_turn: int) -> str:
        try:
            ag = next(a for a in self.agents if a.name == agent_name)
        except StopIteration:
            return "(none)"

        hist = [
            (t, txt)
            for (t, txt) in ag.thought_history
            if isinstance(t, int) and t < current_turn and isinstance(txt, str) and txt.strip()
        ]
        if not hist:
            return "(none)"

        k = max(1, self._thought_window)
        subset = hist[-k:]
        lines = []
        for t, txt in subset:
            lines.append(f"Turn {t}\nThought:{txt.strip()}")
        return "\n".join(lines)