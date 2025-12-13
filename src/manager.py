"""議論全体を統括する DiscussionManager（固定順序・割り込みなし・各人3回）"""
from __future__ import annotations

import json
import random
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
        self.config = config

        # 固定順序設定
        order_cfg = config.get("discussion", {}).get("speaking_order")
        if order_cfg and isinstance(order_cfg, list) and all(
            isinstance(n, str) for n in order_cfg
        ):
            self.order: List[str] = list(order_cfg)
        else:
            self.order = [a.name for a in agents]

        self.turns_per_agent: int = int(
            config.get("discussion", {}).get("turns_per_agent", 3)
        )
        self.max_turns: int = len(self.order) * self.turns_per_agent

        # name -> Agent
        self._agent_by_name = {a.name: a for a in self.agents}
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

        # 早期終了用
        self.early_stop_answer: Optional[str] = None
        self.early_stop_turn: Optional[int] = None
        self.early_stop_states: Optional[List[Dict[str, Any]]] = None

        self._write_log()  # 空配列でファイルを作成

    # ───────────────────────── 公開 API ───────────────────────── #
    def run_discussion(self) -> Dict[str, Dict[str, str]]:
        print(f"=== Debate Start (Fixed Order): {self.topic} ===")
        self._initialize_discussion()

        turn = 0
        for _r in range(1, self.turns_per_agent + 1):
            for name in self.order:
                turn += 1
                self._run_fixed_turn(turn, speaker_name=name)

                if self.early_stop_answer is not None:
                    print(
                        f"=== Early consensus reached at turn {turn}: "
                        f"answer={self.early_stop_answer} ==="
                    )
                    break
            if self.early_stop_answer is not None:
                break

        print("=== Debate End ===")
        self._collect_final_answers()
        return self.final_answers

    # ───────────────────────── 初期化 ───────────────────────── #
    def _initialize_discussion(self) -> None:
        # まず全員を normal にリセット
        for ag in self.agents:
            ag.role = "normal"
            ag.adversary_target = None

        # 0) adversary 設定
        adv_cfg = self.config.get("adversary", {})
        if adv_cfg.get("enabled", False):
            # 名前リストの取得
            target_names = adv_cfg.get("agent_names", [])
            # 互換性: agent_names が空なら agent_name を確認
            if not target_names and "agent_name" in adv_cfg:
                val = adv_cfg["agent_name"]
                if val:
                    target_names = [val]
            
            strategy = adv_cfg.get("target_strategy", "fixed")
            fixed_label = (adv_cfg.get("fixed_label") or "D").strip().upper()

            # 全敵対者で共通の誤答ターゲットを決定
            # とりあえず 4 択想定で候補ラベルを作る
            valid_labels = ["A", "B", "C", "D"]
            
            final_target = fixed_label
            if strategy == "random_wrong":
                # 正解が不明なので便宜的にランダム選択
                final_target = random.choice(valid_labels)
            
            if final_target not in valid_labels:
                final_target = "D"

            print(f"[System] Adversary Strategy: {strategy}, Target Answer: {final_target}")
            print(f"[System] Targeted Agents: {target_names}")

            for ag in self.agents:
                if ag.name in target_names:
                    ag.set_adversary(final_target)
                    print(f"[System] Agent {ag.name} is set as ADVERSARY.")
                else:
                    print(f"[System] Agent {ag.name} is set as NORMAL.")

        # 1) Peer 情報の登録
        all_names = [a.name for a in self.agents]
        for ag in self.agents:
            ag.set_peers(all_names)

        # 2) 初回回答
        for ag in self.agents:
            peer_names = [n for n in all_names if n != ag.name]
            ag.generate_initial_answer(self.topic, self.max_turns, peer_names)
            print(
                f"[Init] {ag.name} → Answer={ag.initial_answer.get('answer','')}, "
                f"Reason={ag.initial_answer.get('reason','')}"
            )

        # 3) 全初回回答を共有
        all_initial = "\n".join(
            f"Name: {ag.name},\nAnswer: {ag.initial_answer.get('answer','')},\nreason: {ag.initial_answer.get('reason','') }"
            "\n"
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
                "adversary": adv_cfg,
            },
            "roles": {ag.name: ag.role for ag in self.agents},
        }
        self.log_data.append(init_record)
        self._write_log()


    # ───────────────────── 固定順序の1ターン処理 ───────────────────── #
    def _run_fixed_turn(self, turn: int, speaker_name: str) -> None:
        speaker = self._agent_by_name[speaker_name]

        # 発言者向け turn_log（直近の発話のみ）
        turn_log_for_speaker = self._build_turn_log(limit=HISTORY_WINDOW)

        turns_left_for_agent = self._turns_left_of_agent_after_this_turn(
            speaker_name, turn
        )
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

        # 非発言者の thought / current_answer / consensus を取得
        listener_thoughts: List[Dict[str, Any]] = []
        for ag in self.agents:
            if ag is speaker:
                continue
            thought_info = ag.think_only(
                topic=self.topic,
                turn_log=self._build_turn_log(limit=HISTORY_WINDOW),
                turn=turn,
                max_turn=self.max_turns,
            )
            listener_thoughts.append(
                {
                    "agent_name": ag.name,
                    "thought": thought_info.get("thought", ""),
                    "answer": thought_info.get("answer", ""),
                    "consensus": bool(thought_info.get("consensus", False)),
                }
            )

        # 全エージェントの「最新の」状態を thought_history から取得
        agent_states: List[Dict[str, Any]] = []
        for ag in self.agents:
            if ag.thought_history:
                last_turn, thought, current_answer, consensus = ag.thought_history[-1]
            else:
                # 修正: thought_history が空の場合は初期回答を使用する
                thought = ""  # 発言者の思考は表出しないので空でOK
                current_answer = ag.initial_answer.get("answer", "")
                consensus = False

            agent_states.append(
                {
                    "agent_name": ag.name,
                    # "thought": thought,  # 必要なら有効化
                    "answer": current_answer,
                    "consensus": consensus,
                }
            )

        # 合意判定（全員 consensus==True かつ current_answer が一致）
        consensus_all_true = bool(agent_states) and all(
            st["consensus"] for st in agent_states
        )
        consensus_answer: Optional[str] = None
        if consensus_all_true:
            answers = {st["answer"] for st in agent_states if st["answer"]}
            if len(answers) == 1:
                only_ans = next(iter(answers))
                if only_ans in {"A", "B", "C", "D"}:
                    consensus_answer = only_ans
                    self.early_stop_answer = only_ans
                    self.early_stop_turn = turn
                    self.early_stop_states = agent_states
                    print(
                        f"[Consensus] Early stop triggered at turn {turn}, "
                        f"answer={only_ans}"
                    )

        record: Dict[str, Any] = {
            "turn": turn,
            "event_type": "utterance",
            "speaker": speaker_name,
            "content": utterance,
            "listener_thoughts": listener_thoughts,
            "agent_states": agent_states,
            "consensus_all_true": consensus_all_true,
            "consensus_answer": consensus_answer,
        }
        if self.early_stop_answer is not None and self.early_stop_turn == turn:
            record["early_stop"] = True
        self.log_data.append(record)
        self._write_log()

    # ──────────────────── Turn-log（発話履歴） ──────────────────── #
    def _build_turn_log(self, limit: int) -> str:
        lines: List[str] = []
        for i, (spk, txt) in enumerate(self.history[-limit:], start=1):
            lines.append(f"Turn{i}")
            lines.append(f"{spk}: {txt}")
        return "\n".join(lines)

    def _turns_left_of_agent_after_this_turn(self, agent_name: str, turn: int) -> int:
        idx_in_order = self.order.index(agent_name)
        per_round = len(self.order)
        spoken_count = 0
        for t in range(1, turn + 1):
            pos = (t - 1) % per_round
            if self.order[pos] == agent_name:
                spoken_count += 1
        total_allowed = self.turns_per_agent
        return max(total_allowed - spoken_count, 0)

    # ──────────────────── 最終回答収集 ──────────────────── #
    def _collect_final_answers(self) -> None:
        print("=== Collecting final answers ===")

        # トークン使用量の取得（Handlerはシングルトン的に共有されている想定）
        token_usage = {}
        if self.agents and self.agents[0].llm_handler:
            token_usage = self.agents[0].llm_handler.total_token_usage
            print(f"[Usage] Total Tokens: {token_usage}")

        if self.early_stop_answer is not None and self.early_stop_states is not None:
            self.final_answers = {}
            for ag in self.agents:
                st = next(
                    (s for s in self.early_stop_states if s["agent_name"] == ag.name),
                    None,
                )
                reason = ""
                if st is not None:
                    reason = st.get("thought", "")  # thought は上でコメントアウトしているので基本 ""
                self.final_answers[ag.name] = {
                    "answer": self.early_stop_answer,
                    "reason": reason,
                }
                print(f"[FINAL/EARLY] {ag.name} -> {self.final_answers[ag.name]}")

            self.log_data.append(
                {
                    "turn": "final",
                    "event_type": "final_answers",
                    "answers": self.final_answers,
                    "early_stop": True,
                    "early_stop_turn": self.early_stop_turn,
                    "early_stop_answer": self.early_stop_answer,
                    "token_usage": token_usage,  # ログに追加
                }
            )
            self._write_log()
            return

        # 通常ケース
        debate_history = "\n".join(
            f"Turn{i} \n {spk}: {txt}"
            for i, (spk, txt) in enumerate(self.history[-1000:], start=1)
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
                "early_stop": False,
                "token_usage": token_usage, # ログに追加
            }
        )
        self._write_log()

    # ──────────────────── JSON 書込み ──────────────────── #
    def _write_log(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)