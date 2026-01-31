#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
複数 run の「実行ログ（event_type/agent_actions を持つ JSON 配列）」を比較し、
元の analyze_runs.py と同等の分析を、turn ではなく token（public_tokens_used）軸で行う。

主な仕様
- 対象シナリオ（初期回答 initial_answers の正誤構成）を抽出
  * デフォルト: 「二人が誤答・一人が正答」
  * --scenario two_correct_one_wrong で「二人正答・一人誤答」に切替可能

- agent_actions[*].action_plan["answer"] を「途中回答」として利用
  * ある turn で agent_actions に登場しないエージェントは、直前の既知回答を引き継ぐ（forward-fill）

- token ごとの多数決正解率（全 run / 共通シナリオ）を可視化
- token ごとの各エージェント途中回答正解率（1枚の図に3本、凡例付き）を run ごとに出力
- 行動選択分布（action_plan.action）の分布を run ごとに出力
- interrupt 推定（ログの event_type が utterance のままでも interrupt とみなす）
  * 前ターンで action_plan.action == "interrupt" を選んだエージェントがいて、
    次ターンで speaker がそのエージェントに切り替わっていたら、
    その次ターンを interrupt として扱う（event_type_fixed="interrupt"）
  * これは「実現された発話種別（speak/interrupt）分布」集計にも反映

入出力（pair 出力ディレクトリ配下）
- token_accuracy_runwise.png    : 各 run の「各自シナリオ集合」での token-wise 多数決正解率
- token_accuracy_common.png     : 共通シナリオ集合での token-wise 多数決正解率
- final_accuracy_bar.png        : 最終正解率（各自シナリオ / 共通シナリオ）バー
- per_agent_token_accuracy_<run>.png : token-wise 各エージェント正解率（1図）
- action_distribution_<run>.png      : action_plan.action 分布（1図）
- realized_speaker_distribution_<run>.png : 実現発話（speak/interrupt）分布（1図）
- analysis_results.json         : 集計結果（JSON）

使い方
  python analyze_runs_exec_log.py RUN1 RUN2 [RUN3 ...]
  python analyze_runs_exec_log.py RUN1 RUN2 --max-problem-index 200
  python analyze_runs_exec_log.py RUN1 RUN2 --scenario two_correct_one_wrong
"""

import argparse
import json
import os
import re
from glob import glob
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Optional
import random

import matplotlib.pyplot as plt



# =========================
# ユーティリティ
# =========================

def resolve_run_dir(run_arg: str) -> str:
    """引数がそのままディレクトリならそれを、そうでなければ logs/<arg> を探す。"""
    if os.path.isdir(run_arg):
        return os.path.abspath(run_arg)
    candidate = os.path.join("logs", run_arg)
    if os.path.isdir(candidate):
        return os.path.abspath(candidate)
    raise FileNotFoundError(f"Run directory not found: {run_arg} or logs/{run_arg}")


def find_accuracy_file(run_dir: str) -> Optional[str]:
    """run_dir 内の accuracy_log を探す。jsonl 優先、なければ json。"""
    candidates = [
        os.path.join(run_dir, "accuracy_log.jsonl"),
        os.path.join(run_dir, "accuracy_log.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def load_accuracy(run_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    accuracy_log を読み込み、problem_id -> 情報 の dict を返す。

    想定フォーマット例:
        {"question_id": 1, "pred": "B", "gold": "B", "correct": true}

    question_id=1 の場合は:
      - "1"
      - "problem_001"
    の両方のキーで参照できるようにする。
    """
    acc_path = find_accuracy_file(run_dir)
    if acc_path is None:
        raise FileNotFoundError(f"accuracy_log not found in {run_dir}")

    data: List[Dict[str, Any]] = []
    if acc_path.endswith(".jsonl"):
        with open(acc_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
    else:
        with open(acc_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                for pid, info in loaded.items():
                    info = dict(info)
                    info.setdefault("problem_id", pid)
                    data.append(info)
            elif isinstance(loaded, list):
                data = loaded
            else:
                raise ValueError(f"Unsupported accuracy_log format: {acc_path}")

    def get_field(d: Dict[str, Any], candidates: List[str], default=None):
        for k in candidates:
            if k in d:
                return d[k]
        return default

    acc_by_pid: Dict[str, Dict[str, Any]] = {}

    for row in data:
        pid_raw = get_field(row, ["problem_id", "qid", "id", "question_id"])
        if pid_raw is None:
            continue

        gold = get_field(row, ["gold", "gold_answer", "label", "answer"])
        final_answer = get_field(row, ["final_answer", "pred", "model_answer"])
        is_correct = get_field(row, ["is_correct", "correct"])

        entry = {
            "gold": gold,
            "final_answer": final_answer,
            "is_correct": bool(is_correct) if is_correct is not None and gold is not None and final_answer is not None
            else (final_answer == gold if gold is not None and final_answer is not None else None),
        }

        # --- ID 正規化 ---
        keys_for_this: List[str] = []
        if isinstance(pid_raw, int):
            idx = pid_raw
            keys_for_this.append(str(idx))
            keys_for_this.append(f"problem_{idx:03d}")
        else:
            s = str(pid_raw)
            keys_for_this.append(s)
            m = re.match(r"^problem_(\d+)$", s)
            if m:
                idx = int(m.group(1))
                keys_for_this.append(str(idx))
                keys_for_this.append(f"problem_{idx:03d}")
            else:
                if s.isdigit():
                    idx = int(s)
                    keys_for_this.append(f"problem_{idx:03d}")

        for k in set(keys_for_this):
            acc_by_pid[k] = entry

    return acc_by_pid


def pid_to_index(pid: str) -> Optional[int]:
    """'problem_001' -> 1"""
    try:
        base = os.path.basename(pid)
        if "_" not in base:
            return None
        num_str = base.split("_")[-1]
        return int(num_str)
    except Exception:
        return None


def filter_problems_by_max_index(
    problems: Dict[str, Dict[str, Any]],
    acc_by_pid: Dict[str, Dict[str, Any]],
    max_index: Optional[int],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """max_index が指定されていれば、problem_001〜problem_N のみ残す。"""
    if max_index is None:
        return problems, acc_by_pid

    allowed_pids = []
    for pid in problems.keys():
        idx = pid_to_index(pid)
        if idx is None:
            continue
        if idx <= max_index:
            allowed_pids.append(pid)

    allowed_set = set(allowed_pids)
    new_problems = {pid: problems[pid] for pid in problems.keys() if pid in allowed_set}
    new_acc = {pid: info for pid, info in acc_by_pid.items() if pid in allowed_set}

    return new_problems, new_acc


def majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    """単純多数決。同票なら None。"""
    filtered = [a for a in answers if isinstance(a, str) and a.strip()]
    if not filtered:
        return None
    counts = Counter(filtered)
    most_common = counts.most_common()
    if len(most_common) == 1:
        return most_common[0][0]
    if most_common[0][1] == most_common[1][1]:
        return None
    return most_common[0][0]


def majority_with_random_tie(answers: List[Optional[str]], rng: "random.Random") -> Optional[str]:
    """多数決。同票の場合は一様ランダムに1つ選ぶ。"""
    filtered = [a for a in answers if isinstance(a, str) and a.strip()]
    if not filtered:
        return None
    counts = Counter(filtered)
    most = counts.most_common()
    top_freq = most[0][1]
    tied = [a for a, c in most if c == top_freq]
    return rng.choice(tied)


def scenario_list_with_index(pids: List[str]) -> List[Dict[str, Any]]:
    items = []
    for pid in pids:
        idx = pid_to_index(pid)
        items.append({"pid": pid, "index": idx})
    items.sort(key=lambda x: (x["index"] is None, x["index"] if x["index"] is not None else 0))
    return items


def normalize_turn(turn: Any) -> int:
    """
    turn をソート可能な int に正規化する。
    - int はそのまま
    - 数字文字列は int 化
    - それ以外（"final" 等）は十分大きい値に飛ばす
    """
    if isinstance(turn, int):
        return turn
    if isinstance(turn, str) and turn.isdigit():
        return int(turn)
    return 10**9


# =========================
# 実行ログ（新形式）の読み込み
# =========================

def _looks_like_exec_log(obj: Any) -> bool:
    if not isinstance(obj, list) or not obj:
        return False
    head = obj[0]
    if not isinstance(head, dict):
        return False
    return ("event_type" in head) and ("turn" in head)


def find_exec_log_file(problem_dir: str) -> Optional[str]:
    """
    problem_dir 内の実行ログファイルを探す。
    優先候補 -> なければ *.json を軽く判定。
    """
    candidates = [
        os.path.join(problem_dir, "execution_log.json"),
        os.path.join(problem_dir, "event_log.json"),
        os.path.join(problem_dir, "run_log.json"),
        os.path.join(problem_dir, "discussion_log.json"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    for p in sorted(glob(os.path.join(problem_dir, "*.json"))):
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if _looks_like_exec_log(obj):
                return p
        except Exception:
            continue
    return None


def _norm_action(a: Any) -> Optional[str]:
    if not isinstance(a, str):
        return None
    s = a.strip().lower()
    # "interrupr" のようなタイポも吸収
    if s.startswith("interrup"):
        return "interrupt"
    return s


def load_exec_logs(run_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    run_dir/problem_*/ 以下の実行ログを読み込み、
    pid -> {
      "agents": [..],
      "initial_answers": {agent: ans},
      "token_budget": int|None,
      "snapshots": [{"turn":..,"turn_sort":..,"tokens":..,"answers":{..},"speaker":..,"event_type_fixed":..}],
      "action_dist": {agent: Counter(action_plan.action)},                               # 全イベント
      "action_dist_by_ctx": {ctx: {agent: Counter}}  (ctx ∈ {"talk","silence"})          # イベント別
      "realized_speaker_dist": {agent: Counter({"speak":x,"interrupt":y})},
      "final_answers": {agent: answer}  # ログ最終行から取得できる場合のみ
    }
    """
    result: Dict[str, Dict[str, Any]] = {}
    is_roundrobin = os.path.basename(run_dir).endswith("_roundrobin")
    problem_dirs = sorted(glob(os.path.join(run_dir, "problem_*")))

    for pdir in problem_dirs:
        pid = os.path.basename(pdir)
        log_path = find_exec_log_file(pdir)
        if log_path is None:
            continue

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {log_path}: {e}")
            continue

        if not _looks_like_exec_log(records):
            continue

        initial_answers: Dict[str, str] = {}
        token_budget: Optional[int] = None

        # 初期回答を探す（turn==0 かつ initial_answers あり）
        for rec in records:
            if not isinstance(rec, dict):
                continue
            if rec.get("initial_answers") and isinstance(rec.get("initial_answers"), dict):
                ia = rec["initial_answers"]
                for agent, info in ia.items():
                    if isinstance(info, dict):
                        ans = info.get("answer")
                        if isinstance(ans, str):
                            initial_answers[agent] = ans.strip()
                tb = rec.get("public_token_budget")
                if isinstance(tb, int):
                    token_budget = tb
                break

        if not initial_answers:
            continue

        agents = sorted(initial_answers.keys())

        # 行動分布（roundrobin では利用しない）
        action_dist: Optional[Dict[str, Counter]] = {a: Counter() for a in agents} if not is_roundrobin else None
        action_dist_by_ctx: Optional[Dict[str, Dict[str, Counter]]] = (
            {ctx: {a: Counter() for a in agents} for ctx in ["talk", "silence"]} if not is_roundrobin else None
        )
        # 実現された発話分布（roundrobin では利用しない）
        realized_speaker_dist: Optional[Dict[str, Counter]] = {a: Counter() for a in agents} if not is_roundrobin else None

        # forward-fill 用の回答状態
        current_answers: Dict[str, str] = dict(initial_answers)

        snapshots: List[Dict[str, Any]] = []

        prev_speaker: Optional[str] = None
        prev_interrupters: set = set()

        # 「この turn で speak/interrupt を選んだエージェント」を次 turn の speaker と対応付ける
        prev_planned_speaker_actions: Dict[str, str] = {}

        final_answers_map: Dict[str, str] = {}

        for rec in records:
            if not isinstance(rec, dict):
                continue

            event_type = rec.get("event_type")
            speaker = rec.get("speaker")
            turn = rec.get("turn")
            turn_sort = normalize_turn(turn)

            # final_answers レコードは最後に出るので先に拾っておく
            if event_type == "final_answers":
                ans_map = rec.get("answers")
                if isinstance(ans_map, dict):
                    for ag, ans in ans_map.items():
                        if isinstance(ans, str):
                            final_answers_map[ag] = ans.strip()
                continue
            

            # token 使用量（累積）
            tokens_budget = rec.get("public_token_budget")
            tokens_left = rec.get("public_tokens_left")
            tokens_used = tokens_budget - tokens_left if (isinstance(tokens_budget, int) and isinstance(tokens_left, int)) else None

            if tokens_used is None:
                if turn == 0:
                    tokens_used = 0
            if not isinstance(tokens_used, int):
                tokens_used = None

            # interrupt 推定を先に行う（ctx 判定にも使う）
            event_type_fixed = event_type
            if event_type == "utterance" and isinstance(speaker, str):
                if speaker in prev_interrupters and speaker != prev_speaker:
                    event_type_fixed = "interrupt"

            # イベント種別で行動分布の文脈を決める（event_type_fixed を使用）
            ctx_for_action: Optional[str] = None
            if event_type_fixed == "silence" or event_type == "silence":
                ctx_for_action = "silence"
            elif event_type_fixed in ("utterance", "interrupt"):
                ctx_for_action = "talk"

            # この turn の agent_actions を処理
            planned_speaker_actions: Dict[str, str] = {}

            if is_roundrobin:
                # roundrobin: 回答更新は agent_states を見る。行動分布は集計しない。
                agent_states = rec.get("agent_states")
                if isinstance(agent_states, list):
                    for st in agent_states:
                        if not isinstance(st, dict):
                            continue
                        agent = st.get("agent_name")
                        ans = st.get("answer")
                        if agent in agents and isinstance(ans, str):
                            current_answers[agent] = ans.strip()
            else:
                agent_actions = rec.get("agent_actions")
                # 仕様変更: public_tokens_used が 0 の間は initial_answers を保持し、agent_actions の answer は無視
                freeze_answers = (tokens_used == 0)
                if isinstance(agent_actions, list):
                    for aa in agent_actions:
                        if not isinstance(aa, dict):
                            continue
                        agent = aa.get("agent_name")
                        ap = aa.get("action_plan")
                        if agent not in agents or not isinstance(ap, dict):
                            continue

                        act = _norm_action(ap.get("action"))
                        ans = ap.get("answer")
                        if (not freeze_answers) and isinstance(ans, str):
                            current_answers[agent] = ans.strip()
                        if act and action_dist is not None:
                            action_dist[agent][act] += 1
                            if ctx_for_action and action_dist_by_ctx is not None and agent in action_dist_by_ctx[ctx_for_action]:
                                action_dist_by_ctx[ctx_for_action][agent][act] += 1
                            if act in ("speak", "interrupt"):
                                planned_speaker_actions[agent] = act

            # speaker 実現分布（roundrobinでは集計しない）
            if not is_roundrobin and realized_speaker_dist is not None:
                if isinstance(speaker, str) and speaker in agents:
                    if speaker in prev_planned_speaker_actions:
                        realized = prev_planned_speaker_actions[speaker]
                        if event_type_fixed == "interrupt":
                            realized = "interrupt"
                        realized_speaker_dist[speaker][realized] += 1
                    else:
                        if event_type_fixed == "interrupt":
                            realized_speaker_dist[speaker]["interrupt"] += 1
                        elif event_type == "utterance":
                            realized_speaker_dist[speaker]["speak"] += 1

            # tokens_used が取れた地点のみ snapshot を保存
            if tokens_used is not None:
                snapshots.append({
                    "turn": turn,
                    "turn_sort": turn_sort,  # ★ int 正規化
                    "tokens": tokens_used,
                    "answers": dict(current_answers),
                    "speaker": speaker,
                    "event_type_fixed": event_type_fixed,
                })

            # 次 turn 用更新
            prev_speaker = speaker if isinstance(speaker, str) else prev_speaker
            prev_interrupters = set()
            if (not is_roundrobin) and isinstance(rec.get("agent_actions"), list):
                for aa in rec.get("agent_actions"):
                    if not isinstance(aa, dict):
                        continue
                    agent = aa.get("agent_name")
                    ap = aa.get("action_plan")
                    if agent not in agents or not isinstance(ap, dict):
                        continue
                    act = _norm_action(ap.get("action"))
                    if act == "interrupt":
                        prev_interrupters.add(agent)

            prev_planned_speaker_actions = planned_speaker_actions

            tb = rec.get("public_token_budget")
            if token_budget is None and isinstance(tb, int):
                token_budget = tb

        # snapshots が空なら最低限 token=0 を入れる
        if not snapshots:
            snapshots = [{
                "turn": 0,
                "turn_sort": 0,
                "tokens": 0,
                "answers": dict(initial_answers),
                "speaker": None,
                "event_type_fixed": "plan",
            }]

        # token 昇順に整列（turn は turn_sort を使う）
        snapshots_sorted = sorted(
            snapshots,
            key=lambda x: (x.get("tokens", 0), x.get("turn_sort", 10**9))
        )

        # 同 token の重複は「後勝ち」（最後の状態を採用）
        dedup: Dict[int, Dict[str, Any]] = {}
        for s in snapshots_sorted:
            t = s["tokens"]
            dedup[t] = s
        snapshots_sorted = [dedup[t] for t in sorted(dedup.keys())]

        result[pid] = {
            "agents": agents,
            "initial_answers": initial_answers,
            "token_budget": token_budget,
            "snapshots": snapshots_sorted,
            "action_dist": {a: dict(action_dist[a]) for a in agents} if action_dist is not None else {},
            "action_dist_by_ctx": (
                {ctx: {a: dict(action_dist_by_ctx[ctx][a]) for a in action_dist_by_ctx[ctx]} for ctx in action_dist_by_ctx}
                if action_dist_by_ctx is not None else {}
            ),
            "realized_speaker_dist": {a: dict(realized_speaker_dist[a]) for a in agents} if realized_speaker_dist is not None else {},
            "final_answers": final_answers_map,
        }

    return result


# =========================
# シナリオ抽出
# =========================

def select_scenarios(
    problems: Dict[str, Dict[str, Any]],
    acc_by_pid: Dict[str, Dict[str, Any]],
    scenario: str,
) -> List[str]:
    """
    initial_answers と gold を見て scenario に合う pid を抽出する。
      - two_wrong_one_correct: 1 正解 + 2 誤答
      - two_correct_one_wrong: 2 正解 + 1 誤答
    """
    scenario = scenario.strip().lower()
    selected: List[str] = []

    for pid, pdata in problems.items():
        acc = acc_by_pid.get(pid)
        if not acc:
            continue
        gold = acc.get("gold")
        if not isinstance(gold, str):
            continue

        initial = pdata.get("initial_answers", {})
        if not isinstance(initial, dict) or not initial:
            continue

        if scenario == "all":
            selected.append(pid)
            continue

        correct_cnt = sum(1 for ans in initial.values() if isinstance(ans, str) and ans == gold)
        wrong_cnt = len(initial) - correct_cnt

        if scenario == "two_wrong_one_correct":
            if correct_cnt == 1 and wrong_cnt == 2:
                selected.append(pid)
        elif scenario == "two_correct_one_wrong":
            if correct_cnt == 2 and wrong_cnt == 1:
                selected.append(pid)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    return selected


# =========================
# メトリクス（token-wise）
# =========================

def compute_final_accuracy_for_set(
    pids: List[str],
    acc_by_pid: Dict[str, Dict[str, Any]],
) -> float:
    if not pids:
        return 0.0
    correct = 0
    total = 0
    for pid in pids:
        acc = acc_by_pid.get(pid)
        if not acc:
            continue
        is_correct = acc.get("is_correct")
        if is_correct is None:
            gold = acc.get("gold")
            fa = acc.get("final_answer")
            if isinstance(gold, str) and isinstance(fa, str):
                is_correct = (gold == fa)
            else:
                continue
        total += 1
        if is_correct:
            correct += 1
    return (correct / total) if total else 0.0


def _max_token_for_pids(pids: List[str], problems: Dict[str, Dict[str, Any]]) -> int:
    max_seen = 0
    for pid in pids:
        pdata = problems.get(pid)
        if not pdata:
            continue
        tb = pdata.get("token_budget")
        if isinstance(tb, int) and tb > max_seen:
            max_seen = tb
        snaps = pdata.get("snapshots", [])
        if isinstance(snaps, list) and snaps:
            t = snaps[-1].get("tokens")
            if isinstance(t, int) and t > max_seen:
                max_seen = t
    return max_seen


def compute_tokenwise_majority_accuracy(
    pids: List[str],
    problems: Dict[str, Dict[str, Any]],
    acc_by_pid: Dict[str, Dict[str, Any]],
    token_step: int,
) -> Tuple[List[int], List[float]]:
    """
    token=0..max_token の多数決正解率を list で返す。
    仕様:
      - forward-fill: 各スナップショットの回答を、その時点以降に適用。
      - 同票（3人全てバラバラ等）のときは不正解（None）扱い。
    """
    if not pids:
        return [], []

    max_token = _max_token_for_pids(pids, problems)
    tokens_out = list(range(0, max_token + 1, token_step))
    # 各問題ごとに token ごとの 0/1 を用意
    per_pid_acc: Dict[str, List[float]] = {}

    for pid in pids:
        pdata = problems.get(pid)
        if not pdata:
            continue
        snaps = pdata.get("snapshots", [])
        agents = pdata.get("agents", [])
        acc = acc_by_pid.get(pid)
        if not acc or not snaps:
            continue
        gold = acc.get("gold")
        if not isinstance(gold, str):
            continue

        # 長さ max_token+1 の配列
        arr = [0.0] * (max_token + 1)
        prev_tok = 0
        # 最初のスナップショットの精度を初期値にする
        first_snap = snaps[0]
        ans_map_first = first_snap.get("answers", {}) or {}
        first_maj = majority_vote([ans_map_first.get(a) for a in agents])
        prev_acc = 1.0 if first_maj == gold else 0.0

        for s in snaps:
            tok = s.get("tokens")
            if not isinstance(tok, int):
                continue
            # 区間 [prev_tok, tok) を prev_acc で塗る
            end = min(tok, max_token + 1)
            for t in range(prev_tok, end):
                arr[t] = prev_acc

            # 現在の snapshot で精度を計算
            ans_map = s.get("answers", {}) or {}
            maj = majority_vote([ans_map.get(a) for a in agents])
            curr_acc = 1.0 if maj == gold else 0.0

            prev_acc = curr_acc
            prev_tok = tok

        # 末尾を塗る
        for t in range(prev_tok, max_token + 1):
            arr[t] = prev_acc

        per_pid_acc[pid] = arr

    if not per_pid_acc:
        return [], []

    acc_series: List[float] = []
    for tok in tokens_out:
        vals = [arr[tok] for arr in per_pid_acc.values() if tok < len(arr)]
        acc_series.append(sum(vals) / len(vals) if vals else 0.0)

    return tokens_out, acc_series


def compute_tokenwise_per_agent_accuracy(
    pids: List[str],
    problems: Dict[str, Dict[str, Any]],
    acc_by_pid: Dict[str, Dict[str, Any]],
    token_step: int,
) -> Tuple[List[int], Dict[str, List[float]]]:
    """
    agent -> token=0..max_token の正解率（list）を返す。
    """
    if not pids:
        return [], {}

    max_token = _max_token_for_pids(pids, problems)

    agents_all: List[str] = []
    for pid in pids:
        pdata = problems.get(pid)
        if pdata and isinstance(pdata.get("agents"), list):
            agents_all = pdata["agents"]
            break
    if not agents_all:
        return [], {}

    pid_snaps: Dict[str, List[Dict[str, Any]]] = {}
    pid_ptr: Dict[str, int] = {}

    for pid in pids:
        pdata = problems.get(pid)
        if not pdata:
            continue
        snaps = pdata.get("snapshots", [])
        if not isinstance(snaps, list) or not snaps:
            continue
        pid_snaps[pid] = snaps
        pid_ptr[pid] = 0

    if not pid_snaps:
        return [], {}

    tokens: List[int] = []
    series: Dict[str, List[float]] = {a: [] for a in agents_all}

    for tok in range(0, max_token + 1, token_step):
        correct_cnt = {a: 0 for a in agents_all}
        total_cnt = {a: 0 for a in agents_all}

        for pid, snaps in pid_snaps.items():
            acc = acc_by_pid.get(pid)
            if not acc:
                continue
            gold = acc.get("gold")
            if not isinstance(gold, str):
                continue

            ptr = pid_ptr[pid]
            while ptr + 1 < len(snaps) and isinstance(snaps[ptr + 1].get("tokens"), int) and snaps[ptr + 1]["tokens"] <= tok:
                ptr += 1
            pid_ptr[pid] = ptr

            ans_map = snaps[ptr].get("answers", {})

            for a in agents_all:
                ans = ans_map.get(a)
                if not isinstance(ans, str):
                    continue
                total_cnt[a] += 1
                if ans == gold:
                    correct_cnt[a] += 1

        for a in agents_all:
            series[a].append((correct_cnt[a] / total_cnt[a]) if total_cnt[a] else 0.0)

        tokens.append(tok)

    return tokens, series


def analyze_interrupt_effects(
    pids: List[str],
    problems: Dict[str, Dict[str, Any]],
    acc_by_pid: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    interrupt が発生したターンで、割り込みを行ったエージェントの回答が
    直前からどう変化したかを評価する。
    戻り値:
      {
        "total_interrupts": int,
        "evaluated": int,              # gold と最終回答が両方あるもの
        "improved": int,              # 割り込み後〜最終で誰かが正答化
        "worsened": int,              # 割り込み後〜最終で誰かが誤答化
        "unchanged": int,
        "details": [...],              # 少数の例を保持（最初の50件）
      }
    """
    details: List[Dict[str, Any]] = []
    total = 0
    evaluated = improved = worsened = unchanged = 0
    unchanged_correct_to_correct = 0
    unchanged_wrong_to_wrong = 0

    for pid in pids:
        pdata = problems.get(pid)
        acc = acc_by_pid.get(pid)
        if not pdata or not acc:
            continue
        gold = acc.get("gold")
        if not isinstance(gold, str):
            continue
        snaps = pdata.get("snapshots", [])
        final_answers_map = pdata.get("final_answers", {}) or {}
        if not isinstance(final_answers_map, dict):
            final_answers_map = {}
        processed_agents = set()  # 同一問題・同一エージェントの割り込みは1回にまとめる
        for i, snap in enumerate(snaps):
            if snap.get("event_type_fixed") != "interrupt":
                continue
            total += 1
            speaker = snap.get("speaker")
            if not isinstance(speaker, str):
                continue
            if speaker in processed_agents:
                continue
            processed_agents.add(speaker)

            # 直前スナップショットの回答（全員）
            answers_before = {}
            if i > 0:
                answers_before = snaps[i - 1].get("answers", {}) or {}
            elif snaps:
                answers_before = snaps[0].get("answers", {}) or {}

            # 最終回答（全員）。final_answers_map が空なら最後の snapshot を代用。
            answers_final = final_answers_map if final_answers_map else (snaps[-1].get("answers", {}) if snaps else {})
            if not isinstance(answers_final, dict):
                answers_final = {}

            # 評価対象のエージェント集合：割り込み実施者＋その他
            agent_names = set(answers_final.keys()) | set(answers_before.keys())
            if not agent_names:
                continue

            def is_correct(ans: Any) -> Optional[bool]:
                return ans == gold if isinstance(ans, str) else None

            improved_flag = False
            worsened_flag = False
            for ag in agent_names:
                before = answers_before.get(ag)
                after = answers_final.get(ag)
                cb = is_correct(before)
                ca = is_correct(after)
                if cb is None or ca is None:
                    continue
                if (not cb) and ca:
                    improved_flag = True
                if cb and (not ca):
                    worsened_flag = True

            evaluated += 1
            if improved_flag:
                improved += 1
                outcome = "improved"
            elif worsened_flag:
                worsened += 1
                outcome = "worsened"
            else:
                # unchanged: 分類する
                unchanged += 1
                # unchanged の内訳判定（任意のエージェントで正解->正解 or 不正解->不正解が維持されたかを見る）
                # 判定は、全員の before/after を調べ、少なくとも1人が正解を維持していれば c2c、
                # 全員が不正解を維持なら w2w、それ以外は0
                c2c = False
                w2w = True  # 一人でも正解していれば False にする
                for ag in agent_names:
                    before = answers_before.get(ag)
                    after = answers_final.get(ag)
                    cb = is_correct(before)
                    ca = is_correct(after)
                    if cb is True and ca is True:
                        c2c = True
                    if ca is True:
                        w2w = False
                if c2c:
                    unchanged_correct_to_correct += 1
                elif w2w:
                    unchanged_wrong_to_wrong += 1
                outcome = "unchanged"

            if len(details) < 50:
                details.append({
                    "pid": pid,
                    "speaker": speaker,
                    "gold": gold,
                    "before": {ag: answers_before.get(ag) for ag in agent_names},
                    "final": {ag: answers_final.get(ag) for ag in agent_names},
                    "outcome": outcome,
                })

    return {
        "total_interrupts": total,
        "evaluated": evaluated,
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "unchanged_correct_to_correct": unchanged_correct_to_correct,
        "unchanged_wrong_to_wrong": unchanged_wrong_to_wrong,
        "details": details,
    }


def analyze_speak_effects(
    pids: List[str],
    problems: Dict[str, Dict[str, Any]],
    acc_by_pid: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    interrupt ではない utterance の効果を評価する。
    - event_type_fixed != interrupt かつ event_type == utterance を対象
    - 同一エージェントの連続 utterance は 1 回として扱う
    改善/悪化判定は interrupt_effects と同じロジック:
      speaker が最終正答で誰かが正答化 → improved
      speaker が最終誤答で誰かが誤答化 → worsened
    """
    details: List[Dict[str, Any]] = []
    total = evaluated = improved = worsened = unchanged = 0
    unchanged_correct_to_correct = 0
    unchanged_wrong_to_wrong = 0

    for pid in pids:
        pdata = problems.get(pid)
        acc = acc_by_pid.get(pid)
        if not pdata or not acc:
            continue
        gold = acc.get("gold")
        if not isinstance(gold, str):
            continue
        snaps = pdata.get("snapshots", [])
        final_answers_map = pdata.get("final_answers", {}) or {}
        if not isinstance(final_answers_map, dict):
            final_answers_map = {}

        prev_speaker_for_speak = None

        for i, snap in enumerate(snaps):
            if snap.get("event_type_fixed") == "interrupt":
                prev_speaker_for_speak = snap.get("speaker")
                continue
            if snap.get("event_type") != "utterance":
                prev_speaker_for_speak = snap.get("speaker")
                continue

            speaker = snap.get("speaker")
            if not isinstance(speaker, str):
                prev_speaker_for_speak = speaker
                continue
            # 同一エージェントの連続 utterance はスキップ
            if speaker == prev_speaker_for_speak:
                continue
            prev_speaker_for_speak = speaker

            total += 1

            answers_before = {}
            if i > 0:
                answers_before = snaps[i - 1].get("answers", {}) or {}
            elif snaps:
                answers_before = snaps[0].get("answers", {}) or {}

            answers_final = final_answers_map if final_answers_map else (snaps[-1].get("answers", {}) if snaps else {})
            if not isinstance(answers_final, dict):
                answers_final = {}

            agent_names = set(answers_final.keys()) | set(answers_before.keys())
            if not agent_names:
                continue

            def is_correct(ans: Any) -> Optional[bool]:
                return ans == gold if isinstance(ans, str) else None

            speaker_before_correct = is_correct(answers_before.get(speaker)) is True
            any_improved = any(
                (is_correct(answers_before.get(ag)) is False) and (is_correct(answers_final.get(ag)) is True)
                for ag in agent_names
            )
            improved_flag = speaker_before_correct and any_improved
            worsened_flag = (not speaker_before_correct) and any_improved

            evaluated += 1
            if improved_flag:
                improved += 1
                outcome = "improved"
            elif worsened_flag:
                worsened += 1
                outcome = "worsened"
            else:
                unchanged += 1
                c2c = False
                w2w = True
                for ag in agent_names:
                    cb = is_correct(answers_before.get(ag))
                    ca = is_correct(answers_final.get(ag))
                    if cb is True and ca is True:
                        c2c = True
                    if ca is True:
                        w2w = False
                if c2c:
                    unchanged_correct_to_correct += 1
                elif w2w:
                    unchanged_wrong_to_wrong += 1
                outcome = "unchanged"

            if len(details) < 50:
                details.append({
                    "pid": pid,
                    "speaker": speaker,
                    "gold": gold,
                    "before": {ag: answers_before.get(ag) for ag in agent_names},
                    "final": {ag: answers_final.get(ag) for ag in agent_names},
                    "outcome": outcome,
                })

    return {
        "total_speak": total,
        "evaluated": evaluated,
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "unchanged_correct_to_correct": unchanged_correct_to_correct,
        "unchanged_wrong_to_wrong": unchanged_wrong_to_wrong,
        "details": details,
    }


def aggregate_action_distribution(
    pids: List[str],
    problems: Dict[str, Dict[str, Any]],
) -> Dict[str, Counter]:
    """
    action_plan.action の分布を、対象 pid 全体で集計する。
    return: {agent: Counter(action -> count)}
    """
    dist: Dict[str, Counter] = defaultdict(Counter)
    for pid in pids:
        pdata = problems.get(pid)
        if not pdata:
            continue
        ad = pdata.get("action_dist", {})
        if isinstance(ad, dict):
            for agent, m in ad.items():
                if isinstance(m, dict):
                    dist[agent].update(m)
    return dist


def aggregate_action_distribution_by_context(
    pids: List[str],
    problems: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Counter]]:
    """
    action_plan.action の分布をイベント文脈ごとに集計する。
    return: {context: {agent: Counter(action)}}
    context は "talk" (utterance/interrupt) と "silence" を想定。
    """
    dist: Dict[str, Dict[str, Counter]] = {
        "talk": defaultdict(Counter),
        "silence": defaultdict(Counter),
    }
    for pid in pids:
        pdata = problems.get(pid)
        if not pdata:
            continue
        by_ctx = pdata.get("action_dist_by_ctx", {})
        if isinstance(by_ctx, dict):
            for ctx, agents_map in by_ctx.items():
                if ctx not in dist:
                    continue
                if isinstance(agents_map, dict):
                    for agent, m in agents_map.items():
                        if isinstance(m, dict):
                            dist[ctx][agent].update(m)
    return dist


def aggregate_realized_speaker_distribution(
    pids: List[str],
    problems: Dict[str, Dict[str, Any]],
) -> Dict[str, Counter]:
    """
    実現された speaker 行動（speak/interrupt）を集計する。
    interrupt 推定を反映した event_type_fixed に基づいてカウント済みのものを足し合わせる。
    """
    dist: Dict[str, Counter] = defaultdict(Counter)
    for pid in pids:
        pdata = problems.get(pid)
        if not pdata:
            continue
        rd = pdata.get("realized_speaker_dist", {})
        if isinstance(rd, dict):
            for agent, m in rd.items():
                if isinstance(m, dict):
                    dist[agent].update(m)
    return dist


# =========================
# プロット
# =========================


def plot_token_accuracy_multi(
    data_list: List[Dict[str, Any]],
    title: str,
    out_path: str,
) -> None:
    # グラフのサイズを適切に設定（論文等の標準的な比率）
    plt.figure(figsize=(8, 5))

    # markers = ["o", "s", "^", "D", "v", "x", "*"]
    # linestyles = ["-", "--", "-.", ":", "-", "--", "-."]

    has_plot = False
    for i, item in enumerate(data_list):
        label = item["label"]
        tokens = item["tokens"]
        series = item["series"]
        if not series or not tokens:
            continue

        xs = tokens
        ys = series
        
        plt.plot(
            xs, 
            ys, 
            label=label,
            marker="o",
        )
        has_plot = True

    plt.xlabel("Public tokens used")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0.0, 1.05)
    
    # グリッドも細く、控えめに設定
    plt.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.5)
    
    if has_plot:
        # 凡例のフォントサイズを微調整
        plt.legend(fontsize='small', frameon=True)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300) # 解像度を上げて保存
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def plot_final_accuracy_bar(
    bar_data: List[Dict[str, Any]],
    title: str,
    out_path: str,
) -> None:
    if not bar_data:
        print("[INFO] No data to plot for final accuracy bar.")
        return

    labels = [d["label"] for d in bar_data]
    acc_all = [d["acc_all"] for d in bar_data]
    acc_common = [d["acc_common"] for d in bar_data]

    x = list(range(len(labels)))
    width = 0.35

    plt.figure()
    plt.bar([xi - width / 2 for xi in x], acc_all, width=width, label="All target scenarios")
    plt.bar([xi + width / 2 for xi in x], acc_common, width=width, label="Common scenarios")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Final accuracy")
    plt.ylim(0.0, 1.05)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def plot_per_agent_token_accuracy(
    tokens: List[int],
    per_agent_series: Dict[str, List[float]],
    title: str,
    out_path: str,
) -> None:
    if not per_agent_series or not tokens:
        print(f"[INFO] No per-agent token accuracy to plot: {out_path}")
        return

    plt.figure()
    for agent in sorted(per_agent_series.keys()):
        ys = per_agent_series[agent]
        if not ys:
            continue
        plt.plot(tokens, ys, marker="o", label=agent, alpha=0.85)

    plt.xlabel("Public tokens used")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def plot_action_distribution_grouped(
    dist: Dict[str, Counter],
    title: str,
    out_path: str,
) -> None:
    """
    x: action 種類, 各 action に agent ごとの棒を並べる。
    """
    if not dist:
        print(f"[INFO] No action distribution to plot: {out_path}")
        return

    agents = sorted(dist.keys())
    actions = sorted({act for a in agents for act in dist[a].keys()})
    if not actions:
        print(f"[INFO] No actions found to plot: {out_path}")
        return

    x = list(range(len(actions)))
    n_agents = len(agents)
    width = 0.8 / max(1, n_agents)

    plt.figure()
    for i, agent in enumerate(agents):
        ys = [dist[agent].get(act, 0) for act in actions]
        offsets = [xi - 0.4 + (i + 0.5) * width for xi in x]
        plt.bar(offsets, ys, width=width, label=agent)

    plt.xticks(x, actions, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def make_comparison_out_dir(base_out_dir: str, run_dirs: List[str], max_idx: Optional[int], scenario: str) -> str:
    tags = [os.path.basename(r.rstrip(os.sep)) for r in run_dirs]
    dir_name = "__vs__".join(tags)
    if len(dir_name) > 150:
        dir_name = dir_name[:140] + "_etc"
    if max_idx is not None:
        dir_name += f"_max{max_idx:03d}"
    dir_name += f"_{scenario}"
    out_dir = os.path.join(base_out_dir, dir_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# =========================
# メイン
# =========================

def main():
    parser = argparse.ArgumentParser(description="Compare multiple runs with exec logs (token-wise analysis).")
    parser.add_argument("runs", nargs="+", help="Run directories or IDs (e.g. run_2025... run_2025...)")
    parser.add_argument("--out-dir", default="analysis_outputs", help="出力ディレクトリ (default: analysis_outputs)")
    parser.add_argument("--max-problem-index", type=int, default=None, help="problem_001〜problem_N の N")
    parser.add_argument("--token-step", type=int, default=25, help="token を何刻みでプロットするか (default: 25)")
    parser.add_argument(
        "--scenario",
        default="two_wrong_one_correct",
        choices=["two_wrong_one_correct", "two_correct_one_wrong", "all"],
        help="対象シナリオ (default: two_wrong_one_correct). 'all' は初期回答の正誤構成で絞り込まない。",
    )
    args = parser.parse_args()
    if args.token_step <= 0:
        raise ValueError("--token-step must be a positive integer")

    run_dirs = [resolve_run_dir(r) for r in args.runs]
    if len(run_dirs) < 2:
        print("[WARN] It is recommended to provide at least 2 runs for comparison.")

    os.makedirs(args.out_dir, exist_ok=True)
    pair_out_dir = make_comparison_out_dir(args.out_dir, run_dirs, args.max_problem_index, args.scenario)

    print(f"[INFO] Comparing {len(run_dirs)} runs.")
    print(f"[INFO] Output dir: {pair_out_dir}")
    if args.max_problem_index is not None:
        print(f"[INFO] Using problems up to problem_{args.max_problem_index:03d}")
    print(f"[INFO] Scenario: {args.scenario}")

    all_runs_data: List[Dict[str, Any]] = []

    # ---- run ごとの読み込み ----
    for r_dir in run_dirs:
        tag = os.path.basename(r_dir.rstrip(os.sep))
        run_is_roundrobin = tag.endswith("_roundrobin")
        print(f"[INFO] Loading: {tag}")

        acc_full = load_accuracy(r_dir)
        probs_full = load_exec_logs(r_dir)

        print(f"[DEBUG] {tag}: #accuracy entries = {len(acc_full)}, #exec problems = {len(probs_full)}")

        probs, acc = filter_problems_by_max_index(probs_full, acc_full, args.max_problem_index)

        scenarios = select_scenarios(probs, acc, args.scenario)
        # 3人バラバラ（初回回答が全員異なる）は除外
        filtered = []
        for pid in scenarios:
            ia = probs.get(pid, {}).get("initial_answers", {})
            if not isinstance(ia, dict):
                continue
            uniq = set(ia.values())
            if len(uniq) == 3:
                continue
            filtered.append(pid)
        # 件数上限（max-problem-index）を「何件まで」に読み替え、先頭から切り詰め
        if args.max_problem_index is not None:
            filtered = filtered[: args.max_problem_index]
        scenarios = filtered
        print(f"  -> Found {len(scenarios)} scenarios after filtering (exclude 3-way tie, cap by max-problem-index).")

        final_acc_all = compute_final_accuracy_for_set(scenarios, acc)
        tokens_all, token_majority_all = compute_tokenwise_majority_accuracy(scenarios, probs, acc, args.token_step)
        pa_tokens_all, per_agent_token_all = compute_tokenwise_per_agent_accuracy(scenarios, probs, acc, args.token_step)
        action_dist_all = aggregate_action_distribution(scenarios, probs) if not run_is_roundrobin else {}
        action_dist_ctx_all = aggregate_action_distribution_by_context(scenarios, probs) if not run_is_roundrobin else {}
        realized_speaker_dist_all = aggregate_realized_speaker_distribution(scenarios, probs) if not run_is_roundrobin else {}
        interrupt_effects_all = analyze_interrupt_effects(scenarios, probs, acc) if not run_is_roundrobin else {}
        speak_effects_all = analyze_speak_effects(scenarios, probs, acc)

        all_runs_data.append({
            "dir": r_dir,
            "tag": tag,
            "acc": acc,
            "probs": probs,
            "scenarios": scenarios,
            "metrics": {
                "final_accuracy_all_target": final_acc_all,
                "token_majority_accuracy_all": token_majority_all,
                "token_majority_tokens_all": tokens_all,
                "per_agent_token_accuracy_all": per_agent_token_all,
                "per_agent_tokens_all": pa_tokens_all,
                "action_distribution_all": {a: dict(action_dist_all[a]) for a in action_dist_all},
                "action_distribution_by_ctx_all": {
                    ctx: {a: dict(action_dist_ctx_all.get(ctx, {}).get(a, {})) for a in action_dist_ctx_all.get(ctx, {})}
                    for ctx in action_dist_ctx_all
                },
                "realized_speaker_distribution_all": {a: dict(realized_speaker_dist_all[a]) for a in realized_speaker_dist_all},
                "interrupt_effects_all": interrupt_effects_all,
                "speak_effects_all": speak_effects_all,
                "is_roundrobin": run_is_roundrobin,
            },
        })

    if not all_runs_data:
        print("[WARN] No runs loaded.")
        return

    # ---- 共通シナリオ ----
    common_pids_set = set(all_runs_data[0]["scenarios"])
    for i in range(1, len(all_runs_data)):
        common_pids_set &= set(all_runs_data[i]["scenarios"])
    common_pids = sorted(list(common_pids_set))
    # max-problem-index を「件数上限」として共通問題にも適用
    if args.max_problem_index is not None:
        common_pids = common_pids[: args.max_problem_index]

    print(f"[INFO] Common scenarios (in ALL runs) = {len(common_pids)}")
    if common_pids:
        print("[INFO] Common scenarios (problem index):")
        for item in scenario_list_with_index(common_pids):
            print(f"  - {item['pid']} (index={item['index']})")

    # ---- 共通シナリオのメトリクス & 出力準備 ----
    plot_data_all = []
    plot_data_common = []
    bar_data = []

    json_output: Dict[str, Any] = {
        "scenario": args.scenario,
        "runs": {},
        "common": {
            "scenarios": scenario_list_with_index(common_pids),
        }
    }
    tags = ["proposed framework", "dynamic order without interruption", "fixed order"]
    i = 0

    for r_data in all_runs_data:
        tag = tags[i]
        i += 1
        run_is_roundrobin = r_data["metrics"].get("is_roundrobin", False)

        acc_final_common = compute_final_accuracy_for_set(common_pids, r_data["acc"])
        tokens_common, token_majority_common = compute_tokenwise_majority_accuracy(common_pids, r_data["probs"], r_data["acc"], args.token_step)
        pa_tokens_common, per_agent_token_common = compute_tokenwise_per_agent_accuracy(common_pids, r_data["probs"], r_data["acc"], args.token_step)
        action_dist_common = aggregate_action_distribution(common_pids, r_data["probs"]) if not run_is_roundrobin else {}
        action_dist_ctx_common = aggregate_action_distribution_by_context(common_pids, r_data["probs"]) if not run_is_roundrobin else {}
        realized_speaker_dist_common = aggregate_realized_speaker_distribution(common_pids, r_data["probs"]) if not run_is_roundrobin else {}
        interrupt_effects_common = analyze_interrupt_effects(common_pids, r_data["probs"], r_data["acc"]) if not run_is_roundrobin else {}
        speak_effects_common = analyze_speak_effects(common_pids, r_data["probs"], r_data["acc"])

        r_data["metrics"]["final_accuracy_common"] = acc_final_common
        r_data["metrics"]["token_majority_accuracy_common"] = token_majority_common
        r_data["metrics"]["token_majority_tokens_common"] = tokens_common
        r_data["metrics"]["per_agent_token_accuracy_common"] = per_agent_token_common
        r_data["metrics"]["per_agent_tokens_common"] = pa_tokens_common
        r_data["metrics"]["action_distribution_common"] = {a: dict(action_dist_common[a]) for a in action_dist_common}
        r_data["metrics"]["action_distribution_by_ctx_common"] = {
            ctx: {a: dict(action_dist_ctx_common.get(ctx, {}).get(a, {})) for a in action_dist_ctx_common.get(ctx, {})}
            for ctx in action_dist_ctx_common
        }
        r_data["metrics"]["realized_speaker_distribution_common"] = {a: dict(realized_speaker_dist_common[a]) for a in realized_speaker_dist_common}
        r_data["metrics"]["interrupt_effects_common"] = interrupt_effects_common
        r_data["metrics"]["speak_effects_common"] = speak_effects_common

        print(f"[RESULT] {tag} | Final Acc (All Target): {r_data['metrics']['final_accuracy_all_target']:.3f}")
        print(f"[RESULT] {tag} | Final Acc (Common Only): {acc_final_common:.3f}")

        plot_data_all.append({
            "label": tag,
            "tokens": r_data["metrics"]["token_majority_tokens_all"],
            "series": r_data["metrics"]["token_majority_accuracy_all"],
        })
        plot_data_common.append({
            "label": f"{tag} (common)",
            "tokens": tokens_common,
            "series": token_majority_common,
        })
        bar_data.append({
            "label": tag,
            "acc_all": r_data["metrics"]["final_accuracy_all_target"],
            "acc_common": acc_final_common,
        })

        # run ごとの per-agent token accuracy（全対象）
        out_pa = os.path.join(pair_out_dir, f"per_agent_token_accuracy_{tag}.png")
        plot_per_agent_token_accuracy(
            r_data["metrics"]["per_agent_tokens_all"],
            r_data["metrics"]["per_agent_token_accuracy_all"],
            title=f"{tag} - Per-Agent Token-wise Accuracy ({args.scenario}, all target)",
            out_path=out_pa,
        )

        if not r_data["metrics"]["is_roundrobin"]:
            # run ごとの行動分布（action_plan.action）
            out_ad = os.path.join(pair_out_dir, f"action_distribution_{tag}.png")
            plot_action_distribution_grouped(
                aggregate_action_distribution(r_data["scenarios"], r_data["probs"]),
                title=f"{tag} - Action Selection Distribution (action_plan.action, all target)",
                out_path=out_ad,
            )
            # イベント別の行動分布（talk / silence）
            for ctx_label, ctx_key in [("talk", "talk"), ("silence", "silence")]:
                out_ctx = os.path.join(pair_out_dir, f"action_distribution_{ctx_label}_{tag}.png")
                plot_action_distribution_grouped(
                    aggregate_action_distribution_by_context(r_data["scenarios"], r_data["probs"]).get(ctx_key, {}),
                    title=f"{tag} - Action Selection Distribution ({ctx_label}, all target)",
                    out_path=out_ctx,
                )

            # run ごとの実現発話分布（speak/interrupt）
            out_rs = os.path.join(pair_out_dir, f"realized_speaker_distribution_{tag}.png")
            plot_action_distribution_grouped(
                aggregate_realized_speaker_distribution(r_data["scenarios"], r_data["probs"]),
                title=f"{tag} - Realized Speaker Distribution (speak/interrupt, inferred interrupt, all target)",
                out_path=out_rs,
            )

        json_output["runs"][tag] = {
            "dir": r_data["dir"],
            "target_scenarios": scenario_list_with_index(r_data["scenarios"]),
            "metrics": {
                "final_accuracy_all_target": r_data["metrics"]["final_accuracy_all_target"],
                "final_accuracy_common": acc_final_common,
                "token_majority_accuracy_all": r_data["metrics"]["token_majority_accuracy_all"],
                "token_majority_accuracy_common": token_majority_common,
                "token_majority_tokens_all": r_data["metrics"]["token_majority_tokens_all"],
                "token_majority_tokens_common": tokens_common,
                "per_agent_token_accuracy_all": r_data["metrics"]["per_agent_token_accuracy_all"],
                "per_agent_token_accuracy_common": per_agent_token_common,
                "per_agent_tokens_all": r_data["metrics"]["per_agent_tokens_all"],
                "per_agent_tokens_common": pa_tokens_common,
                "action_distribution_all": r_data["metrics"]["action_distribution_all"],
                "action_distribution_common": r_data["metrics"]["action_distribution_common"],
                "action_distribution_by_ctx_all": r_data["metrics"]["action_distribution_by_ctx_all"],
                "action_distribution_by_ctx_common": r_data["metrics"]["action_distribution_by_ctx_common"],
                "realized_speaker_distribution_all": r_data["metrics"]["realized_speaker_distribution_all"],
                "realized_speaker_distribution_common": r_data["metrics"]["realized_speaker_distribution_common"],
                "interrupt_effects_all": r_data["metrics"]["interrupt_effects_all"],
                "interrupt_effects_common": r_data["metrics"]["interrupt_effects_common"],
                "speak_effects_all": r_data["metrics"]["speak_effects_all"],
                "speak_effects_common": r_data["metrics"]["speak_effects_common"],
            }
        }

    # ---- グラフ: token-wise 多数決（各自シナリオ） ----
    out_path_all = os.path.join(pair_out_dir, "token_accuracy_runwise.png")
    plot_token_accuracy_multi(
        plot_data_all,
        title=f"Token-wise Majority Accuracy ({args.scenario}, each run)",
        out_path=out_path_all,
    )

    # ---- グラフ: token-wise 多数決（共通シナリオ） ----
    out_path_common = os.path.join(pair_out_dir, "token_accuracy_common.png")
    plot_token_accuracy_multi(
        plot_data_common,
        title=f"Token-wise Majority Accuracy (common problems only, {args.scenario})",
        out_path=out_path_common,
    )

    # ---- 最終正解率バー ----
    out_path_bar = os.path.join(pair_out_dir, "final_accuracy_bar.png")
    plot_final_accuracy_bar(
        bar_data,
        title=f"Final Accuracy ({args.scenario})",
        out_path=out_path_bar,
    )

    # ---- JSON 保存 ----
    out_json = os.path.join(pair_out_dir, "analysis_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved analysis results to {out_json}")


if __name__ == "__main__":
    main()
