#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
複数の run ログ（2つ以上）を比較し、以下を計算・可視化するスクリプト。

対象シナリオ:
  - initial_answer で「二人が正答・一人が誤答」を選んでいる問題のみ。

1. 各 run ごとに：
   - 上記シナリオのみを抽出
   - そのシナリオ集合における最終解答の正解率を算出
   - 上記シナリオ集合における各ターンの多数決正解率を算出
   - 各エージェントのターン別正解率と「前ターンから解答を変えた割合」を算出し、プロット

2. 指定された「すべての run」で共通して
   「二人正解・一人誤答」となっている同一問題のみを対象に、
   - 各 run における最終正解率を算出
   - 各 run におけるターンごとの多数決正解率を算出

3. グラフ出力:
   - 全 run について、各自のシナリオ集合でのターン別多数決正解率
     => turn_accuracy_runwise.png
   - 全 run について、共通シナリオ集合でのターン別多数決正解率
     => turn_accuracy_common.png
   - 全 run について、最終正解率（全シナリオ / 共通シナリオ）のバーグラフ
     => final_accuracy_bar.png

4. 追加:
   - 各 run について、エージェントごとのターン別正解率 / 解答変更率のグラフ
     => per_agent_turn_accuracy_<run>.png
        per_agent_turn_change_rate_<run>.png

使い方:
    python analyze_runs.py RUN1 RUN2 [RUN3 ...]
    python analyze_runs.py RUN1 RUN2 --max-problem-index 200
"""

import argparse
import json
import os
import re
from glob import glob
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Optional

import matplotlib.pyplot as plt


# ------------- ユーティリティ ------------- #

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

        # --- ID の正規化 ---
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


def load_discussion(run_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    run_dir 内の problem_* ディレクトリから discussion_log.json を読み込み、
    problem_id -> {
        "initial_answers": {agent: "A"/"B"/...},
        "turn_answers": {turn(int): {agent: "A"/...}}
    }
    を返す。

    turn は文字列・整数混在の可能性があるため、必ず int に正規化する。
    """
    problem_dirs = sorted(glob(os.path.join(run_dir, "problem_*")))
    result: Dict[str, Dict[str, Any]] = {}
    for pdir in problem_dirs:
        pid = os.path.basename(pdir)  # problem_001 など
        dlog_path = os.path.join(pdir, "discussion_log.json")
        if not os.path.isfile(dlog_path):
            continue
        with open(dlog_path, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load {dlog_path}: {e}")
                continue

        if not isinstance(records, list) or len(records) == 0:
            continue

        initial_answers: Dict[str, str] = {}
        turn_answers: Dict[int, Dict[str, str]] = {}

        # ---------------- 初回回答 (turn == 0) ----------------
        for rec in records:
            turn_raw = rec.get("turn")
            if isinstance(turn_raw, str):
                try:
                    turn0 = int(turn_raw)
                except ValueError:
                    continue
            else:
                turn0 = turn_raw

            if turn0 == 0 and rec.get("initial_answers"):
                ia = rec["initial_answers"]
                for agent, info in ia.items():
                    ans = info.get("answer")
                    if isinstance(ans, str):
                        initial_answers[agent] = ans.strip()
                turn_answers[0] = dict(initial_answers)
                break

        if not initial_answers:
            continue

        # ---------------- 各ターンの answer ----------------
        for rec in records:
            turn = rec.get("turn")
            if isinstance(turn, str):
                try:
                    turn = int(turn)
                except ValueError:
                    continue

            if turn is None or turn == 0:
                continue

            answers_by_agent: Dict[str, str] = {}

            # 旧バージョン: consensus_state
            if "consensus_state" in rec and isinstance(rec["consensus_state"], dict):
                for agent, info in rec["consensus_state"].items():
                    ans = info.get("answer")
                    if isinstance(ans, str):
                        answers_by_agent[agent] = ans.strip()

            # 新バージョン: agent_states
            elif "agent_states" in rec and isinstance(rec["agent_states"], list):
                for st in rec["agent_states"]:
                    agent = st.get("agent_name")
                    ans = st.get("current_answer")
                    if agent and isinstance(ans, str):
                        answers_by_agent[agent] = ans.strip()

            if answers_by_agent:
                turn_answers[turn] = answers_by_agent

        result[pid] = {
            "initial_answers": initial_answers,
            "turn_answers": turn_answers,
        }

    return result


def majority_vote(answers: List[str]) -> Optional[str]:
    """単純多数決（票数最大のもの）。同票のときは None を返す。"""
    filtered = [a for a in answers if a]
    if not filtered:
        return None
    counts = Counter(filtered)
    most_common = counts.most_common()
    if len(most_common) == 1:
        return most_common[0][0]
    if most_common[0][1] == most_common[1][1]:
        return None
    return most_common[0][0]


def pid_to_index(pid: str) -> Optional[int]:
    """
    "problem_001" -> 1 のように数値部分を返す。
    期待形式と違う場合は None。
    """
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
    """
    max_index が指定されていれば、problem_001〜problem_max_index だけを残す。
    problems, acc_by_pid の両方をフィルタする。
    """
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


# ------------- 集計ロジック ------------- #

def select_two_correct_one_wrong(
    problems: Dict[str, Dict[str, Any]],
    acc_by_pid: Dict[str, Dict[str, Any]],
) -> List[str]:
    """
    initial_answers と gold を見て、
    「3人中ちょうど2人が正解・1人が不正解」の problem_id を返す。
    """
    selected: List[str] = []
    for pid, pdata in problems.items():
        acc = acc_by_pid.get(pid)
        if not acc:
            continue
        gold = acc.get("gold")
        if not isinstance(gold, str):
            continue
        initial = pdata.get("initial_answers", {})
        if not initial:
            continue

        correct_cnt = sum(1 for ans in initial.values() if ans == gold)
        if correct_cnt == 2 and len(initial) - correct_cnt == 1:
            selected.append(pid)
    return selected


def compute_final_accuracy_for_set(
    pids: List[str],
    acc_by_pid: Dict[str, Dict[str, Any]],
) -> float:
    """指定された problem_id 集合に対し、最終解答の正解率を計算。"""
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
    if total == 0:
        return 0.0
    return correct / total


def compute_turnwise_majority_accuracy(
    scenario_pids: List[str],
    problems: Dict[str, Dict[str, Any]],
    acc_by_pid: Dict[str, Dict[str, Any]],
) -> Dict[int, float]:
    """
    指定したシナリオ集合に対し、ターンごとの多数決正解率を計算。
    戻り値: {turn: accuracy(float)}  （turn 0 は initial_answers 多数決）
    """
    acc_by_turn: Dict[int, List[int]] = defaultdict(list)

    for pid in scenario_pids:
        pdata = problems.get(pid)
        acc = acc_by_pid.get(pid)
        if not pdata or not acc:
            continue
        gold = acc.get("gold")
        if not isinstance(gold, str):
            continue

        turn_answers: Dict[int, Dict[str, str]] = pdata.get("turn_answers", {})
        if not turn_answers:
            continue

        for turn, ans_map in turn_answers.items():
            maj = majority_vote(list(ans_map.values()))
            if maj is None:
                acc_by_turn[turn].append(0)
            else:
                acc_by_turn[turn].append(1 if maj == gold else 0)

    turn_accuracy: Dict[int, float] = {}
    for t, vals in acc_by_turn.items():
        if not vals:
            continue
        turn_accuracy[t] = sum(vals) / len(vals)
    return turn_accuracy


# ------------- エージェントごとのターン別統計 ------------- #

def compute_per_agent_turn_stats(
    scenario_pids: List[str],
    problems: Dict[str, Dict[str, Any]],
    acc_by_pid: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
    """
    指定シナリオ集合について、
      - 各エージェントのターン別正解率
      - 各エージェントのターン別「前ターンから回答を変えた割合」
    を計算して返す。

    戻り値:
      (per_agent_turn_accuracy, per_agent_turn_change_rate)

      per_agent_turn_accuracy: {agent: {turn: accuracy}}
      per_agent_turn_change_rate: {agent: {turn: change_rate}}
    """
    correct_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    total_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    change_num: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    change_den: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for pid in scenario_pids:
        pdata = problems.get(pid)
        acc = acc_by_pid.get(pid)
        if not pdata or not acc:
            continue
        gold = acc.get("gold")
        if not isinstance(gold, str):
            continue

        turn_answers: Dict[int, Dict[str, str]] = pdata.get("turn_answers", {})
        if not turn_answers:
            continue

        prev_answer: Dict[str, str] = {}
        turns_sorted = sorted(turn_answers.keys())

        for t in turns_sorted:
            ans_map = turn_answers[t]
            for agent, ans in ans_map.items():
                total_counts[agent][t] += 1
                if ans == gold:
                    correct_counts[agent][t] += 1

                if agent in prev_answer:
                    change_den[agent][t] += 1
                    if ans != prev_answer[agent]:
                        change_num[agent][t] += 1

                prev_answer[agent] = ans

    per_agent_acc: Dict[str, Dict[int, float]] = {}
    per_agent_change: Dict[str, Dict[int, float]] = {}

    for agent, t_dict in total_counts.items():
        per_agent_acc[agent] = {}
        for t, tot in t_dict.items():
            if tot == 0:
                continue
            c = correct_counts[agent][t]
            per_agent_acc[agent][t] = c / tot

    for agent, t_dict in change_den.items():
        per_agent_change[agent] = {}
        for t, den in t_dict.items():
            if den == 0:
                continue
            num = change_num[agent][t]
            per_agent_change[agent][t] = num / den

    return per_agent_acc, per_agent_change


def plot_per_agent_turn_stats_single(
    per_agent_turn_acc: Dict[str, Dict[int, float]],
    per_agent_change_rate: Dict[str, Dict[int, float]],
    title_prefix: str,
    out_path_acc: str,
    out_path_change: str,
) -> None:
    """1つの run について、エージェントごとのターン別正解率 / 変化率をプロット。"""
    if not per_agent_turn_acc and not per_agent_change_rate:
        print(f"[INFO] No per-agent stats to plot for {title_prefix}")
        return

    # --- 正解率 ---
    if per_agent_turn_acc:
        plt.figure()
        for agent, t_dict in sorted(per_agent_turn_acc.items(), key=lambda x: x[0]):
            xs = sorted(t_dict.keys())
            ys = [t_dict[t] for t in xs]
            plt.plot(xs, ys, marker="o", label=agent)
        plt.xlabel("Turn")
        plt.ylabel("Accuracy")
        plt.title(f"{title_prefix} - Per-Agent Turn Accuracy")
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path_acc)
        plt.close()
        print(f"[INFO] Saved plot: {out_path_acc}")
    else:
        print(f"[INFO] No per-agent accuracy stats for {title_prefix}")

    # --- 変化率 ---
    if per_agent_change_rate:
        plt.figure()
        for agent, t_dict in sorted(per_agent_change_rate.items(), key=lambda x: x[0]):
            xs = sorted(t_dict.keys())
            ys = [t_dict[t] for t in xs]
            plt.plot(xs, ys, marker="o", label=agent)
        plt.xlabel("Turn")
        plt.ylabel("Change rate (from previous answer)")
        plt.title(f"{title_prefix} - Per-Agent Turn Change Rate")
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path_change)
        plt.close()
        print(f"[INFO] Saved plot: {out_path_change}")
    else:
        print(f"[INFO] No per-agent change-rate stats for {title_prefix}")


# ------------- 複数 run 用プロット ------------- #

def plot_turn_accuracy_multi(
    data_list: List[Dict[str, Any]],
    title: str,
    out_path: str,
) -> None:
    """
    任意の数の Run のターン別多数決正解率をプロットする。
    data_list: [{"label": str, "turn_acc": dict}, ...]
    """
    plt.figure()

    markers = ["o", "s", "^", "D", "v", "x", "*"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-."]

    has_plot = False

    for i, item in enumerate(data_list):
        turn_acc = item["turn_acc"]
        label = item["label"]
        if not turn_acc:
            continue

        xs = sorted(turn_acc.keys())
        ys = [turn_acc[x] for x in xs]

        m = markers[i % len(markers)]
        ls = linestyles[i % len(linestyles)]

        plt.plot(xs, ys, marker=m, linestyle=ls, label=label, alpha=0.8)
        has_plot = True

    plt.xlabel("Turn")
    plt.ylabel("Accuracy (majority vote)")
    plt.title(title)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    if has_plot:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def plot_final_accuracy_bar(
    data_list: List[Dict[str, Any]],
    title: str,
    out_path: str,
) -> None:
    """
    各 run の最終正解率（全対象シナリオ / 共通シナリオ）をバーグラフで描画。
    data_list: [{"label": str, "acc_all": float, "acc_common": float}, ...]
    """
    if not data_list:
        print(f"[INFO] No data to plot for final accuracy bar.")
        return

    labels = [d["label"] for d in data_list]
    acc_all = [d["acc_all"] for d in data_list]
    acc_common = [d["acc_common"] for d in data_list]

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


def make_comparison_out_dir(base_out_dir: str, run_dirs: List[str], max_idx: Optional[int]) -> str:
    """analysis_outputs/run1__vs__run2__vs__run3... を作成"""
    tags = [os.path.basename(r.rstrip(os.sep)) for r in run_dirs]
    dir_name = "__vs__".join(tags)

    # パス長対策（簡易）
    if len(dir_name) > 150:
        dir_name = dir_name[:140] + "_etc"

    if max_idx is not None:
        dir_name += f"_max{max_idx:03d}"

    out_dir = os.path.join(base_out_dir, dir_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def scenario_list_with_index(pids: List[str]) -> List[Dict[str, Any]]:
    """['problem_001', ...] -> [{pid, index}, ...]"""
    items = []
    for pid in pids:
        idx = pid_to_index(pid)
        items.append({"pid": pid, "index": idx})
    items.sort(key=lambda x: (x["index"] is None, x["index"] if x["index"] is not None else 0))
    return items


# ------------- メイン ------------- #

def main():
    parser = argparse.ArgumentParser(description="Compare multiple runs (two-correct-one-wrong scenarios).")
    parser.add_argument("runs", nargs="+", help="Run directories or IDs (e.g. run_2025... run_2025...)")
    parser.add_argument(
        "--out-dir",
        default="analysis_outputs",
        help="グラフなどを出力するディレクトリ (default: analysis_outputs)",
    )
    parser.add_argument(
        "--max-problem-index",
        type=int,
        default=None,
        help="problem_001〜problem_N までのみを対象にする場合の N (例: 200)",
    )
    args = parser.parse_args()

    run_dirs = [resolve_run_dir(r) for r in args.runs]
    if len(run_dirs) < 2:
        print("[WARN] It is recommended to provide at least 2 runs for comparison.")

    base_out_dir = args.out_dir
    os.makedirs(base_out_dir, exist_ok=True)
    pair_out_dir = make_comparison_out_dir(base_out_dir, run_dirs, args.max_problem_index)

    print(f"[INFO] Comparing {len(run_dirs)} runs.")
    print(f"[INFO] Output dir: {pair_out_dir}")
    if args.max_problem_index is not None:
        print(f"[INFO] Using problems up to problem_{args.max_problem_index:03d}")

    all_runs_data: List[Dict[str, Any]] = []

    # ---- 各 run ごとの読み込み・計算 ---- #
    for r_dir in run_dirs:
        tag = os.path.basename(r_dir.rstrip(os.sep))
        print(f"[INFO] Loading data for: {tag}")

        acc_full = load_accuracy(r_dir)
        probs_full = load_discussion(r_dir)

        print(f"[DEBUG] {tag}: #accuracy entries = {len(acc_full)}, #discussion problems = {len(probs_full)}")

        probs, acc = filter_problems_by_max_index(probs_full, acc_full, args.max_problem_index)

        # シナリオ抽出（二人正解・一人誤答）
        scenarios = select_two_correct_one_wrong(probs, acc)
        print(f"  -> Found {len(scenarios)} two-correct-one-wrong scenarios.")

        # 最終正解率（この run 単体での対象シナリオ）
        final_acc_all = compute_final_accuracy_for_set(scenarios, acc)

        # ターンごとの多数決正解率（この run 単体）
        turn_acc_all = compute_turnwise_majority_accuracy(scenarios, probs, acc)

        # エージェントごとのターン別統計（この run 単体）
        per_agent_acc_all, per_agent_change_all = compute_per_agent_turn_stats(scenarios, probs, acc)

        all_runs_data.append({
            "dir": r_dir,
            "tag": tag,
            "acc": acc,
            "probs": probs,
            "scenarios": scenarios,
            "metrics": {
                "final_accuracy_all_target": final_acc_all,
                "turn_accuracy_all": turn_acc_all,
                "per_agent_turn_accuracy_all": per_agent_acc_all,
                "per_agent_turn_change_rate_all": per_agent_change_all,
            },
        })

    if not all_runs_data:
        print("[WARN] No runs loaded.")
        return

    # ---- 共通シナリオの抽出 ---- #
    common_pids_set = set(all_runs_data[0]["scenarios"])
    for i in range(1, len(all_runs_data)):
        common_pids_set &= set(all_runs_data[i]["scenarios"])

    common_pids = sorted(list(common_pids_set))
    print(f"[INFO] Common scenarios (in ALL runs) = {len(common_pids)}")
    if common_pids:
        print("[INFO] Common scenarios (problem index):")
        for item in scenario_list_with_index(common_pids):
            print(f"  - {item['pid']} (index={item['index']})")

    # ---- 共通シナリオに対するメトリクス ---- #
    plot_data_all = []
    plot_data_common = []
    bar_data = []
    json_output = {
        "runs": {},
        "common": {
            "scenarios": scenario_list_with_index(common_pids),
            "metrics": {}
        }
    }

    for r_data in all_runs_data:
        tag = r_data["tag"]

        # 共通シナリオでの最終正解率 & ターン別多数決正解率 & エージェント別統計
        acc_final_common = compute_final_accuracy_for_set(common_pids, r_data["acc"])
        turn_acc_common = compute_turnwise_majority_accuracy(common_pids, r_data["probs"], r_data["acc"])
        per_agent_acc_common, per_agent_change_common = compute_per_agent_turn_stats(common_pids, r_data["probs"], r_data["acc"])

        r_data["metrics"]["final_accuracy_common"] = acc_final_common
        r_data["metrics"]["turn_accuracy_common"] = turn_acc_common
        r_data["metrics"]["per_agent_turn_accuracy_common"] = per_agent_acc_common
        r_data["metrics"]["per_agent_turn_change_rate_common"] = per_agent_change_common

        json_output["runs"][tag] = {
            "dir": r_data["dir"],
            "target_scenarios": scenario_list_with_index(r_data["scenarios"]),
            "metrics": r_data["metrics"],
        }

        print(f"[RESULT] {tag} | Final Acc (All Target): {r_data['metrics']['final_accuracy_all_target']:.3f}")
        print(f"[RESULT] {tag} | Final Acc (Common Only): {acc_final_common:.3f}")

        plot_data_all.append({
            "label": tag,
            "turn_acc": r_data["metrics"]["turn_accuracy_all"],
        })
        plot_data_common.append({
            "label": f"{tag} (common)",
            "turn_acc": turn_acc_common,
        })
        bar_data.append({
            "label": tag,
            "acc_all": r_data["metrics"]["final_accuracy_all_target"],
            "acc_common": acc_final_common,
        })

    # ---- グラフ 1: 各 run のターン別多数決正解率（全シナリオ） ---- #
    out_path_all = os.path.join(pair_out_dir, "turn_accuracy_runwise.png")
    plot_turn_accuracy_multi(
        plot_data_all,
        title="Turn-wise Majority Accuracy (two-correct-one-wrong, each run)",
        out_path=out_path_all,
    )

    # ---- グラフ 2: 共通シナリオでのターン別多数決正解率 ---- #
    out_path_common = os.path.join(pair_out_dir, "turn_accuracy_common.png")
    plot_turn_accuracy_multi(
        plot_data_common,
        title="Turn-wise Majority Accuracy (common problems only, two-correct-one-wrong)",
        out_path=out_path_common,
    )

    # ---- グラフ 3: 最終正解率バーグラフ ---- #
    out_path_bar = os.path.join(pair_out_dir, "final_accuracy_bar.png")
    plot_final_accuracy_bar(
        bar_data,
        title="Final Accuracy (two-correct-one-wrong scenarios)",
        out_path=out_path_bar,
    )

    # ---- 各 run ごとのエージェント別グラフ ---- #
    for r_data in all_runs_data:
        tag = r_data["tag"]
        per_agent_acc = r_data["metrics"]["per_agent_turn_accuracy_all"]
        per_agent_change = r_data["metrics"]["per_agent_turn_change_rate_all"]
        out_acc = os.path.join(pair_out_dir, f"per_agent_turn_accuracy_{tag}.png")
        out_change = os.path.join(pair_out_dir, f"per_agent_turn_change_rate_{tag}.png")
        plot_per_agent_turn_stats_single(
            per_agent_acc,
            per_agent_change,
            title_prefix=f"{tag} (two-correct-one-wrong, all target)",
            out_path_acc=out_acc,
            out_path_change=out_change,
        )

    # ---- JSON 保存 ---- #
    out_json = os.path.join(pair_out_dir, "analysis_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved analysis results to {out_json}")


if __name__ == "__main__":
    main()
