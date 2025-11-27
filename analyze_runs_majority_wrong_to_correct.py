#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
初回に「多数派が誤答を選んでいる」が、かつ「少なくとも 1 人は正答を選んでいる」
問題だけを対象に、最終的に正答に到達した確率を評価するスクリプト。

2つの run ログを比較し、以下を計算:

1. 各 run ごとに：
   - initial_answer で「多数派が誤答」かつ「1人以上は正答」のシナリオのみを抽出
   - そのシナリオ集合における最終解答の正解率を算出

2. 両方の run でこの条件を満たしている同一問題のみを対象に、
   - それぞれの run における最終正解率を算出

3. 上記シナリオ集合について、
   - 各ターンでの各エージェントの answer から多数決を取り、
   - ターンごとの多数決正解率を算出・可視化（PNG保存）

結果は標準出力に加え、
  analysis_outputs/<run1>__vs__<run2>[_maxNNN]/
の下に JSON と PNG として保存する。

使い方:
    python analyze_runs_majority_wrong_to_correct.py RUN1 RUN2
    python analyze_runs_majority_wrong_to_correct.py RUN1 RUN2 --max-problem-index 200
"""

import argparse
import json
import os
import re
from glob import glob
from collections import Counter, defaultdict
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

    想定フォーマット:
        {"question_id": 1, "pred": "B", "gold": "B", "correct": true}

    ID マッピング:
        question_id=1 -> キー "1" と "problem_001" の両方で参照可能にする。
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
        # question_id を含めて柔軟に取得
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

        # 数値の場合: 1 -> "1", "problem_001"
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
        "turn_answers": {turn: {agent: "A"/...}}
    }
    を返す。
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

        # turn 0 レコードを探す
        for rec in records:
            if rec.get("turn") == 0 and rec.get("initial_answers"):
                ia = rec["initial_answers"]
                for agent, info in ia.items():
                    ans = info.get("answer")
                    if isinstance(ans, str):
                        initial_answers[agent] = ans.strip()
                turn_answers[0] = dict(initial_answers)
                break

        if not initial_answers:
            continue

        # 各 turn の answer を抽出
        for rec in records:
            turn = rec.get("turn")
            if turn is None or turn == 0:
                continue

            answers_by_agent: Dict[str, str] = {}

            if "consensus_state" in rec and isinstance(rec["consensus_state"], dict):
                for agent, info in rec["consensus_state"].items():
                    ans = info.get("answer")
                    if isinstance(ans, str):
                        answers_by_agent[agent] = ans.strip()
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


def make_pair_out_dir(base_out_dir: str, run1_dir: str, run2_dir: str, max_idx: Optional[int]) -> str:
    """analysis_outputs 以下に run1_vs_run2[_maxNNN] のサブフォルダを作成して返す。"""
    r1_tag = os.path.basename(run1_dir.rstrip(os.sep))
    r2_tag = os.path.basename(run2_dir.rstrip(os.sep))
    pair_name = f"{r1_tag}__vs__{r2_tag}"
    if max_idx is not None:
        pair_name += f"_max{max_idx:03d}"
    out_dir = os.path.join(base_out_dir, pair_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def scenario_list_with_index(pids: List[str]) -> List[Dict[str, Any]]:
    """['problem_001', ...] を [{pid, index}, ...] に変換してソート。"""
    items = []
    for pid in pids:
        idx = pid_to_index(pid)
        items.append({"pid": pid, "index": idx})
    items.sort(key=lambda x: (x["index"] is None, x["index"] if x["index"] is not None else 0))
    return items


# ------------- シナリオ選択 & 集計 ------------- #

def select_initial_majority_wrong(
    problems: Dict[str, Dict[str, Any]],
    acc_by_pid: Dict[str, Dict[str, Any]],
) -> List[str]:
    """
    「初回に多数派が誤答を選択しており、かつ少なくとも1人は正答を選択している」
    problem_id を返す。

    条件:
      - initial_answers が存在
      - 同じ選択肢に 2票以上入っている（多数派）
      - その多数派の選択肢 != gold
      - かつ answers の中に gold を選んでいるエージェントが 1人以上いる
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

        answers = list(initial.values())
        counts = Counter(answers)
        if not counts:
            continue

        most_common = counts.most_common()
        # 同率トップなら多数派なしとみなす
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            continue

        maj_answer, maj_count = most_common[0]
        # 明確な多数派（2票以上）
        if maj_count < 2:
            continue

        # 多数派が誤答
        if maj_answer == gold:
            continue

        # 少なくとも1人は正答を選んでいる
        has_at_least_one_correct = any(a == gold for a in answers)
        if not has_at_least_one_correct:
            continue

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


def plot_turn_accuracy(
    run1_turn_acc: Dict[int, float],
    run2_turn_acc: Dict[int, float],
    title: str,
    out_path: str,
    label1: str,
    label2: str,
) -> None:
    """ターンごとの accuracy を2本の線でプロットして保存。"""
    plt.figure()
    has_label = False

    if run1_turn_acc:
        xs1 = sorted(run1_turn_acc.keys())
        ys1 = [run1_turn_acc[x] for x in xs1]
        plt.plot(xs1, ys1, marker="o", label=label1)
        has_label = True
    if run2_turn_acc:
        xs2 = sorted(run2_turn_acc.keys())
        ys2 = [run2_turn_acc[x] for x in xs2]
        plt.plot(xs2, ys2, marker="o", linestyle="--", label=label2)
        has_label = True

    plt.xlabel("Turn")
    plt.ylabel("Accuracy (majority vote)")
    plt.title(title)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    if has_label:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


# ------------- メイン ------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run1", help="1つ目の run ディレクトリ or run ID (e.g., run_20251120_194122)")
    parser.add_argument("run2", help="2つ目の run ディレクトリ or run ID (e.g., run_20251120_171903_roundrobin)")
    parser.add_argument(
        "--out-dir",
        default="analysis_outputs",
        help="結果を出力するディレクトリ (default: analysis_outputs)",
    )
    parser.add_argument(
        "--max-problem-index",
        type=int,
        default=None,
        help="problem_001〜problem_N までのみを対象にする場合の N (例: 200)",
    )
    args = parser.parse_args()

    run1_dir = resolve_run_dir(args.run1)
    run2_dir = resolve_run_dir(args.run2)

    base_out_dir = args.out_dir
    os.makedirs(base_out_dir, exist_ok=True)
    pair_out_dir = make_pair_out_dir(base_out_dir, run1_dir, run2_dir, args.max_problem_index)

    print(f"[INFO] Run1 dir: {run1_dir}")
    print(f"[INFO] Run2 dir: {run2_dir}")
    print(f"[INFO] Output dir: {pair_out_dir}")
    if args.max_problem_index is not None:
        print(f"[INFO] Using problems up to problem_{args.max_problem_index:03d}")

    # ---- ログ読み込み ---- #
    print("[INFO] Loading accuracy logs...")
    acc1_full = load_accuracy(run1_dir)
    acc2_full = load_accuracy(run2_dir)

    print("[INFO] Loading discussion logs...")
    probs1_full = load_discussion(run1_dir)
    probs2_full = load_discussion(run2_dir)

    print(f"[DEBUG] Run1: #accuracy entries = {len(acc1_full)}, #discussion problems = {len(probs1_full)}")
    print(f"[DEBUG] Run2: #accuracy entries = {len(acc2_full)}, #discussion problems = {len(probs2_full)}")

    inter1 = set(acc1_full.keys()) & set(probs1_full.keys())
    inter2 = set(acc2_full.keys()) & set(probs2_full.keys())
    print(f"[DEBUG] Run1: #overlap between accuracy & discussion IDs = {len(inter1)}")
    print(f"[DEBUG] Run2: #overlap between accuracy & discussion IDs = {len(inter2)}")
    if inter1:
        print(f"[DEBUG] Run1 example overlapping IDs: {sorted(list(inter1))[:5]}")
    if inter2:
        print(f"[DEBUG] Run2 example overlapping IDs: {sorted(list(inter2))[:5]}")

    # ---- problem_001〜problem_N のみを対象に絞り込み ---- #
    probs1, acc1 = filter_problems_by_max_index(probs1_full, acc1_full, args.max_problem_index)
    probs2, acc2 = filter_problems_by_max_index(probs2_full, acc2_full, args.max_problem_index)

    # ---- 「初回多数派誤答 + 少なくとも1人正答」シナリオの抽出 ---- #
    maj_wrong1 = select_initial_majority_wrong(probs1, acc1)
    maj_wrong2 = select_initial_majority_wrong(probs2, acc2)
    common_pids = sorted(set(maj_wrong1) & set(maj_wrong2))

    print(f"[INFO] Run1: #initial-majority-wrong-with-one-correct scenarios = {len(maj_wrong1)}")
    if maj_wrong1:
        print("[INFO] Run1 scenarios (problem index):")
        for item in scenario_list_with_index(maj_wrong1):
            print(f"  - {item['pid']} (index={item['index']})")

    print(f"[INFO] Run2: #initial-majority-wrong-with-one-correct scenarios = {len(maj_wrong2)}")
    if maj_wrong2:
        print("[INFO] Run2 scenarios (problem index):")
        for item in scenario_list_with_index(maj_wrong2):
            print(f"  - {item['pid']} (index={item['index']})")

    print(f"[INFO] Common scenarios (in both runs) = {len(common_pids)}")
    if common_pids:
        print("[INFO] Common scenarios (problem index):")
        for item in scenario_list_with_index(common_pids):
            print(f"  - {item['pid']} (index={item['index']})")

    # ---- 各 run の最終正解率（run ごとに条件を満たすシナリオ全体） ---- #
    acc_final_run1 = compute_final_accuracy_for_set(maj_wrong1, acc1)
    acc_final_run2 = compute_final_accuracy_for_set(maj_wrong2, acc2)
    print(f"[RESULT] Run1 final accuracy (initial-majority-wrong-with-one-correct scenarios): {acc_final_run1:.3f}")
    print(f"[RESULT] Run2 final accuracy (initial-majority-wrong-with-one-correct scenarios): {acc_final_run2:.3f}")

    # ---- 共通シナリオでの最終正解率 ---- #
    acc_final_run1_common = compute_final_accuracy_for_set(common_pids, acc1)
    acc_final_run2_common = compute_final_accuracy_for_set(common_pids, acc2)
    print(f"[RESULT] Run1 final accuracy (common initial-majority-wrong-with-one-correct scenarios): {acc_final_run1_common:.3f}")
    print(f"[RESULT] Run2 final accuracy (common initial-majority-wrong-with-one-correct scenarios): {acc_final_run2_common:.3f}")

    # ---- ターンごとの多数決 accuracy （run ごとのシナリオ集合） ---- #
    turn_acc_run1_all = compute_turnwise_majority_accuracy(maj_wrong1, probs1, acc1)
    turn_acc_run2_all = compute_turnwise_majority_accuracy(maj_wrong2, probs2, acc2)
    out_path_all = os.path.join(pair_out_dir, "turn_accuracy_initial_majority_wrong_with_one_correct_runwise.png")
    plot_turn_accuracy(
        turn_acc_run1_all,
        turn_acc_run2_all,
        title="Turn-wise Majority Accuracy (initial majority wrong, one correct, each run)",
        out_path=out_path_all,
        label1="Run1",
        label2="Run2",
    )

    # ---- 共通シナリオに絞ったターンごとの多数決 accuracy ---- #
    turn_acc_run1_common = compute_turnwise_majority_accuracy(common_pids, probs1, acc1)
    turn_acc_run2_common = compute_turnwise_majority_accuracy(common_pids, probs2, acc2)
    out_path_common = os.path.join(pair_out_dir, "turn_accuracy_initial_majority_wrong_with_one_correct_common.png")
    plot_turn_accuracy(
        turn_acc_run1_common,
        turn_acc_run2_common,
        title="Turn-wise Majority Accuracy (common initial majority wrong, one correct)",
        out_path=out_path_common,
        label1="Run1 (common)",
        label2="Run2 (common)",
    )

    # ---- JSON に保存 ---- #
    scenarios_info = {
        "run1": {
            "dir": run1_dir,
            "initial_majority_wrong_with_one_correct_scenarios": scenario_list_with_index(maj_wrong1),
        },
        "run2": {
            "dir": run2_dir,
            "initial_majority_wrong_with_one_correct_scenarios": scenario_list_with_index(maj_wrong2),
        },
        "common": {
            "scenarios": scenario_list_with_index(common_pids),
        },
    }

    metrics_info = {
        "run1": {
            "final_accuracy_initial_majority_wrong_with_one_correct_all": acc_final_run1,
            "final_accuracy_initial_majority_wrong_with_one_correct_common": acc_final_run1_common,
            "turn_accuracy_all": turn_acc_run1_all,
            "turn_accuracy_common": turn_acc_run1_common,
        },
        "run2": {
            "final_accuracy_initial_majority_wrong_with_one_correct_all": acc_final_run2,
            "final_accuracy_initial_majority_wrong_with_one_correct_common": acc_final_run2_common,
            "turn_accuracy_all": turn_acc_run2_all,
            "turn_accuracy_common": turn_acc_run2_common,
        },
    }

    out_scen = os.path.join(pair_out_dir, "majority_wrong_with_one_correct_scenarios.json")
    out_met = os.path.join(pair_out_dir, "majority_wrong_with_one_correct_metrics.json")
    with open(out_scen, "w", encoding="utf-8") as f:
        json.dump(scenarios_info, f, ensure_ascii=False, indent=2)
    with open(out_met, "w", encoding="utf-8") as f:
        json.dump(metrics_info, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved {os.path.basename(out_scen)} and {os.path.basename(out_met)} into {pair_out_dir}")


if __name__ == "__main__":
    main()
