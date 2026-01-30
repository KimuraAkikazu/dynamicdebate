#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Token-wise majority accuracy (forward-fill, tie=incorrect, token0=0).

仕様（本バージョン）
- turn0 (public_tokens_used=0) では initial_answers を保持し、score は必ず 0.0。
- forward-fill: 観測された最新 snapshot の回答を、その token まで適用する。
- 多数決同票（3人バラバラ等）は None → 不正解扱い。
- 初回回答で 3 人全員バラバラの問題は分析対象から除外。
- --max-problem-index で指定した件数に達したところでシナリオ抽出を打ち切る。
"""

import argparse
import json
import os
import re
from glob import glob
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import random

import matplotlib.pyplot as plt

RNG_SEED = 42


# ---------------- ユーティリティ ---------------- #
def resolve_run_dir(run_arg: str) -> str:
    if os.path.isdir(run_arg):
        return os.path.abspath(run_arg)
    candidate = os.path.join("logs", run_arg)
    if os.path.isdir(candidate):
        return os.path.abspath(candidate)
    raise FileNotFoundError(f"Run directory not found: {run_arg} or logs/{run_arg}")


def find_accuracy_file(run_dir: str) -> Optional[str]:
    for name in ["accuracy_log.jsonl", "accuracy_log.json"]:
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    return None


def load_accuracy(run_dir: str) -> Dict[str, Dict[str, Any]]:
    acc_path = find_accuracy_file(run_dir)
    if acc_path is None:
        raise FileNotFoundError(f"accuracy_log not found in {run_dir}")
    data: List[Dict[str, Any]] = []
    if acc_path.endswith(".jsonl"):
        with open(acc_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        with open(acc_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                data = loaded
            elif isinstance(loaded, dict):
                for pid, info in loaded.items():
                    info = dict(info)
                    info.setdefault("problem_id", pid)
                    data.append(info)

    acc_by_pid: Dict[str, Dict[str, Any]] = {}
    for row in data:
        pid_raw = row.get("problem_id") or row.get("qid") or row.get("id") or row.get("question_id")
        if pid_raw is None:
            continue
        gold = row.get("gold") or row.get("gold_answer") or row.get("label") or row.get("answer")
        final_answer = row.get("final_answer") or row.get("pred") or row.get("model_answer")
        is_correct = row.get("is_correct") or row.get("correct")
        entry = {
            "gold": gold,
            "final_answer": final_answer,
            "is_correct": bool(is_correct) if is_correct is not None else (final_answer == gold if gold and final_answer else None),
        }
        if isinstance(pid_raw, int):
            pid = f"problem_{pid_raw:03d}"
            acc_by_pid[pid] = entry
            acc_by_pid[str(pid_raw)] = entry
        else:
            s = str(pid_raw)
            acc_by_pid[s] = entry
            m = re.match(r"^problem_(\d+)$", s)
            if m:
                idx = int(m.group(1))
                acc_by_pid[str(idx)] = entry
                acc_by_pid[f"problem_{idx:03d}"] = entry
    return acc_by_pid


def _looks_like_exec_log(obj: Any) -> bool:
    return isinstance(obj, list) and obj and isinstance(obj[0], dict) and ("event_type" in obj[0]) and ("turn" in obj[0])


def find_exec_log_file(problem_dir: str) -> Optional[str]:
    for name in ["execution_log.json", "event_log.json", "run_log.json", "discussion_log.json"]:
        path = os.path.join(problem_dir, name)
        if os.path.isfile(path):
            return path
    for p in sorted(glob(os.path.join(problem_dir, "*.json"))):
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if _looks_like_exec_log(obj):
                return p
        except Exception:
            continue
    return None


def normalize_turn(turn: Any) -> int:
    if isinstance(turn, int):
        return turn
    if isinstance(turn, str) and turn.isdigit():
        return int(turn)
    return 10**9


def load_exec_logs(run_dir: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    problem_dirs = sorted(glob(os.path.join(run_dir, "problem_*")))

    for pdir in problem_dirs:
        pid = os.path.basename(pdir)
        log_path = find_exec_log_file(pdir)
        if log_path is None:
            continue
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            continue
        if not _looks_like_exec_log(records):
            continue

        initial_answers: Dict[str, str] = {}
        token_budget: Optional[int] = None
        for rec in records:
            if isinstance(rec, dict) and isinstance(rec.get("initial_answers"), dict):
                ia = rec["initial_answers"]
                for ag, info in ia.items():
                    if isinstance(info, dict):
                        ans = info.get("answer")
                        if isinstance(ans, str):
                            initial_answers[ag] = ans.strip()
                tb = rec.get("public_token_budget")
                if isinstance(tb, int):
                    token_budget = tb
                break
        if not initial_answers:
            continue

        agents = sorted(initial_answers.keys())
        current_answers = dict(initial_answers)
        snapshots: List[Dict[str, Any]] = []

        for rec in records:
            if not isinstance(rec, dict):
                continue
            event_type = rec.get("event_type")
            speaker = rec.get("speaker")
            turn = rec.get("turn")
            turn_sort = normalize_turn(turn)
            tokens_used = rec.get("public_tokens_used")
            if tokens_used is None:
                if turn == 0:
                    tokens_used = 0
            if not isinstance(tokens_used, int):
                tokens_used = None

            # 更新ロジック: tokens_used==0 では plan の answer を無視
            agent_actions = rec.get("agent_actions")
            freeze_answers = (tokens_used == 0)
            if isinstance(agent_actions, list):
                for aa in agent_actions:
                    if not isinstance(aa, dict):
                        continue
                    ag = aa.get("agent_name")
                    ap = aa.get("action_plan")
                    if ag in agents and isinstance(ap, dict):
                        ans = ap.get("answer")
                        if (not freeze_answers) and isinstance(ans, str):
                            current_answers[ag] = ans.strip()

            if tokens_used is not None:
                snapshots.append({
                    "turn": turn,
                    "turn_sort": turn_sort,
                    "tokens": tokens_used,
                    "answers": dict(current_answers),
                    "speaker": speaker,
                    "event_type": event_type,
                })

            tb = rec.get("public_token_budget")
            if token_budget is None and isinstance(tb, int):
                token_budget = tb

        if not snapshots:
            snapshots = [{
                "turn": 0,
                "turn_sort": 0,
                "tokens": 0,
                "answers": dict(initial_answers),
                "speaker": None,
                "event_type": "plan",
            }]

        snapshots_sorted = sorted(snapshots, key=lambda x: (x.get("tokens", 0), x.get("turn_sort", 10**9)))
        # 同 token は後勝ち
        dedup = {}
        for s in snapshots_sorted:
            t = s["tokens"]
            dedup[t] = s
        snapshots_sorted = [dedup[t] for t in sorted(dedup.keys())]

        result[pid] = {
            "agents": agents,
            "initial_answers": initial_answers,
            "token_budget": token_budget,
            "snapshots": snapshots_sorted,
        }
    return result


# ---------------- シナリオ抽出 ---------------- #
def select_scenarios(problems: Dict[str, Dict[str, Any]], acc_by_pid: Dict[str, Dict[str, Any]], scenario: str) -> List[str]:
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
        correct_cnt = sum(1 for ans in initial.values() if isinstance(ans, str) and ans == gold)
        wrong_cnt = len(initial) - correct_cnt
        if scenario == "two_wrong_one_correct":
            if correct_cnt == 1 and wrong_cnt == 2:
                selected.append(pid)
        elif scenario == "two_correct_one_wrong":
            if correct_cnt == 2 and wrong_cnt == 1:
                selected.append(pid)
    return selected


# ---------------- メトリクス計算 ---------------- #
def majority_with_random_tie(answers: List[Optional[str]], rng: random.Random) -> Optional[str]:
    filtered = [a for a in answers if isinstance(a, str) and a.strip()]
    if not filtered:
        return None
    counts = Counter(filtered)
    most = counts.most_common()
    top_freq = most[0][1]
    tied = [a for a, c in most if c == top_freq]
    return rng.choice(tied)


def majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    """単純多数決。同票は None。"""
    filtered = [a for a in answers if isinstance(a, str) and a.strip()]
    if not filtered:
        return None
    counts = Counter(filtered)
    most = counts.most_common()
    if len(most) == 1:
        return most[0][0]
    if most[0][1] == most[1][1]:
        return None
    return most[0][0]


def compute_tokenwise_majority_accuracy_forward_fill(
    pids: List[str],
    problems: Dict[str, Dict[str, Any]],
    acc_by_pid: Dict[str, Dict[str, Any]],
    token_step: int,
) -> Tuple[List[int], List[float]]:
    """
    forward-fill で token-wise 正解率を計算する。
    - token=0 は常に 0.0
    - 同票は不正解扱い
    """
    if not pids:
        return [], []
    max_token = 0
    for pid in pids:
        pdata = problems.get(pid)
        if not pdata:
            continue
        snaps = pdata.get("snapshots", [])
        if snaps:
            max_token = max(max_token, snaps[-1].get("tokens", 0) or 0)
        tb = pdata.get("token_budget")
        if isinstance(tb, int):
            max_token = max(max_token, tb)

    tokens_out: List[int] = list(range(0, max_token + 1, token_step))
    series: List[float] = []

    for tok in tokens_out:
        if tok == 0:
            series.append(0.0)
            continue
        correct = 0
        total = 0
        for pid in pids:
            pdata = problems.get(pid)
            acc = acc_by_pid.get(pid)
            if not pdata or not acc:
                continue
            gold = acc.get("gold")
            if not isinstance(gold, str):
                continue
            snaps = pdata.get("snapshots", [])
            if not snaps:
                continue
            # forward-fill: tokens <= tok の最新を使う
            use = snaps[0]
            for s in snaps:
                if isinstance(s.get("tokens"), int) and s["tokens"] <= tok:
                    use = s
                else:
                    break
            ans_map = use.get("answers", {}) or {}
            maj = majority_vote([ans_map.get(a) for a in pdata.get("agents", [])])
            total += 1
            if maj == gold:
                correct += 1
        series.append((correct / total) if total else 0.0)
    return tokens_out, series


def compute_final_accuracy(pids: List[str], acc_by_pid: Dict[str, Dict[str, Any]]) -> float:
    correct = 0
    total = 0
    for pid in pids:
        acc = acc_by_pid.get(pid)
        if not acc:
            continue
        ic = acc.get("is_correct")
        if ic is None:
            gold = acc.get("gold")
            fa = acc.get("final_answer")
            ic = (gold == fa) if isinstance(gold, str) and isinstance(fa, str) else None
        if ic is None:
            continue
        total += 1
        if ic:
            correct += 1
    return (correct / total) if total else 0.0


# ---------------- プロット ---------------- #
def plot_token_accuracy_multi(data_list: List[Dict[str, Any]], title: str, out_path: str) -> None:
    plt.figure(figsize=(8, 5))
    for item in data_list:
        xs = item["tokens"]
        ys = item["series"]
        label = item["label"]
        plt.plot(xs, ys, marker="o", linewidth=1.2, markersize=3, markevery=1, label=label)
    plt.xlabel("Public tokens used")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(fontsize="small")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


# ---------------- メイン ---------------- #
def main():
    parser = argparse.ArgumentParser(description="Token-wise majority accuracy (backfill, random tie) analyzer.")
    parser.add_argument("runs", nargs="+", help="Run directories or IDs")
    parser.add_argument("--out-dir", default="analysis_outputs", help="Output directory")
    parser.add_argument("--max-problem-index", type=int, default=None, help="Limit problems to problem_N")
    parser.add_argument(
        "--scenario",
        default="two_wrong_one_correct",
        choices=["two_wrong_one_correct", "two_correct_one_wrong"],
        help="Scenario filter",
    )
    parser.add_argument("--token-step", type=int, default=25, help="Token step")
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="Random seed for tie break")
    args = parser.parse_args()

    run_dirs = [resolve_run_dir(r) for r in args.runs]
    os.makedirs(args.out_dir, exist_ok=True)

    out_dir = os.path.join(args.out_dir, "__vs__".join(os.path.basename(r) for r in run_dirs) + f"_max{args.max_problem_index or 'all'}_{args.scenario}_v2")
    os.makedirs(out_dir, exist_ok=True)

    plot_data = []
    json_output: Dict[str, Any] = {"scenario": args.scenario, "runs": {}}

    for r_dir in run_dirs:
        tag = os.path.basename(r_dir.rstrip(os.sep))
        acc_full = load_accuracy(r_dir)
        probs_full = load_exec_logs(r_dir)
        # max index filterは後段でシナリオ数に対して適用するためここではそのまま
        probs = probs_full
        acc = acc_full

        scenarios = select_scenarios(probs, acc, args.scenario)
        # 3人バラバラは除外
        filtered = []
        for pid in scenarios:
            ia = probs.get(pid, {}).get("initial_answers", {})
            uniq = set(ia.values())
            if len(uniq) == 3:
                continue
            filtered.append(pid)
        # 上限件数に達したら切り詰め
        if args.max_problem_index is not None:
            filtered = filtered[: args.max_problem_index]
        scenarios = filtered
        final_acc = compute_final_accuracy(scenarios, acc)
        tokens, series = compute_tokenwise_majority_accuracy_forward_fill(scenarios, probs, acc, args.token_step)

        plot_data.append({"label": tag, "tokens": tokens, "series": series})
        json_output["runs"][tag] = {
            "target_scenarios": scenarios,
            "metrics": {
                "final_accuracy_all_target": final_acc,
                "token_majority_tokens_all": tokens,
                "token_majority_accuracy_all": series,
            },
        }
        print(f"[RESULT] {tag} Final Acc: {final_acc:.3f}")

    out_plot = os.path.join(out_dir, "token_accuracy_runwise_v2.png")
    plot_token_accuracy_multi(plot_data, f"Token-wise Majority Accuracy (backfill, step={args.token_step})", out_plot)

    out_json = os.path.join(out_dir, "analysis_results_v2.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved analysis results to {out_json}")


if __name__ == "__main__":
    main()
