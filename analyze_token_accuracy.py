#!/usr/bin/env python3
"""
Token-budget aligned accuracy curves from discussion logs.

Usage:
  python analyze_token_accuracy.py --runs run_20260125_075717 run_20260124_182857 --num_problems 100 --bucket 50 --out plots/token_accuracy.png

Assumptions:
  - Each run directory under ./logs/run_* contains problem_* subdirs with discussion_log.json.
  - Each discussion_log record has cumulative 'public_tokens_used'; if missing, we approximate with whitespace token counts.
  - Gold label is read from adversary_meta.json (key 'gold_label') when present; otherwise the problem is skipped for accuracy.
  - Majority answer per step is computed from agent_actions[*].action_plan.answer (A-D).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional


RUN_ROOT = Path(__file__).resolve().parent / "logs"


def load_discussion(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def majority_answer(actions: List[dict]) -> Optional[str]:
    answers = []
    for a in actions:
        ans = a.get("action_plan", {}).get("answer")
        if isinstance(ans, str):
            ans = ans.strip().upper()
            if ans in {"A", "B", "C", "D"}:
                answers.append(ans)
    if not answers:
        return None
    counts = Counter(answers)
    top = counts.most_common()
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def event_tokens(event: dict, fallback_so_far: int, tokenizer=None) -> int:
    if "public_tokens_used" in event:
        return int(event.get("public_tokens_used", fallback_so_far))
    text = event.get("content", "") or ""
    approx = len(text.split())
    return fallback_so_far + approx


def load_gold(problem_dir: Path) -> Optional[str]:
    meta = problem_dir / "adversary_meta.json"
    if not meta.exists():
        return None
    try:
        with meta.open("r", encoding="utf-8") as f:
            data = json.load(f)
        gold = data.get("gold_label")
        if isinstance(gold, str):
            gold = gold.strip().upper()
            if gold in {"A", "B", "C", "D"}:
                return gold
    except Exception:
        return None
    return None


def process_problem(log_path: Path, bucket: int) -> List[Tuple[int, Optional[str]]]:
    data = load_discussion(log_path)
    steps: List[Tuple[int, Optional[str]]] = []
    cum_tokens = 0
    for ev in data:
        if ev.get("event_type") not in {"utterance", "interrupt"}:
            continue
        cum_tokens = event_tokens(ev, cum_tokens)
        majority = majority_answer(ev.get("agent_actions", []))
        steps.append((cum_tokens, majority))
    if not steps:
        return []
    # bucketize
    bucketed: Dict[int, Optional[str]] = {}
    for t, ans in steps:
        b = (t // bucket) * bucket
        bucketed.setdefault(b, ans)
    return sorted(bucketed.items())


def aggregate_runs(run_ids: List[str], num_problems: int, bucket: int):
    results = {}
    for run_id in run_ids:
        run_dir = RUN_ROOT / run_id
        if not run_dir.exists():
            print(f"[warn] run dir not found: {run_dir}")
            continue
        probs = sorted(p for p in run_dir.glob("problem_*") if p.is_dir())
        probs = probs[:num_problems] if num_problems > 0 else probs
        bucket_correct: Counter = Counter()
        bucket_total: Counter = Counter()
        max_bucket = 0
        for pdir in probs:
            gold = load_gold(pdir)
            log_path = pdir / "discussion_log.json"
            if not log_path.exists():
                continue
            steps = process_problem(log_path, bucket)
            for b, ans in steps:
                max_bucket = max(max_bucket, b)
                if gold:
                    bucket_total[b] += 1
                    if ans == gold:
                        bucket_correct[b] += 1
        xs = list(range(0, max_bucket + bucket, bucket))
        acc = []
        for x in xs:
            if bucket_total[x]:
                acc.append(bucket_correct[x] / bucket_total[x])
            else:
                acc.append(None)
        results[run_id] = (xs, acc)
    return results


def plot(results: Dict[str, Tuple[List[int], List[Optional[float]]]], bucket: int, out: Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for run_id, (xs, acc) in results.items():
        ys = [a if a is not None else 0 for a in acc]
        plt.plot(xs, ys, label=run_id)
    plt.xlabel(f"Public tokens (bucket={bucket})")
    plt.ylabel("Majority accuracy")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    print(f"[info] saved plot -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run IDs under ./logs/")
    ap.add_argument("--num_problems", type=int, default=-1, help="number of problems to analyze (<=0 means all)")
    ap.add_argument("--bucket", type=int, default=50, help="token bucket size")
    ap.add_argument("--out", type=Path, default=Path("analysis_outputs/token_accuracy.png"))
    args = ap.parse_args()
    res = aggregate_runs(args.runs, args.num_problems, args.bucket)
    if not res:
        print("[warn] no data to plot")
        return
    plot(res, args.bucket, args.out)


if __name__ == "__main__":
    main()
