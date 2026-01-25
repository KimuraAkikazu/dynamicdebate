#!/usr/bin/env python3
"""
Comprehensive log analysis for multi-agent debate runs.

Features
- Public-token bucketed majority accuracy (AUC-style) per run (like analyze_token_accuracy).
- Model-side token usage (prompt/completion/total) aggregated per run/problem/agent using prompt_log JSONL.
- Token efficiency: public_tokens_used vs model total tokens.
- Early-correction metric: public tokens until majority first matches gold.
- Interrupt helpfulness: fraction of interrupt events that move majority toward gold.

Usage
  python analyze_token_usage.py --runs run_20260125_075717 run_20260125_074300 \
      --num_problems 200 --bucket 50 \
      --out_dir analysis_outputs/token_usage

Outputs (under out_dir)
- token_accuracy.png        : public-token bucketed majority accuracy curves for all runs
- token_efficiency.png      : bar of model_total_tokens vs public_tokens (per run, averaged per problem)
- summary.json              : machine-readable metrics per run

Notes
- Expects each run dir: logs/run_*/problem_*/discussion_log.json and prompt_log_*.jsonl.
- Gold label is taken from adversary_meta.json (gold_label). Problems without gold are skipped for accuracy metrics.
- When public_tokens_used is missing in discussion_log, we approximate with whitespace token counts.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


RUN_ROOT = Path(__file__).resolve().parent / "logs"


# ------------------------ I/O helpers ------------------------ #
def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def find_prompt_log(problem_dir: Path) -> Optional[Path]:
    logs = sorted(problem_dir.glob("prompt_log_*.jsonl"))
    return logs[-1] if logs else None


# ------------------------ majority helpers ------------------------ #
def majority_answer(actions: List[dict]) -> Optional[str]:
    answers: List[str] = []
    for a in actions:
        ans = a.get("action_plan", {}).get("answer") if isinstance(a, dict) else None
        if isinstance(ans, str):
            ans = ans.strip().upper()
            if ans in {"A", "B", "C", "D"}:
                answers.append(ans)
    if not answers:
        return None
    cnt = Counter(answers)
    top = cnt.most_common()
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def bucketize_steps(steps: List[Tuple[int, Optional[str]]], bucket: int) -> Dict[int, Optional[str]]:
    buckets: Dict[int, Optional[str]] = {}
    for t, ans in steps:
        b = (t // bucket) * bucket
        buckets.setdefault(b, ans)
    return buckets


# ------------------------ parsing per problem ------------------------ #
def load_gold(problem_dir: Path) -> Optional[str]:
    meta = problem_dir / "adversary_meta.json"
    if not meta.exists():
        return None
    try:
        data = load_json(meta)
        gold = data.get("gold_label")
        if isinstance(gold, str):
            gold = gold.strip().upper()
            if gold in {"A", "B", "C", "D"}:
                return gold
    except Exception:
        return None
    return None


def parse_discussion(problem_dir: Path, bucket: int):
    log_path = problem_dir / "discussion_log.json"
    if not log_path.exists():
        return None
    data = load_json(log_path)
    steps: List[Tuple[int, Optional[str], str]] = []  # (tokens, majority, event_type)
    cum_tokens = 0
    last_majority: Optional[str] = None
    interrupts_helpful = []  # (event_type, majority_change)

    for ev in data:
        et = ev.get("event_type")
        if et not in {"utterance", "interrupt"}:
            continue
        if "public_tokens_used" in ev:
            cum_tokens = int(ev.get("public_tokens_used", cum_tokens))
        else:
            text = ev.get("content", "") or ""
            cum_tokens += max(1, len(text.split()))
        maj = majority_answer(ev.get("agent_actions", []))
        if et == "interrupt":
            if last_majority != maj:
                interrupts_helpful.append((et, last_majority, maj))
        last_majority = maj
        steps.append((cum_tokens, maj, et))

    buckets = bucketize_steps([(t, m) for t, m, _ in steps], bucket)
    return {
        "buckets": buckets,
        "steps": steps,
        "max_tokens": cum_tokens,
        "interrupt_changes": interrupts_helpful,
    }


def parse_prompt_tokens(prompt_log: Path):
    agg = defaultdict(lambda: defaultdict(int))  # agent -> metric -> tokens
    totals = defaultdict(int)
    for rec in load_jsonl(prompt_log):
        agent = rec.get("agent", "?")
        stats = rec.get("token_stats") or {}
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            agg[agent][k] += int(stats.get(k, 0))
            totals[k] += int(stats.get(k, 0))
    return agg, totals


def process_problem(problem_dir: Path, bucket: int):
    gold = load_gold(problem_dir)
    disc = parse_discussion(problem_dir, bucket)
    if not disc:
        return None
    prompt_log = find_prompt_log(problem_dir)
    agent_tok = {}
    model_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if prompt_log:
        agent_tok, model_totals = parse_prompt_tokens(prompt_log)
    return {
        "gold": gold,
        "buckets": disc["buckets"],
        "steps": disc["steps"],
        "max_tokens": disc["max_tokens"],
        "interrupt_changes": disc["interrupt_changes"],
        "agent_tok": agent_tok,
        "model_totals": model_totals,
    }


# ------------------------ aggregation per run ------------------------ #
def aggregate_run(run_dir: Path, num_problems: int, bucket: int):
    problems = sorted(p for p in run_dir.glob("problem_*") if p.is_dir())
    if num_problems > 0:
        problems = problems[:num_problems]

    bucket_correct = Counter()
    bucket_total = Counter()
    max_bucket = 0
    model_tokens_sum = Counter()
    public_tokens_sum = 0
    public_tokens_count = 0
    interrupts_toward = 0
    interrupts_total = 0
    tokens_to_first_correct = []

    for pdir in problems:
        res = process_problem(pdir, bucket)
        if not res:
            continue
        gold = res["gold"]
        # bucket accuracy
        for b, ans in res["buckets"].items():
            max_bucket = max(max_bucket, b)
            if gold:
                bucket_total[b] += 1
                if ans == gold:
                    bucket_correct[b] += 1
        # tokens to first correct majority
        if gold:
            for t, maj, _ in res["steps"]:
                if maj == gold:
                    tokens_to_first_correct.append(t)
                    break
        # interrupts
        for _, before, after in res["interrupt_changes"]:
            interrupts_total += 1
            if gold and before != gold and after == gold:
                interrupts_toward += 1
        # token usage
        model_tokens_sum.update(res["model_totals"])
        public_tokens_sum += res["max_tokens"]
        public_tokens_count += 1

    xs = list(range(0, max_bucket + bucket, bucket))
    acc_curve = []
    for x in xs:
        if bucket_total[x]:
            acc_curve.append(bucket_correct[x] / bucket_total[x])
        else:
            acc_curve.append(None)

    summary = {
        "problems": len(problems),
        "model_tokens": dict(model_tokens_sum),
        "avg_public_tokens": public_tokens_sum / public_tokens_count if public_tokens_count else 0,
        "interrupt_help_rate": (interrupts_toward / interrupts_total) if interrupts_total else None,
        "tokens_to_first_correct_avg": sum(tokens_to_first_correct) / len(tokens_to_first_correct) if tokens_to_first_correct else None,
    }

    return {
        "xs": xs,
        "acc": acc_curve,
        "summary": summary,
    }


# ------------------------ plotting ------------------------ #
def plot_accuracy(run_results: Dict[str, dict], bucket: int, out_path: Path):
    plt.figure(figsize=(10, 6))
    for run_id, res in run_results.items():
        ys = [y if y is not None else 0 for y in res["acc"]]
        plt.plot(res["xs"], ys, label=run_id)
    plt.xlabel(f"Public tokens (bucket={bucket})")
    plt.ylabel("Majority accuracy")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[info] saved {out_path}")


def plot_efficiency(run_results: Dict[str, dict], out_path: Path):
    labels = []
    model_tokens = []
    public_tokens = []
    for run_id, res in run_results.items():
        labels.append(run_id)
        mt = res["summary"]["model_tokens"].get("total_tokens", 0)
        model_tokens.append(mt)
        public_tokens.append(res["summary"].get("avg_public_tokens", 0))

    x = range(len(labels))
    plt.figure(figsize=(10, 6))
    plt.bar(x, model_tokens, width=0.4, label="Model total tokens")
    plt.bar([i + 0.4 for i in x], public_tokens, width=0.4, label="Avg public tokens/problem")
    plt.xticks([i + 0.2 for i in x], labels, rotation=45, ha="right")
    plt.ylabel("Tokens")
    plt.title("Token efficiency")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[info] saved {out_path}")


# ------------------------ main ------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run IDs under ./logs")
    ap.add_argument("--num_problems", type=int, default=-1, help="problems to analyze (-1 for all)")
    ap.add_argument("--bucket", type=int, default=50, help="public token bucket size")
    ap.add_argument("--out_dir", type=Path, default=Path("analysis_outputs/token_usage"))
    args = ap.parse_args()

    run_results: Dict[str, dict] = {}
    for rid in args.runs:
        run_dir = RUN_ROOT / rid
        if not run_dir.exists():
            print(f"[warn] skip missing run {rid}")
            continue
        run_results[rid] = aggregate_run(run_dir, args.num_problems, args.bucket)

    if not run_results:
        print("[warn] no runs processed")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy(run_results, args.bucket, args.out_dir / "token_accuracy.png")
    plot_efficiency(run_results, args.out_dir / "token_efficiency.png")

    # dump summary
    summary = {rid: res["summary"] for rid, res in run_results.items()}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] saved summary -> {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

