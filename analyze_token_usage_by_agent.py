#!/usr/bin/env python3
"""
Token-wise accuracy analysis per agent (no consensus dependency).

What it does
- Reads discussion logs such as logs/run_20260125_060659 or logs/run_20260125_060849.
- Collects intermediate answers from agent_actions["answer"] at every turn.
- Derives interrupt events even when event_type stays "utterance":
  if an agent chose action_plan.action == "interrupt" on turn t and becomes the
  speaker on turn t+1, that turn is treated as an interrupt.
- Uses initial_answers as 0-token answers and carries forward prior answers when a
  speaker does not supply an answer.
- Computes accuracy vs gold label bucketed by public tokens.
- Aggregates model-token usage from prompt_log_*.jsonl and average public tokens.
- Measures interrupt usefulness (accuracy delta before/after interrupt) and AUC-like
  average accuracy across buckets.
- Plots:
  * token_accuracy_overall.png      : overall token-wise accuracy per run.
  * token_accuracy_by_agent_<run>.png: per-agent token-wise accuracy for each run.
  * action_distribution_<run>.png   : action selection distribution per agent.
- Writes summary.json with bucket accuracies and action counts.

Usage example
  python analyze_token_usage_by_agent.py --runs run_20260125_060659 run_20260125_060849 \
      --bucket 25 --out_dir analysis_outputs/token_usage_by_agent
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


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


def parse_prompt_tokens(prompt_log: Path) -> Counter:
    agg = Counter()
    for rec in load_jsonl(prompt_log):
        if not isinstance(rec, dict):
            continue
        stats = rec.get("token_stats") or {}
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            try:
                agg[k] += int(stats.get(k, 0))
            except Exception:
                continue
    return agg


# ------------------------ core parsing ------------------------ #
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


def process_problem(problem_dir: Path, bucket: int):
    log_path = problem_dir / "discussion_log.json"
    if not log_path.exists():
        return None

    data = load_json(log_path)
    gold = load_gold(problem_dir)

    # per-problem accumulators
    action_counts: Dict[str, Counter] = defaultdict(Counter)
    bucket_answers: Dict[str, Dict[int, str]] = defaultdict(dict)  # agent -> bucket -> ans
    interrupt_deltas: List[Tuple[float, int]] = []  # (delta accuracy, tokens_after)
    max_tokens = 0

    prev_interrupt_agents: set[str] = set()
    last_answer: Dict[str, Optional[str]] = {}
    cum_tokens = 0

    # seed with initial answers at token 0
    if data:
        init = data[0].get("initial_answers") or {}
        for agent, meta in init.items():
            if isinstance(meta, dict):
                ans = meta.get("answer")
                if isinstance(ans, str):
                    ans_norm = ans.strip().upper()
                    if ans_norm in {"A", "B", "C", "D"}:
                        last_answer[agent] = ans_norm
                        bucket_answers[agent][0] = ans_norm

    for ev in data:
        raw_event_type = ev.get("event_type")
        speaker = ev.get("speaker")

        # derive interrupt when event_type is missing
        derived_event_type = raw_event_type
        if (
            raw_event_type == "utterance"
            and speaker
            and speaker in prev_interrupt_agents
        ):
            derived_event_type = "interrupt"

        tokens_before = cum_tokens

        # update token counter
        if "public_tokens_used" in ev:
            cum_tokens = int(ev.get("public_tokens_used", cum_tokens))
        else:
            text = ev.get("content", "") or ""
            cum_tokens += max(1, len(text.split()))
        max_tokens = max(max_tokens, cum_tokens)

        actions = ev.get("agent_actions") or []

        # prepare interrupt candidates for the next turn
        next_interrupt_candidates: set[str] = set()

        # start with previous answers; update if current turn provides one
        answers_current = dict(last_answer)

        for a in actions:
            if not isinstance(a, dict):
                continue
            agent = a.get("agent_name") or "unknown"
            plan = a.get("action_plan") or {}

            act = plan.get("action")
            if isinstance(act, str):
                action_counts[agent][act] += 1
                if act == "interrupt":
                    next_interrupt_candidates.add(agent)

            ans = plan.get("answer")
            if isinstance(ans, str):
                ans_norm = ans.strip().upper()
                if ans_norm in {"A", "B", "C", "D"}:
                    answers_current[agent] = ans_norm

        # interrupt delta calculation (before vs after answers)
        if derived_event_type == "interrupt" and gold:
            agents_considered = [a for a in set(list(last_answer.keys()) + list(answers_current.keys())) if answers_current.get(a) or last_answer.get(a)]
            if agents_considered:
                correct_before = sum(1 for a in agents_considered if last_answer.get(a) == gold) / len(agents_considered)
                correct_after = sum(1 for a in agents_considered if answers_current.get(a) == gold) / len(agents_considered)
                interrupt_deltas.append((correct_after - correct_before, cum_tokens))

        # if speaker missing answer, carry forward automatically via answers_current
        last_answer = answers_current

        # fill buckets covered in this turn with the current answers
        start_bucket = (tokens_before // bucket) * bucket
        end_bucket = (cum_tokens // bucket) * bucket
        for b in range(start_bucket, end_bucket + 1, bucket):
            for agent, ans in answers_current.items():
                if ans:
                    bucket_answers[agent][b] = ans

        prev_interrupt_agents = next_interrupt_candidates

    return {
        "gold": gold,
        "bucket_answers": bucket_answers,
        "max_tokens": max_tokens,
        "action_counts": action_counts,
        "interrupt_deltas": interrupt_deltas,
    }


# ------------------------ aggregation per run ------------------------ #
def aggregate_run(run_dir: Path, num_problems: int, bucket: int):
    problems = sorted(p for p in run_dir.glob("problem_*") if p.is_dir())
    if num_problems > 0:
        problems = problems[:num_problems]

    overall_correct = Counter()
    overall_total = Counter()
    per_agent_correct: Dict[str, Counter] = defaultdict(Counter)
    per_agent_total: Dict[str, Counter] = defaultdict(Counter)
    action_counts: Dict[str, Counter] = defaultdict(Counter)
    interrupt_deltas: List[float] = []
    public_tokens_sum = 0
    public_tokens_count = 0
    model_tokens_sum = Counter()

    max_bucket = 0
    processed = 0

    for pdir in problems:
        res = process_problem(pdir, bucket)
        if not res:
            continue

        gold = res["gold"]

        for agent, buckets in res["bucket_answers"].items():
            for b, ans in buckets.items():
                max_bucket = max(max_bucket, b)
                per_agent_total[agent][b] += 1
                if gold and ans == gold:
                    per_agent_correct[agent][b] += 1
                if gold:
                    overall_total[b] += 1
                    if ans == gold:
                        overall_correct[b] += 1

        for agent, cnt in res["action_counts"].items():
            action_counts[agent].update(cnt)

        for delta, _tok in res["interrupt_deltas"]:
            interrupt_deltas.append(delta)

        # public token usage
        public_tokens_sum += res["max_tokens"]
        public_tokens_count += 1

        # model token usage
        prompt_log = find_prompt_log(pdir)
        if prompt_log:
            tok = parse_prompt_tokens(prompt_log)
            model_tokens_sum.update(tok)

        processed += 1

    xs = list(range(0, max_bucket + bucket, bucket)) if max_bucket > 0 else [0]

    def to_curve(correct: Counter, total: Counter):
        return [
            (correct[x] / total[x]) if total[x] else None
            for x in xs
        ]

    overall_curve = to_curve(overall_correct, overall_total)
    per_agent_curves = {
        agent: to_curve(per_agent_correct[agent], per_agent_total[agent])
        for agent in per_agent_total.keys()
    }

    def curve_auc(curve: List[Optional[float]]) -> Optional[float]:
        vals = [v for v in curve if v is not None]
        return sum(vals) / len(vals) if vals else None

    overall_auc = curve_auc(overall_curve)
    per_agent_auc = {agent: curve_auc(curve) for agent, curve in per_agent_curves.items()}

    interrupt_summary = {
        "count": len(interrupt_deltas),
        "delta_mean": (sum(interrupt_deltas) / len(interrupt_deltas)) if interrupt_deltas else None,
        "delta_positive_rate": (sum(1 for d in interrupt_deltas if d > 0) / len(interrupt_deltas)) if interrupt_deltas else None,
    }

    return {
        "xs": xs,
        "overall_curve": overall_curve,
        "per_agent_curves": per_agent_curves,
        "action_counts": action_counts,
        "problems": processed,
        "interrupt": interrupt_summary,
        "overall_auc": overall_auc,
        "per_agent_auc": per_agent_auc,
        "public_tokens_avg": public_tokens_sum / public_tokens_count if public_tokens_count else 0,
        "model_tokens": dict(model_tokens_sum),
    }


# ------------------------ plotting ------------------------ #
PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _get_font(size: int = 14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _line_chart(series: Dict[str, List[Optional[float]]], xs: List[int], title: str, x_label: str, y_label: str, out_path: Path):
    width, height = 1200, 720
    margin_left, margin_right, margin_top, margin_bottom = 120, 80, 80, 100
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    x_max = max(xs) if xs else 1
    font = _get_font(16)
    small_font = _get_font(13)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    def x_to_px(x: int) -> float:
        if x_max == 0:
            return margin_left
        return margin_left + (x / x_max) * plot_w

    def y_to_px(y: float) -> float:
        y_clamped = min(max(y, 0.0), 1.0)
        return margin_top + (1 - y_clamped) * plot_h

    # grid and ticks
    for i in range(6):
        y_val = i * 0.2
        y_px = y_to_px(y_val)
        draw.line((margin_left, y_px, width - margin_right, y_px), fill="#e5e5e5", width=1)
        draw.text((margin_left - 50, y_px - 8), f"{y_val:.1f}", fill="black", font=small_font)

    tick_count = 6
    step = max(1, int(round(x_max / (tick_count - 1)))) if x_max else 1
    for x_tick in range(0, x_max + step, step):
        x_px = x_to_px(x_tick)
        draw.line((x_px, height - margin_bottom, x_px, margin_top), fill="#f0f0f0", width=1)
        draw.text((x_px - 10, height - margin_bottom + 8), str(x_tick), fill="black", font=small_font)

    # axes
    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill="black", width=2)
    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill="black", width=2)

    # lines
    for idx, (label, ys_raw) in enumerate(series.items()):
        pts = []
        for x_val, y_val in zip(xs, ys_raw):
            if y_val is None:
                continue
            pts.append((x_to_px(x_val), y_to_px(y_val)))
        if len(pts) >= 2:
            draw.line(pts, fill=PALETTE[idx % len(PALETTE)], width=3)
        for px, py in pts:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=PALETTE[idx % len(PALETTE)])

    # labels and title
    draw.text((margin_left, margin_top - 50), title, fill="black", font=font)
    draw.text((margin_left + plot_w / 2 - 40, height - margin_bottom + 40), x_label, fill="black", font=font)
    draw.text((20, margin_top + plot_h / 2 - 20), y_label, fill="black", font=font)

    # legend
    legend_x = width - margin_right - 200
    legend_y = margin_top
    for idx, label in enumerate(series.keys()):
        y_pos = legend_y + idx * 22
        draw.rectangle((legend_x, y_pos, legend_x + 18, y_pos + 12), fill=PALETTE[idx % len(PALETTE)], outline="black")
        draw.text((legend_x + 24, y_pos - 2), label, fill="black", font=small_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[info] saved {out_path}")


def _stacked_bar(action_counts: Dict[str, Counter], title: str, out_path: Path):
    if not action_counts:
        return
    width, height = 1200, 720
    margin_left, margin_right, margin_top, margin_bottom = 120, 80, 80, 120
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    agents = sorted(action_counts.keys())
    actions = sorted({a for cnt in action_counts.values() for a in cnt})
    max_total = max(sum(action_counts[ag].values()) for ag in agents) or 1

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _get_font(16)
    small_font = _get_font(13)

    bar_spacing = plot_w / max(len(agents), 1)
    bar_width = bar_spacing * 0.6

    # grid and ticks
    ticks = 5
    for i in range(ticks + 1):
        val = max_total * i / ticks
        y_px = margin_top + plot_h * (1 - val / max_total)
        draw.line((margin_left, y_px, width - margin_right, y_px), fill="#e5e5e5", width=1)
        draw.text((margin_left - 50, y_px - 8), f"{int(val)}", fill="black", font=small_font)

    # axes
    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill="black", width=2)
    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill="black", width=2)

    # bars
    for i, agent in enumerate(agents):
        x_center = margin_left + bar_spacing * i + bar_spacing / 2
        bottom = height - margin_bottom
        for j, action in enumerate(actions):
            count = action_counts[agent].get(action, 0)
            bar_h = (count / max_total) * plot_h
            top = bottom - bar_h
            draw.rectangle(
                (x_center - bar_width / 2, top, x_center + bar_width / 2, bottom),
                fill=PALETTE[j % len(PALETTE)],
                outline="black",
            )
            bottom = top
        draw.text(
            (x_center - bar_width / 2, height - margin_bottom + 10),
            agent,
            fill="black",
            font=small_font,
        )

    # legend
    legend_x = width - margin_right - 200
    legend_y = margin_top
    for j, action in enumerate(actions):
        y_pos = legend_y + j * 22
        draw.rectangle((legend_x, y_pos, legend_x + 18, y_pos + 12), fill=PALETTE[j % len(PALETTE)], outline="black")
        draw.text((legend_x + 24, y_pos - 2), action, fill="black", font=small_font)

    draw.text((margin_left, margin_top - 50), title, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[info] saved {out_path}")


def plot_overall(run_results: Dict[str, dict], bucket: int, out_path: Path):
    if not run_results:
        return
    all_xs = sorted({x for res in run_results.values() for x in res["xs"]})
    series = {}
    for rid, res in run_results.items():
        value_map = {x: y for x, y in zip(res["xs"], res["overall_curve"])}
        series[rid] = [value_map.get(x) for x in all_xs]
    _line_chart(series, all_xs, f"Token-wise accuracy (bucket={bucket})", "Public tokens", "Accuracy", out_path)


def plot_per_agent(run_id: str, xs: List[int], curves: Dict[str, List[Optional[float]]], out_path: Path):
    _line_chart(curves, xs, f"Token-wise accuracy by agent ({run_id})", "Public tokens", "Accuracy", out_path)


def plot_action_distribution(run_id: str, action_counts: Dict[str, Counter], out_path: Path):
    _stacked_bar(action_counts, f"Action selection distribution ({run_id})", out_path)


# ------------------------ main ------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run IDs under ./logs")
    ap.add_argument("--num_problems", type=int, default=-1, help="problems to analyze (-1 for all)")
    ap.add_argument("--bucket", type=int, default=50, help="public token bucket size")
    ap.add_argument("--out_dir", type=Path, default=Path("analysis_outputs/token_usage_by_agent"))
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

    # overall plot across runs
    plot_overall(run_results, args.bucket, args.out_dir / "token_accuracy_overall.png")

    # per-run plots
    for rid, res in run_results.items():
        plot_per_agent(
            rid,
            res["xs"],
            res["per_agent_curves"],
            args.out_dir / f"token_accuracy_by_agent_{rid}.png",
        )
        plot_action_distribution(
            rid,
            res["action_counts"],
            args.out_dir / f"action_distribution_{rid}.png",
        )

    # summary json
    summary = {}
    for rid, res in run_results.items():
        summary[rid] = {
            "problems": res["problems"],
            "bucket": args.bucket,
            "overall_auc": res.get("overall_auc"),
            "per_agent_auc": res.get("per_agent_auc"),
            "public_tokens_avg": res.get("public_tokens_avg"),
            "model_tokens": res.get("model_tokens"),
            "interrupt": res.get("interrupt"),
            "overall_accuracy": {str(x): res["overall_curve"][i] for i, x in enumerate(res["xs"])},
            "per_agent_accuracy": {
                agent: {str(x): curve[i] for i, x in enumerate(res["xs"])}
                for agent, curve in res["per_agent_curves"].items()
            },
            "action_distribution": {agent: dict(cnt) for agent, cnt in res["action_counts"].items()},
        }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[info] saved summary -> {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
