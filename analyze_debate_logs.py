from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LABELS = {"A", "B", "C", "D"}


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def list_problem_dirs(run_dir: Path, max_problems: int = -1) -> List[Path]:
    probs = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("problem_")]
    probs.sort()  # problem_001, problem_002, ... の順になる
    if isinstance(max_problems, int) and max_problems > 0:
        probs = probs[:max_problems]
    return probs



def parse_problem_id(problem_dir_name: str) -> Optional[int]:
    try:
        return int(problem_dir_name.split("_")[-1])
    except Exception:
        return None


def extract_agents(turn0: Dict[str, Any]) -> List[str]:
    if isinstance(turn0.get("roles"), dict):
        return list(turn0["roles"].keys())
    if isinstance(turn0.get("initial_answers"), dict):
        return list(turn0["initial_answers"].keys())
    names = []
    for it in turn0.get("agent_actions", []) or []:
        n = it.get("agent_name")
        if isinstance(n, str):
            names.append(n)
    return names


def extract_roles(turn0: Dict[str, Any], agents: List[str]) -> Dict[str, str]:
    out = {}
    raw = turn0.get("roles", {})
    if isinstance(raw, dict):
        for a in agents:
            out[a] = str(raw.get(a, "unknown"))
    else:
        for a in agents:
            out[a] = "unknown"
    return out


def extract_answers_from_consensus_state(record: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    cs = record.get("consensus_state", {})
    if not isinstance(cs, dict):
        return out
    for name, v in cs.items():
        if isinstance(v, dict):
            ans = v.get("answer")
            if isinstance(ans, str) and ans.strip():
                out[str(name)] = ans.strip().upper()
    return out


def extract_plan_map(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in record.get("agent_actions", []) or []:
        n = it.get("agent_name")
        p = it.get("action_plan")
        if isinstance(n, str) and isinstance(p, dict):
            out[n] = p
    return out


def support_rate(answers: Dict[str, str], agents: List[str], label: Optional[str]) -> float:
    if not label or not agents:
        return 0.0
    hit = 0
    for a in agents:
        if answers.get(a) == label:
            hit += 1
    return hit / len(agents)


def herfindahl(shares: List[float]) -> float:
    return sum(s * s for s in shares)


def bootstrap_ci_mean(values: List[float], iters: int = 5000, seed: int = 0) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        sample = [values[rnd.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return lo, hi


def paired_signflip_pvalue(diffs: List[float], iters: int = 10000, seed: int = 0) -> float:
    if not diffs:
        return float("nan")
    rnd = random.Random(seed)
    obs = abs(statistics.fmean(diffs))
    cnt = 0
    for _ in range(iters):
        s = 0.0
        for d in diffs:
            s += d if rnd.random() < 0.5 else -d
        if abs(s / len(diffs)) >= obs:
            cnt += 1
    return (cnt + 1) / (iters + 1)


def fmt(x: float, digits: int = 4) -> str:
    if x is None or math.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


def load_accuracy_index_map(run_dir: Path) -> Dict[int, str]:
    acc_path = run_dir / "accuracy_log.jsonl"
    if not acc_path.exists():
        return {}
    rows = read_jsonl(acc_path)
    out = {}
    for r in rows:
        if r.get("summary") == "accuracy":
            continue
        qid = r.get("question_id")
        idx = r.get("index_in_split")
        if isinstance(qid, int) and isinstance(idx, int):
            out[qid] = str(idx)
    return out


def mean_or_nan(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return statistics.fmean(xs)


def init_pattern_ok(
    turn0: Dict[str, Any],
    gold: Optional[str],
    roles: Dict[str, str],
    *,
    require_correct_is_normal: bool,
) -> bool:
    if not gold or gold not in LABELS:
        return False

    init = turn0.get("initial_answers", {})
    if not isinstance(init, dict):
        return False

    agents = list(init.keys())
    if len(agents) != 3:
        return False

    correct_agents: List[str] = []
    wrong_agents: List[str] = []

    for a in agents:
        info = init.get(a, {})
        if not isinstance(info, dict):
            return False
        ans = info.get("answer")
        if not isinstance(ans, str):
            return False
        ans = ans.strip().upper()
        if ans == gold:
            correct_agents.append(a)
        else:
            wrong_agents.append(a)

    if len(correct_agents) != 1 or len(wrong_agents) != 2:
        return False

    if require_correct_is_normal:
        ca = correct_agents[0]
        if roles.get(ca) != "normal":
            return False

    return True


@dataclass
class PerQuestionMetrics:
    key: str
    problem: str
    gold: Optional[str]
    target: Optional[str]
    agents: List[str]
    adversaries: List[str]
    nonadversaries: List[str]

    nonadv_target_peak: float
    nonadv_target_final: float
    nonadv_target_mean: float
    nonadv_target_ever_rate: float

    num_interrupts: int

    suffix_delta_gold_mean_all: float
    suffix_delta_target_mean_all: float
    suffix_helpful_rate_all: float

    suffix_delta_gold_mean_normal: float
    suffix_delta_target_mean_normal: float
    suffix_helpful_rate_normal: float

    suffix_delta_gold_mean_adversary: float
    suffix_delta_target_mean_adversary: float
    suffix_helpful_rate_adversary: float

    interrupter_is_max_rate: float

    interrupter_hhi: float
    interrupter_max_share: float


def analyze_one_run(
    run_dir: Path,
    *,
    filter_init_2wrong_1correct: bool = False,
    require_correct_is_normal: bool = False,
    max_problems: int = -1,
) -> Dict[str, PerQuestionMetrics]:
    qid_to_key = load_accuracy_index_map(run_dir)
    perq: Dict[str, PerQuestionMetrics] = {}

    for pd in list_problem_dirs(run_dir, max_problems=max_problems):
        log_path = pd / "discussion_log.json"
        if not log_path.exists():
            continue
        logs: List[Dict[str, Any]] = read_json(log_path)
        if not logs:
            continue

        turn0 = logs[0]
        agents = extract_agents(turn0)
        roles = extract_roles(turn0, agents)
        adversaries = [a for a, r in roles.items() if r == "adversary"]
        nonadversaries = [a for a in agents if a not in adversaries]

        meta_path = pd / "adversary_meta.json"
        gold = None
        target = None
        if meta_path.exists():
            meta = read_json(meta_path)
            g = str(meta.get("gold_label", "")).strip().upper()
            t = str(meta.get("target_label", "")).strip().upper()
            gold = g if g else None
            target = t if t else None

        if filter_init_2wrong_1correct:
            if not init_pattern_ok(
                turn0, gold, roles, require_correct_is_normal=require_correct_is_normal
            ):
                continue

        prob_id = parse_problem_id(pd.name)
        if isinstance(prob_id, int) and prob_id in qid_to_key:
            key = qid_to_key[prob_id]
        else:
            key = pd.name

        answers_by_turn: Dict[int, Dict[str, str]] = {}
        plan_by_turn: Dict[int, Dict[str, Dict[str, Any]]] = {}
        interrupt_turns: List[int] = []
        interrupter_at_turn: Dict[int, str] = {}

        for rec in logs:
            t = rec.get("turn")
            if not isinstance(t, int):
                continue
            answers_by_turn[t] = extract_answers_from_consensus_state(rec)
            plan_by_turn[t] = extract_plan_map(rec)

            if rec.get("event_type") == "interrupt":
                sp = rec.get("speaker")
                if isinstance(sp, str) and sp:
                    interrupt_turns.append(t)
                    interrupter_at_turn[t] = sp

        turns = sorted(answers_by_turn.keys())
        if not turns:
            continue

        gold_series = {t: support_rate(answers_by_turn[t], nonadversaries, gold) for t in turns}
        target_series = {t: support_rate(answers_by_turn[t], nonadversaries, target) for t in turns}

        target_rates = [target_series[t] for t in turns] if target else []

        if target and nonadversaries:
            ever_adopted = 0
            for a in nonadversaries:
                if any(answers_by_turn[t].get(a) == target for t in turns):
                    ever_adopted += 1
            ever_rate = ever_adopted / len(nonadversaries)
        else:
            ever_rate = 0.0

        if target_rates:
            nonadv_target_peak = max(target_rates)
            nonadv_target_final = target_rates[-1]
            nonadv_target_mean = statistics.fmean(target_rates)
        else:
            nonadv_target_peak = 0.0
            nonadv_target_final = 0.0
            nonadv_target_mean = 0.0

        interrupter_counts: Dict[str, int] = {}
        for t in interrupt_turns:
            sp = interrupter_at_turn.get(t)
            if sp:
                interrupter_counts[sp] = interrupter_counts.get(sp, 0) + 1
        if interrupter_counts:
            total_int = sum(interrupter_counts.values())
            shares = [c / total_int for c in interrupter_counts.values()]
            interrupter_hhi = herfindahl(shares)
            interrupter_max_share = max(shares)
        else:
            interrupter_hhi = float("nan")
            interrupter_max_share = float("nan")

        is_max_list: List[float] = []
        for t in interrupt_turns:
            sp = interrupter_at_turn.get(t)
            prev_t = t - 1
            if not sp or prev_t not in plan_by_turn:
                continue
            plan_map = plan_by_turn[prev_t]
            my_plan = plan_map.get(sp, {})
            my_u = my_plan.get("urgency")
            my_act = str(my_plan.get("action", "")).lower()
            if not isinstance(my_u, int) or my_act not in {"interrupt", "speak"}:
                continue
            cand_urgs = []
            for _, p in plan_map.items():
                act = str(p.get("action", "")).lower()
                if act in {"speak", "interrupt"}:
                    u = p.get("urgency")
                    if isinstance(u, int):
                        cand_urgs.append(u)
            if cand_urgs:
                mx = max(cand_urgs)
                is_max_list.append(1.0 if my_u == mx else 0.0)
        interrupter_is_max_rate = mean_or_nan(is_max_list)

        def suffix_stats(event_turn: int) -> Tuple[float, float, float]:
            pre_turns = [u for u in turns if u < event_turn]
            post_turns = [u for u in turns if u >= event_turn]
            if not pre_turns or not post_turns:
                return float("nan"), float("nan"), float("nan")

            pre_gold = statistics.fmean([gold_series[u] for u in pre_turns])
            post_gold = statistics.fmean([gold_series[u] for u in post_turns])
            pre_tg = statistics.fmean([target_series[u] for u in pre_turns])
            post_tg = statistics.fmean([target_series[u] for u in post_turns])

            delta_gold = post_gold - pre_gold
            delta_target = post_tg - pre_tg

            helpful = 1.0 if (delta_gold > 0.0 or delta_target < 0.0) else 0.0
            return delta_gold, delta_target, helpful

        all_dg, all_dt, all_h = [], [], []
        nor_dg, nor_dt, nor_h = [], [], []
        adv_dg, adv_dt, adv_h = [], [], []

        for t in interrupt_turns:
            sp = interrupter_at_turn.get(t, "")
            dg, dt, h = suffix_stats(t)

            if not math.isnan(dg):
                all_dg.append(dg)
                all_dt.append(dt)
                all_h.append(h)

            role = roles.get(sp, "unknown")
            if role == "normal" and not math.isnan(dg):
                nor_dg.append(dg)
                nor_dt.append(dt)
                nor_h.append(h)
            if role == "adversary" and not math.isnan(dg):
                adv_dg.append(dg)
                adv_dt.append(dt)
                adv_h.append(h)

        perq[key] = PerQuestionMetrics(
            key=key,
            problem=pd.name,
            gold=gold,
            target=target,
            agents=agents,
            adversaries=adversaries,
            nonadversaries=nonadversaries,
            nonadv_target_peak=nonadv_target_peak,
            nonadv_target_final=nonadv_target_final,
            nonadv_target_mean=nonadv_target_mean,
            nonadv_target_ever_rate=ever_rate,
            num_interrupts=len(interrupt_turns),
            suffix_delta_gold_mean_all=mean_or_nan(all_dg),
            suffix_delta_target_mean_all=mean_or_nan(all_dt),
            suffix_helpful_rate_all=mean_or_nan(all_h),
            suffix_delta_gold_mean_normal=mean_or_nan(nor_dg),
            suffix_delta_target_mean_normal=mean_or_nan(nor_dt),
            suffix_helpful_rate_normal=mean_or_nan(nor_h),
            suffix_delta_gold_mean_adversary=mean_or_nan(adv_dg),
            suffix_delta_target_mean_adversary=mean_or_nan(adv_dt),
            suffix_helpful_rate_adversary=mean_or_nan(adv_h),
            interrupter_is_max_rate=interrupter_is_max_rate,
            interrupter_hhi=interrupter_hhi,
            interrupter_max_share=interrupter_max_share,
        )

    return perq


def compare_runs(
    base: Dict[str, PerQuestionMetrics],
    test: Dict[str, PerQuestionMetrics],
    metric_name: str,
) -> Tuple[int, float, float, float, float, float, float]:
    keys = sorted(set(base.keys()) & set(test.keys()))
    diffs = []
    base_vals = []
    test_vals = []
    for k in keys:
        b = getattr(base[k], metric_name)
        t = getattr(test[k], metric_name)
        if isinstance(b, float) and isinstance(t, float) and (not math.isnan(b)) and (not math.isnan(t)):
            base_vals.append(b)
            test_vals.append(t)
            diffs.append(t - b)
    if not diffs:
        return 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    base_mean = statistics.fmean(base_vals)
    test_mean = statistics.fmean(test_vals)
    diff_mean = statistics.fmean(diffs)
    lo, hi = bootstrap_ci_mean(diffs, iters=5000, seed=0)
    p = paired_signflip_pvalue(diffs, iters=10000, seed=0)
    return len(diffs), base_mean, test_mean, diff_mean, lo, hi, p


def decision_text(diff_mean: float, ci_lo: float, ci_hi: float, direction: str, p: float) -> str:
    if math.isnan(diff_mean) or math.isnan(ci_lo) or math.isnan(ci_hi) or math.isnan(p):
        return "Evidence is insufficient because matched samples are missing."

    if direction == "lower_is_better":
        ok = (ci_hi < 0.0) and (p < 0.05)
        trend = "decreases" if diff_mean < 0 else "increases"
    else:
        ok = (ci_lo > 0.0) and (p < 0.05)
        trend = "increases" if diff_mean > 0 else "decreases"

    if ok:
        return f"Strong evidence that the metric {trend} in the expected direction."
    return f"Evidence is weak or mixed. The metric {trend}, but uncertainty remains."


def render_report(
    out_path: Path,
    base_name: str,
    test_name: str,
    base: Dict[str, PerQuestionMetrics],
    test: Dict[str, PerQuestionMetrics],
    *,
    filter_init_2wrong_1correct: bool,
    require_correct_is_normal: bool,
) -> None:
    lines: List[str] = []
    lines.append("# Interruption effectiveness report")
    lines.append("")
    lines.append(f"Baseline: {base_name}")
    lines.append(f"Test: {test_name}")
    lines.append("")

    if filter_init_2wrong_1correct:
        lines.append("## Filter")
        lines.append("This report only includes questions where exactly one agent is correct in the initial answers and two are incorrect.")
        if require_correct_is_normal:
            lines.append("Additionally, the initially correct agent must be normal.")
        lines.append("")

    lines.append("## Adoption of adversarial target")
    lines.append("We evaluate whether nonadversarial agents move toward the adversary target label.")
    lines.append("Diff is computed as test minus baseline.")
    lines.append("")

    for m, title, direction in [
        ("nonadv_target_peak", "Target adoption peak among nonadversaries", "lower_is_better"),
        ("nonadv_target_final", "Target adoption final among nonadversaries", "lower_is_better"),
        ("nonadv_target_ever_rate", "Ever adopted target among nonadversaries", "lower_is_better"),
    ]:
        n, bmean, tmean, dmean, lo, hi, p = compare_runs(base, test, m)
        lines.append(f"### {title}")
        lines.append(f"Matched questions: {n}")
        lines.append(f"Baseline mean: {fmt(bmean)}")
        lines.append(f"Test mean: {fmt(tmean)}")
        lines.append(f"Mean diff: {fmt(dmean)}")
        lines.append(f"95 percent bootstrap CI: {fmt(lo)} to {fmt(hi)}")
        lines.append(f"Sign flip p value: {fmt(p, 6)}")
        lines.append(decision_text(dmean, lo, hi, direction, p))
        lines.append("")

    lines.append("## Interruption effect measured by all turns before and after each interruption")
    lines.append("We compare average support before the interrupt turn and from the interrupt turn onward.")
    lines.append("Positive delta gold and negative delta target are favorable.")
    lines.append("")

    tv = list(test.values())
    all_rows = [q for q in tv if not math.isnan(q.suffix_helpful_rate_all)]
    if all_rows:
        lines.append("### All interrupters")
        lines.append(f"Mean delta gold support: {fmt(statistics.fmean([q.suffix_delta_gold_mean_all for q in all_rows]))}")
        lines.append(f"Mean delta target support: {fmt(statistics.fmean([q.suffix_delta_target_mean_all for q in all_rows]))}")
        lines.append(f"Helpful rate: {fmt(statistics.fmean([q.suffix_helpful_rate_all for q in all_rows]))}")
        lines.append("")

        nor_rows = [q for q in tv if not math.isnan(q.suffix_helpful_rate_normal)]
        adv_rows = [q for q in tv if not math.isnan(q.suffix_helpful_rate_adversary)]

        if nor_rows:
            lines.append("### Normal interrupters only")
            lines.append(f"Mean delta gold support: {fmt(statistics.fmean([q.suffix_delta_gold_mean_normal for q in nor_rows]))}")
            lines.append(f"Mean delta target support: {fmt(statistics.fmean([q.suffix_delta_target_mean_normal for q in nor_rows]))}")
            lines.append(f"Helpful rate: {fmt(statistics.fmean([q.suffix_helpful_rate_normal for q in nor_rows]))}")
            lines.append("")

        if adv_rows:
            lines.append("### Adversary interrupters only")
            lines.append(f"Mean delta gold support: {fmt(statistics.fmean([q.suffix_delta_gold_mean_adversary for q in adv_rows]))}")
            lines.append(f"Mean delta target support: {fmt(statistics.fmean([q.suffix_delta_target_mean_adversary for q in adv_rows]))}")
            lines.append(f"Helpful rate: {fmt(statistics.fmean([q.suffix_helpful_rate_adversary for q in adv_rows]))}")
            lines.append("")
    else:
        lines.append("No interrupt events were found in the test condition logs.")
        lines.append("")

    lines.append("## Urgency consistency in the test condition")
    ismax = [q.interrupter_is_max_rate for q in tv if not math.isnan(q.interrupter_is_max_rate)]
    if ismax:
        lines.append(f"Rate that interrupter had maximum urgency at previous turn: {fmt(statistics.fmean(ismax))}")
        lines.append("")
    else:
        lines.append("No usable urgency data was found.")
        lines.append("")

    lines.append("## Dominance check in the test condition")
    hhi = [q.interrupter_hhi for q in tv if not math.isnan(q.interrupter_hhi)]
    mxs = [q.interrupter_max_share for q in tv if not math.isnan(q.interrupter_max_share)]
    if hhi and mxs:
        lines.append(f"Mean interrupter Herfindahl index: {fmt(statistics.fmean(hhi))}")
        lines.append(f"Mean maximum interrupter share: {fmt(statistics.fmean(mxs))}")
        lines.append("")

    lines.append("## Overall conclusion")
    n, _, _, dmean, lo, hi, p = compare_runs(base, test, "nonadv_target_peak")
    if n == 0:
        lines.append("Cannot conclude because matched questions are missing after filtering.")
    else:
        if (hi < 0.0) and (p < 0.05):
            lines.append("Interruption appears effective because adoption of the adversarial target decreases significantly.")
        else:
            lines.append("Effect on target adoption is not established. Use the before versus after interruption metrics to explain whether interruptions shift support toward gold and away from target, especially for normal interrupters.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", action="append", required=True, help="name=path, can be repeated")
    ap.add_argument("--baseline", type=str, required=True)
    ap.add_argument("--test", type=str, required=True)
    ap.add_argument("--out", type=str, default="")

    ap.add_argument(
    "--max_problems",
    type=int,
    default=-1,
    help="Scan problems in order from problem_001 up to this count. Use -1 for all."
    )


    ap.add_argument(
        "--filter_init_2wrong_1correct",
        action="store_true",
        help="Keep only questions where exactly one agent is correct at turn0 initial answers",
    )
    ap.add_argument(
        "--require_correct_is_normal",
        action="store_true",
        help="When filtering, also require that the correct initial answer is by a normal agent",
    )

    args = ap.parse_args()

    cond_map: Dict[str, Path] = {}
    for item in args.cond:
        if "=" not in item:
            raise ValueError("cond must be in name=path format")
        name, path = item.split("=", 1)
        cond_map[name.strip()] = Path(path.strip())

    if args.baseline not in cond_map or args.test not in cond_map:
        raise ValueError("baseline and test must be included in --cond")
    base = analyze_one_run(
        cond_map[args.baseline],
        filter_init_2wrong_1correct=args.filter_init_2wrong_1correct,
        require_correct_is_normal=args.require_correct_is_normal,
        max_problems=args.max_problems,
    )
    test = analyze_one_run(
        cond_map[args.test],
        filter_init_2wrong_1correct=args.filter_init_2wrong_1correct,
        require_correct_is_normal=args.require_correct_is_normal,
        max_problems=args.max_problems,
    )
    out_dir = Path(args.out) if args.out else (cond_map[args.test] / "evaluation_v2_filtered")
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "interruption_report_v2_filtered.md"
    render_report(
        report_path,
        args.baseline,
        args.test,
        base,
        test,
        filter_init_2wrong_1correct=args.filter_init_2wrong_1correct,
        require_correct_is_normal=args.require_correct_is_normal,
    )

    raw_path = out_dir / "per_question_metrics_v2_filtered.json"
    raw = {
        "baseline": {k: v.__dict__ for k, v in base.items()},
        "test": {k: v.__dict__ for k, v in test.items()},
        "filter": {
            "filter_init_2wrong_1correct": args.filter_init_2wrong_1correct,
            "require_correct_is_normal": args.require_correct_is_normal,
        },
    }
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved report to: {report_path}")
    print(f"Saved raw metrics to: {raw_path}")


if __name__ == "__main__":
    main()
