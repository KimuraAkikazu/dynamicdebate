from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


LABELS = {"A", "B", "C", "D"}

STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","when","while","to","of","in","on","for","with","as","by",
    "is","are","was","were","be","been","being","it","this","that","these","those","i","we","you","they","he","she",
    "my","our","your","their","his","her","me","us","them","not","no","do","does","did","can","could","should","would",
    "will","may","might","must","also","so","because","since","therefore","thus","hence","very","more","most","less",
    "some","any","each","all","about","into","from","at","up","down","over","under","again",
}

EVIDENCE_CUES = {"because", "since", "therefore", "thus", "hence", "as a result", "due to"}


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    toks = re.findall(r"[a-z]+", text)
    return [t for t in toks if t not in STOPWORDS and len(t) >= 3]


def detect_has_evidence(text: str) -> bool:
    t = (text or "").lower()
    if any(cue in t for cue in EVIDENCE_CUES):
        return True
    if "理由" in (text or "") or "根拠" in (text or ""):
        return True
    return False


def hhi_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    shares = [c / total for c in counts.values()]
    return sum(s * s for s in shares)


def max_share_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    return max(c / total for c in counts.values())


def parse_problem_id(problem_dir: Path) -> Optional[int]:
    m = re.match(r"problem_(\d+)$", problem_dir.name)
    if not m:
        return None
    return int(m.group(1))


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


def extract_plan_map(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in record.get("agent_actions", []) or []:
        n = it.get("agent_name")
        p = it.get("action_plan")
        if isinstance(n, str) and isinstance(p, dict):
            out[n] = p
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


def support_rate(answers: Dict[str, str], agents: Sequence[str], label: Optional[str]) -> float:
    if not label or label not in LABELS:
        return float("nan")
    if not agents:
        return float("nan")
    hit = 0
    for a in agents:
        if answers.get(a) == label:
            hit += 1
    return hit / len(agents)


def suffix_delta(turns: List[int], series: Dict[int, float], event_turn: int) -> Optional[float]:
    pre = [t for t in turns if t < event_turn]
    post = [t for t in turns if t >= event_turn]
    if not pre or not post:
        return None
    pre_vals = [series[t] for t in pre if not math.isnan(series[t])]
    post_vals = [series[t] for t in post if not math.isnan(series[t])]
    if not pre_vals or not post_vals:
        return None
    return statistics.fmean(post_vals) - statistics.fmean(pre_vals)


@dataclass
class PerProblemSummary:
    problem_id: int
    turns: int
    agents: List[str]
    adversaries: List[str]
    normals: List[str]
    gold: Optional[str]
    target: Optional[str]
    initial_correct_count: Optional[int]
    planned_interrupts: int
    realized_interrupt_speaker: int
    logged_interrupt_events: int
    urgency_consistency_rate: Optional[float]
    helpful_rate_normals: Optional[float]
    mean_delta_gold_normals: Optional[float]
    mean_delta_target_normals: Optional[float]
    interrupt_before_evidence_rate: Optional[float]
    correct_final: Optional[bool]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)

    ap.add_argument("--start", type=int, default=1, help="inclusive problem id")
    ap.add_argument("--end", type=int, default=-1, help="inclusive problem id, -1 means last")
    ap.add_argument("--max_n", type=int, default=-1, help="limit number of problems after filtering")

    ap.add_argument("--only_1c2w", action="store_true", help="only include problems where exactly 1 initial answer is correct and 2 are incorrect")
    ap.add_argument("--out_dir", type=str, default="")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    # accuracy map if exists
    acc_path = run_dir / "accuracy_log.jsonl"
    correct_by_problem: Dict[int, bool] = {}
    if acc_path.exists():
        rows = read_jsonl(acc_path)
        for r in rows:
            if isinstance(r.get("question_id"), int) and isinstance(r.get("correct"), bool):
                correct_by_problem[int(r["question_id"])] = bool(r["correct"])

    # collect problems
    probs = []
    for d in run_dir.glob("problem_*"):
        pid = parse_problem_id(d)
        if pid is None:
            continue
        probs.append((pid, d))
    probs.sort(key=lambda x: x[0])

    if not probs:
        raise ValueError("No problem_* directories found")

    start = args.start
    end = args.end if args.end != -1 else probs[-1][0]

    selected = [(pid, d) for pid, d in probs if start <= pid <= end]
    if args.max_n != -1:
        selected = selected[: args.max_n]

    # global aggregates
    action_cnt = Counter()
    purpose_cnt = Counter()
    action_purpose_cnt = Counter()
    urgency_by_action = defaultdict(list)

    planned_interrupt_cnt_by_agent = Counter()
    realized_interrupt_cnt_by_agent = Counter()
    logged_interrupt_cnt_by_agent = Counter()

    thought_tokens_by_action = defaultdict(Counter)
    thought_tokens_interrupt_realized = Counter()
    thought_tokens_interrupt_not_realized = Counter()

    interrupt_pair_cnt = Counter()

    summaries: List[PerProblemSummary] = []

    for pid, pd in selected:
        log_path = pd / "discussion_log.json"
        if not log_path.exists():
            continue
        logs: List[Dict[str, Any]] = read_json(log_path)
        if not logs or not isinstance(logs, list):
            continue

        turn0 = logs[0]
        agents = extract_agents(turn0)
        roles = extract_roles(turn0, agents)
        adversaries = [a for a in agents if roles.get(a) == "adversary"]
        normals = [a for a in agents if a not in adversaries]

        # gold and target
        gold = None
        target = None
        meta_path = pd / "adversary_meta.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            g = str(meta.get("gold_label", "")).strip().upper()
            t = str(meta.get("target_label", "")).strip().upper()
            gold = g if g in LABELS else None
            target = t if t in LABELS else None

        # initial correct count
        init_correct = None
        if gold and isinstance(turn0.get("initial_answers"), dict):
            answers0 = []
            for a in agents:
                v = turn0["initial_answers"].get(a, {})
                if isinstance(v, dict):
                    ans = v.get("answer")
                    if isinstance(ans, str):
                        answers0.append(ans.strip().upper())
            if answers0:
                init_correct = sum(1 for x in answers0 if x == gold)

        if args.only_1c2w:
            # for 3 agents only, but keep generic check
            if init_correct is None:
                continue
            if not (init_correct == 1 and (len(agents) - init_correct) == 2):
                continue

        # turn index helpers
        by_turn = {r.get("turn"): r for r in logs if isinstance(r.get("turn"), int)}
        turns = sorted(by_turn.keys())
        max_turn = max(turns) if turns else 0

        # answers series
        answers_by_turn = {t: extract_answers_from_consensus_state(by_turn[t]) for t in turns}

        gold_series_normals = {t: support_rate(answers_by_turn.get(t, {}), normals, gold) for t in turns}
        target_series_normals = {t: support_rate(answers_by_turn.get(t, {}), normals, target) for t in turns}

        # plans per turn
        plan_by_turn = {t: extract_plan_map(by_turn[t]) for t in turns}

        # counts from agent plans
        for t in turns:
            rec = by_turn[t]
            for it in rec.get("agent_actions", []) or []:
                name = it.get("agent_name")
                plan = it.get("action_plan")
                if not isinstance(name, str) or not isinstance(plan, dict):
                    continue
                act = str(plan.get("action", "")).lower()
                pur = str(plan.get("purpose", "")).lower() if plan.get("purpose") is not None else ""
                urg = plan.get("urgency")
                th = str(plan.get("thought", "") or "")

                if act:
                    action_cnt[act] += 1
                if pur:
                    purpose_cnt[pur] += 1
                if act and pur:
                    action_purpose_cnt[(act, pur)] += 1
                if isinstance(urg, int) and act:
                    urgency_by_action[act].append(urg)

                toks = tokenize(th)
                if act and toks:
                    thought_tokens_by_action[act].update(toks)

        # planned interrupts and whether realized
        planned_interrupts = 0
        realized_interrupt_speaker = 0
        logged_interrupt_events = 0

        # urgency consistency for turns where we can infer chosen speaker from previous plans
        urgency_consistency_hits = 0
        urgency_consistency_total = 0

        helpful_flags = []
        delta_gold_list = []
        delta_target_list = []
        before_evidence_flags = []

        # logged interrupt events count and pairs
        for t in turns:
            et = str(by_turn[t].get("event_type", ""))
            sp = by_turn[t].get("speaker")
            if et == "interrupt" and isinstance(sp, str):
                logged_interrupt_events += 1
                logged_interrupt_cnt_by_agent[sp] += 1
                prev_sp = by_turn.get(t - 1, {}).get("speaker")
                if isinstance(prev_sp, str) and prev_sp != sp:
                    interrupt_pair_cnt[(sp, prev_sp)] += 1

        # realized interrupts are cases where previous plan action was interrupt and next speaker is that agent
        for t in turns:
            # planned at t, speaks at t+1
            nxt = t + 1
            if nxt not in by_turn:
                continue
            plans = plan_by_turn.get(t, {})
            nxt_speaker = by_turn[nxt].get("speaker")
            if not isinstance(nxt_speaker, str):
                continue

            # urgency consistency for chosen speaker on any speak or interrupt
            cand = []
            for name, p in plans.items():
                a = str(p.get("action", "")).lower()
                if a in {"speak", "interrupt"}:
                    u = p.get("urgency")
                    if isinstance(u, int):
                        cand.append((name, a, u))
            if cand:
                urgency_consistency_total += 1
                max_u = max(u for _, _, u in cand)
                chosen_u = None
                for n, _, u in cand:
                    if n == nxt_speaker:
                        chosen_u = u
                        break
                if chosen_u is not None and chosen_u == max_u:
                    urgency_consistency_hits += 1

            # planned interrupt events
            for name, p in plans.items():
                if str(p.get("action", "")).lower() != "interrupt":
                    continue
                planned_interrupts += 1
                planned_interrupt_cnt_by_agent[name] += 1

                realized = (nxt_speaker == name)
                if realized:
                    realized_interrupt_speaker += 1
                    realized_interrupt_cnt_by_agent[name] += 1

                th = str(p.get("thought", "") or "")
                toks = tokenize(th)
                if toks:
                    if realized:
                        thought_tokens_interrupt_realized.update(toks)
                    else:
                        thought_tokens_interrupt_not_realized.update(toks)

                # effect only when realized, because it is an actual intervention
                if realized:
                    dg = suffix_delta(turns, gold_series_normals, nxt) if gold else None
                    dt = suffix_delta(turns, target_series_normals, nxt) if target else None
                    if dg is not None:
                        delta_gold_list.append(dg)
                    if dt is not None:
                        delta_target_list.append(dt)

                    helpful = None
                    if dg is not None or dt is not None:
                        helpful = False
                        if dg is not None and dg > 0:
                            helpful = True
                        if dt is not None and dt < 0:
                            helpful = True
                        helpful_flags.append(bool(helpful))

                    prev_content = str(by_turn.get(nxt - 1, {}).get("content", "") or "")
                    before_evidence_flags.append(not detect_has_evidence(prev_content))

        urgency_rate = None
        if urgency_consistency_total > 0:
            urgency_rate = urgency_consistency_hits / urgency_consistency_total

        helpful_rate = None
        if helpful_flags:
            helpful_rate = sum(1 for x in helpful_flags if x) / len(helpful_flags)

        mean_dg = statistics.fmean(delta_gold_list) if delta_gold_list else None
        mean_dt = statistics.fmean(delta_target_list) if delta_target_list else None

        before_evidence_rate = None
        if before_evidence_flags:
            before_evidence_rate = sum(1 for x in before_evidence_flags if x) / len(before_evidence_flags)

        summaries.append(
            PerProblemSummary(
                problem_id=pid,
                turns=max_turn,
                agents=agents,
                adversaries=adversaries,
                normals=normals,
                gold=gold,
                target=target,
                initial_correct_count=init_correct,
                planned_interrupts=planned_interrupts,
                realized_interrupt_speaker=realized_interrupt_speaker,
                logged_interrupt_events=logged_interrupt_events,
                urgency_consistency_rate=urgency_rate,
                helpful_rate_normals=helpful_rate,
                mean_delta_gold_normals=mean_dg,
                mean_delta_target_normals=mean_dt,
                interrupt_before_evidence_rate=before_evidence_rate,
                correct_final=correct_by_problem.get(pid),
            )
        )

    # global summaries
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "run_action_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # dominance from realized interrupts
    realized_hhi = hhi_from_counts(realized_interrupt_cnt_by_agent)
    realized_max_share = max_share_from_counts(realized_interrupt_cnt_by_agent)

    logged_hhi = hhi_from_counts(logged_interrupt_cnt_by_agent)
    logged_max_share = max_share_from_counts(logged_interrupt_cnt_by_agent)

    def top_words(counter: Counter, k: int = 20) -> str:
        items = counter.most_common(k)
        return ", ".join(f"{w}:{c}" for w, c in items)

    # aggregate rates
    all_planned = sum(s.planned_interrupts for s in summaries)
    all_realized = sum(s.realized_interrupt_speaker for s in summaries)
    all_logged = sum(s.logged_interrupt_events for s in summaries)

    # per-problem averages where defined
    def mean_defined(vals: List[Optional[float]]) -> Optional[float]:
        xs = [v for v in vals if isinstance(v, float)]
        return statistics.fmean(xs) if xs else None

    mean_helpful = mean_defined([s.helpful_rate_normals for s in summaries])
    mean_dg = mean_defined([s.mean_delta_gold_normals for s in summaries])
    mean_dt = mean_defined([s.mean_delta_target_normals for s in summaries])
    mean_urg = mean_defined([s.urgency_consistency_rate for s in summaries])
    mean_before_ev = mean_defined([s.interrupt_before_evidence_rate for s in summaries])

    # markdown report
    md = []
    md.append("# Run action analysis report")
    md.append("")
    md.append(f"Run dir: {run_dir}")
    md.append(f"Included problems: {len(summaries)}")
    md.append(f"Problem range: {args.start} to {args.end if args.end != -1 else 'last'}")
    md.append(f"Max problems: {args.max_n if args.max_n != -1 else 'none'}")
    md.append(f"Filter only 1 correct 2 wrong in initial: {args.only_1c2w}")
    md.append("")

    md.append("## Action distribution from agent plans")
    for act, c in action_cnt.most_common():
        md.append(f"- {act}: {c}")
    md.append("")

    md.append("## Purpose distribution from agent plans")
    for pur, c in purpose_cnt.most_common():
        md.append(f"- {pur}: {c}")
    md.append("")

    md.append("## Purpose distribution by action")
    for (act, pur), c in action_purpose_cnt.most_common():
        md.append(f"- {act} -> {pur}: {c}")
    md.append("")

    md.append("## Urgency summary by action")
    for act, vals in urgency_by_action.items():
        vals2 = [v for v in vals if isinstance(v, int)]
        if not vals2:
            continue
        md.append(f"- {act}: mean {statistics.fmean(vals2):.3f}, median {statistics.median(vals2):.3f}, n {len(vals2)}")
    md.append("")

    md.append("## Interruption counts")
    md.append(f"- Planned interrupts in plans: {all_planned}")
    md.append(f"- Realized as next speaker: {all_realized}")
    md.append(f"- Logged event_type interrupt: {all_logged}")
    md.append("")
    md.append("Interpretation: planned interrupt and logged interrupt are different. Realized as next speaker is the most direct evidence of actual intervention.")
    md.append("")

    md.append("## Urgency consistency")
    md.append(f"- Mean per-problem urgency consistency rate: {mean_urg:.4f}" if mean_urg is not None else "- Mean per-problem urgency consistency rate: nan")
    md.append("Interpretation: high values support that speaker selection matches agents' urgency judgments.")
    md.append("")

    md.append("## Dominance check")
    md.append("### Based on realized interrupts")
    md.append(f"- Interrupter HHI: {realized_hhi:.4f}" if not math.isnan(realized_hhi) else "- Interrupter HHI: nan")
    md.append(f"- Max interrupter share: {realized_max_share:.4f}" if not math.isnan(realized_max_share) else "- Max interrupter share: nan")
    md.append("### Based on logged interrupt events")
    md.append(f"- Interrupter HHI: {logged_hhi:.4f}" if not math.isnan(logged_hhi) else "- Interrupter HHI: nan")
    md.append(f"- Max interrupter share: {logged_max_share:.4f}" if not math.isnan(logged_max_share) else "- Max interrupter share: nan")
    md.append("")

    if interrupt_pair_cnt:
        md.append("## Who interrupts whom from logged events")
        for (src, dst), c in interrupt_pair_cnt.most_common():
            md.append(f"- {src} -> {dst}: {c}")
        md.append("")

    md.append("## Effect of realized interrupts on normal agents")
    md.append(f"- Mean helpful rate: {mean_helpful:.4f}" if mean_helpful is not None else "- Mean helpful rate: nan")
    md.append(f"- Mean delta gold support: {mean_dg:.4f}" if mean_dg is not None else "- Mean delta gold support: nan")
    md.append(f"- Mean delta target support: {mean_dt:.4f}" if mean_dt is not None else "- Mean delta target support: nan")
    md.append(f"- Mean interruption before evidence rate: {mean_before_ev:.4f}" if mean_before_ev is not None else "- Mean interruption before evidence rate: nan")
    md.append("")
    md.append("Interpretation: positive delta gold and negative delta target are favorable. High before evidence rate suggests interruptions often happen before reasons are given.")
    md.append("")

    md.append("## Thought keyword profile")
    md.append(f"- speak: {top_words(thought_tokens_by_action.get('speak', Counter()))}")
    md.append(f"- listen: {top_words(thought_tokens_by_action.get('listen', Counter()))}")
    md.append(f"- interrupt in plans realized: {top_words(thought_tokens_interrupt_realized)}")
    md.append(f"- interrupt in plans not realized: {top_words(thought_tokens_interrupt_not_realized)}")
    md.append("")

    md.append("## Per-problem summary")
    md.append("| problem | turns | init correct | planned int | realized int | logged int | urg consistency | helpful rate | d gold | d target | before evidence | final correct |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        md.append(
            f"| {s.problem_id} | {s.turns} | {s.initial_correct_count if s.initial_correct_count is not None else ''} | "
            f"{s.planned_interrupts} | {s.realized_interrupt_speaker} | {s.logged_interrupt_events} | "
            f"{'' if s.urgency_consistency_rate is None else f'{s.urgency_consistency_rate:.3f}'} | "
            f"{'' if s.helpful_rate_normals is None else f'{s.helpful_rate_normals:.3f}'} | "
            f"{'' if s.mean_delta_gold_normals is None else f'{s.mean_delta_gold_normals:.4f}'} | "
            f"{'' if s.mean_delta_target_normals is None else f'{s.mean_delta_target_normals:.4f}'} | "
            f"{'' if s.interrupt_before_evidence_rate is None else f'{s.interrupt_before_evidence_rate:.3f}'} | "
            f"{'' if s.correct_final is None else ('1' if s.correct_final else '0')} |"
        )
    md.append("")

    report_path = out_dir / "run_action_analysis_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    raw_path = out_dir / "run_action_analysis_raw.json"
    raw = {
        "run_dir": str(run_dir),
        "included_problems": len(summaries),
        "problem_range": {"start": args.start, "end": args.end},
        "only_1c2w": args.only_1c2w,
        "counts": {
            "planned_interrupts": all_planned,
            "realized_interrupts": all_realized,
            "logged_interrupt_events": all_logged,
        },
        "action_counts": dict(action_cnt),
        "purpose_counts": dict(purpose_cnt),
        "action_purpose_counts": {f"{k[0]}->{k[1]}": v for k, v in action_purpose_cnt.items()},
        "planned_interrupt_by_agent": dict(planned_interrupt_cnt_by_agent),
        "realized_interrupt_by_agent": dict(realized_interrupt_cnt_by_agent),
        "logged_interrupt_by_agent": dict(logged_interrupt_cnt_by_agent),
        "dominance": {
            "realized_hhi": realized_hhi,
            "realized_max_share": realized_max_share,
            "logged_hhi": logged_hhi,
            "logged_max_share": logged_max_share,
        },
        "interrupt_pairs_logged": {f"{k[0]}->{k[1]}": v for k, v in interrupt_pair_cnt.items()},
        "means": {
            "mean_helpful_rate_normals": mean_helpful,
            "mean_delta_gold_normals": mean_dg,
            "mean_delta_target_normals": mean_dt,
            "mean_urgency_consistency_rate": mean_urg,
            "mean_before_evidence_rate": mean_before_ev,
        },
        "per_problem": [s.__dict__ for s in summaries],
    }
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved report: {report_path}")
    print(f"Saved raw: {raw_path}")


if __name__ == "__main__":
    main()
