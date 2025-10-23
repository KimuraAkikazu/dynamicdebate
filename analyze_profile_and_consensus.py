
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from typing import Dict, Any, List, Optional

def parse_discussion_log(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Per-turn events (turn as int); include 0 for completeness but it's usually "plan" bundle
    turns = [e for e in data if isinstance(e.get("turn"), int)]
    # Early stop event
    early = next((e for e in data if e.get("event_type") == "early_stop"), None)

    # Event counts
    num_interrupts = sum(1 for e in turns if e.get("event_type") == "interrupt")
    num_utterances = sum(1 for e in turns if e.get("event_type") == "utterance")
    num_silence = sum(1 for e in turns if e.get("event_type") == "silence")
    total_turns = len(turns)

    # Speaking counts per agent
    speak_counts: Dict[str, int] = {}
    for e in turns:
        if e.get("event_type") in {"utterance", "interrupt"}:
            spk = e.get("speaker")
            if spk:
                speak_counts[spk] = speak_counts.get(spk, 0) + 1

    # Aggregate plan/action/intent/urgency from each turn's agent_actions
    action_counts: Dict[str, Dict[str, int]] = {}
    intent_counts: Dict[str, Dict[str, int]] = {}
    urgencies: Dict[str, List[int]] = {}

    for e in turns:
        for aa in e.get("agent_actions", []):
            name = aa.get("agent_name")
            plan = aa.get("action_plan") or {}
            if not isinstance(plan, dict) or not name:
                continue
            act = plan.get("action")
            intent = plan.get("intent")
            urg = plan.get("urgency")
            # accumulate
            if name:
                action_counts.setdefault(name, {})
                intent_counts.setdefault(name, {})
                urgencies.setdefault(name, [])
                if act:
                    action_counts[name][act] = action_counts[name].get(act, 0) + 1
                if intent:
                    intent_counts[name][intent] = intent_counts[name].get(intent, 0) + 1
                if isinstance(urg, int):
                    urgencies[name].append(urg)

    had_early_stop = early is not None
    consensus_turn = early.get("turn") if had_early_stop else None
    consensus_answer = early.get("answer") if had_early_stop else None
    consensus_streak = early.get("streak") if had_early_stop else None

    return {
        "num_interrupts": num_interrupts,
        "num_utterances": num_utterances,
        "num_silence": num_silence,
        "total_turns": total_turns,
        "speak_counts": speak_counts,
        "action_counts": action_counts,
        "intent_counts": intent_counts,
        "urgencies": urgencies,
        "had_early_stop": had_early_stop,
        "consensus_turn": consensus_turn,
        "consensus_answer": consensus_answer,
        "consensus_streak": consensus_streak,
    }

def analyze_profile_and_consensus(run_dir: Path) -> Dict[str, Any]:
    probs = sorted([p for p in run_dir.glob("problem_*") if p.is_dir()])

    # Accumulators
    total_problems = 0
    total_interrupts = 0
    total_utterances = 0
    total_silence = 0

    per_problem_rows: List[Dict[str, Any]] = []

    agent_speaks: Dict[str, int] = {}
    agent_action_counts: Dict[str, Dict[str, int]] = {}
    agent_intent_counts: Dict[str, Dict[str, int]] = {}
    agent_urgencies: Dict[str, List[int]] = {}

    early_count = 0
    consensus_turns: List[int] = []
    consensus_answers: Dict[str, int] = {}

    for prob in probs:
        log_path = prob / "discussion_log.json"
        if not log_path.exists():
            continue
        parsed = parse_discussion_log(log_path)

        total_problems += 1
        total_interrupts += parsed["num_interrupts"]
        total_utterances += parsed["num_utterances"]
        total_silence += parsed["num_silence"]

        # per-problem
        per_problem_rows.append({
            "problem": prob.name,
            "num_interrupts": parsed["num_interrupts"],
            "num_utterances": parsed["num_utterances"],
            "num_silence": parsed["num_silence"],
            "total_turns": parsed["total_turns"],
            "had_early_stop": parsed["had_early_stop"],
            "consensus_turn": parsed["consensus_turn"],
            "consensus_answer": parsed["consensus_answer"],
            "consensus_streak": parsed["consensus_streak"],
        })

        if parsed["had_early_stop"]:
            early_count += 1
            if isinstance(parsed["consensus_turn"], int):
                consensus_turns.append(parsed["consensus_turn"])
            ca = parsed["consensus_answer"]
            if isinstance(ca, str) and ca:
                consensus_answers[ca] = max(0, consensus_answers.get(ca, 0)) + 1

        # aggregate agent profiles
        for name, c in parsed["speak_counts"].items():
            agent_speaks[name] = agent_speaks.get(name, 0) + c
        for name, acts in parsed["action_counts"].items():
            agent_action_counts.setdefault(name, {})
            for k, v in acts.items():
                agent_action_counts[name][k] = agent_action_counts[name].get(k, 0) + v
        for name, ints in parsed["intent_counts"].items():
            agent_intent_counts.setdefault(name, {})
            for k, v in ints.items():
                agent_intent_counts[name][k] = agent_intent_counts[name].get(k, 0) + v
        for name, us in parsed["urgencies"].items():
            agent_urgencies.setdefault(name, []).extend(us)

    # Build per-agent profile table
    total_speaks = sum(agent_speaks.values()) if agent_speaks else 0
    per_agent_rows: List[Dict[str, Any]] = []
    all_agents = sorted(set(list(agent_speaks.keys()) + list(agent_action_counts.keys()) +
                            list(agent_intent_counts.keys()) + list(agent_urgencies.keys())))

    default_actions = ["listen", "speak", "interrupt"]
    default_intents = ["agree","disagree","summarize","confirmation","proposal","question","conclusion","think"]

    for name in all_agents:
        speaks = agent_speaks.get(name, 0)
        acts = agent_action_counts.get(name, {})
        ints = agent_intent_counts.get(name, {})
        urg_list = agent_urgencies.get(name, [])

        row = {
            "agent": name,
            "speaks": speaks,
            "speaking_share": (speaks / total_speaks) if total_speaks else 0.0,
            "urg_mean": (sum(urg_list)/len(urg_list)) if urg_list else None,
            "urg_median": (sorted(urg_list)[len(urg_list)//2] if urg_list else None),
            "urg_std": (pd.Series(urg_list).std(ddof=1) if len(urg_list) > 1 else None),
        }
        # actions
        total_a = sum(acts.values())
        for a in sorted(set(list(acts.keys()) + default_actions)):
            cnt = acts.get(a, 0)
            row[f"action_{a}_count"] = cnt
            row[f"action_{a}_prop"] = (cnt/total_a) if total_a else 0.0
        # intents
        total_i = sum(ints.values())
        for it in sorted(set(list(ints.keys()) + default_intents)):
            cnt = ints.get(it, 0)
            row[f"intent_{it}_count"] = cnt
            row[f"intent_{it}_prop"] = (cnt/total_i) if total_i else 0.0

        per_agent_rows.append(row)

    # Consensus dynamics summary
    summary = {
        "total_problems": total_problems,
        "avg_interrupts_per_problem": (total_interrupts/total_problems) if total_problems else 0.0,
        "total_interrupts": total_interrupts,
        "total_utterances": total_utterances,
        "total_silence_events": total_silence,
        "silence_rate_over_all_events": (total_silence / (total_silence + total_utterances)) if (total_silence + total_utterances) else 0.0,
        "early_stop_rate": (early_count/total_problems) if total_problems else 0.0,
        "avg_consensus_turn_among_early": (sum(consensus_turns)/len(consensus_turns)) if consensus_turns else None,
        "consensus_answer_counts": consensus_answers,
    }

    return {
        "per_problem": pd.DataFrame(per_problem_rows),
        "per_agent_profile": pd.DataFrame(per_agent_rows),
        "summary": summary,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze behavior profile and consensus dynamics for a run.")
    parser.add_argument("--run_dir", default="logs/run_20250917_014829", help="Path to run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "analysis_profile_consensus"
    out_dir.mkdir(parents=True, exist_ok=True)

    res = analyze_profile_and_consensus(run_dir)
    per_problem = res["per_problem"]
    per_agent = res["per_agent_profile"]
    summary = res["summary"]

    per_problem.to_csv(out_dir / "per_problem_basic.csv", index=False)
    per_agent.to_csv(out_dir / "per_agent_profile.csv", index=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\nSaved:")
    print(out_dir / "per_problem_basic.csv")
    print(out_dir / "per_agent_profile.csv")
    print(out_dir / "summary.json")

if __name__ == "__main__":
    main()
