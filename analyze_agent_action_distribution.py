from __future__ import annotations
from pathlib import Path
import json
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from typing import Dict, Any, List

def parse_actions_from_log(path: Path) -> Dict[str, Dict[str, int]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    turns = [e for e in data if isinstance(e.get("turn"), int)]
    action_counts: Dict[str, Dict[str, int]] = {}

    for e in turns:
        for aa in e.get("agent_actions", []):
            name = aa.get("agent_name")
            plan = aa.get("action_plan") or {}
            if not isinstance(plan, dict) or not name:
                continue
            act = plan.get("action")
            if not act:
                continue
            action_counts.setdefault(name, {})
            action_counts[name][act] = action_counts[name].get(act, 0) + 1
    return action_counts

def aggregate_agent_actions(run_dir: Path) -> pd.DataFrame:
    probs = sorted([p for p in run_dir.glob("problem_*") if p.is_dir()])

    agent_totals: Dict[str, Dict[str, int]] = {}
    for prob in probs:
        log_path = prob / "discussion_log.json"
        if not log_path.exists():
            continue
        acts = parse_actions_from_log(log_path)
        for name, action_dict in acts.items():
            agent_totals.setdefault(name, {})
            for a, c in action_dict.items():
                agent_totals[name][a] = agent_totals[name].get(a, 0) + c

    # Build dataframe with counts and proportions
    rows: List[Dict[str, Any]] = []
    all_actions = sorted({a for d in agent_totals.values() for a in d.keys()})
    for name, counts in agent_totals.items():
        total = sum(counts.values())
        row: Dict[str, Any] = {"agent": name, "total_actions": total}
        for a in all_actions:
            cnt = counts.get(a, 0)
            row[f"{a}_count"] = cnt
            row[f"{a}_prop"] = (cnt / total) if total else 0.0
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["agent"]).reset_index(drop=True)
    return df

def save_bar_charts(df: pd.DataFrame, out_dir: Path) -> None:
    # One chart per agent: action proportions
    # Each chart single-axes, default matplotlib colors, no styles set.
    action_cols = [c for c in df.columns if c.endswith("_prop") and c != "total_actions_prop"]
    for _, row in df.iterrows():
        agent = row["agent"]
        actions = [c.replace("_prop", "") for c in action_cols]
        values = [row[c] for c in action_cols]
        plt.figure()
        plt.bar(actions, values)
        plt.title(f"Action distribution (proportion) - {agent}")
        plt.ylabel("Proportion")
        plt.xlabel("Action")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_path = out_dir / f"actions_{agent}.png"
        plt.savefig(out_path)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Agent-wise action distribution analysis")
    parser.add_argument("--run_dir", default="logs/run_20250917_014829", help="Path to run directory")
    parser.add_argument("--no_plots", action="store_true", help="Do not generate bar charts")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "analysis_agent_actions"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = aggregate_agent_actions(run_dir)
    out_csv = out_dir / "per_agent_action_distribution.csv"
    df.to_csv(out_csv, index=False)

    if not args.no_plots and not df.empty:
        save_bar_charts(df, out_dir)

    print("Saved:", out_csv)
    if not args.no_plots and not df.empty:
        print("Saved per-agent charts to:", out_dir)

if __name__ == "__main__":
    main()
