from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from llama_cpp import Llama

from build_initial_pool import (
    LETTERS,
    RESPONSE_FORMAT,
    chat_complete,
    format_messages,
    parse_json_output,
)


DEFAULT_POOL_PATH = (
    Path(__file__).resolve().parent
    / "Initial_answer"
    / "initial_pool_20251219_152013"
    / "initial_pool.jsonl"
)


def load_pool_entries(pool_path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with pool_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e

            if "question" not in rec or "choices" not in rec or "gold" not in rec:
                raise ValueError(
                    f"Missing required keys at line {line_no}. Required: question, choices, gold."
                )
            entries.append(rec)
    return entries


def normalize_gold(gold: Any) -> str:
    if isinstance(gold, str):
        g = gold.strip().upper()
        if g in LETTERS:
            return g
    if isinstance(gold, int) and 0 <= gold < len(LETTERS):
        return LETTERS[gold]
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate single-LLM accuracy on initial_pool.jsonl (one-shot per question)."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to gguf model")
    parser.add_argument("--pool_path", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--num", type=int, default=-1, help="Number of questions (-1 for all)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--n_ctx", type=int, default=2048)
    parser.add_argument("--gen_seed0", type=int, default=1000)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    pool_path = args.pool_path.resolve()
    if not pool_path.exists():
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    entries = load_pool_entries(pool_path)
    if not entries:
        raise ValueError(f"No entries found in pool file: {pool_path}")

    if args.num < 0:
        selected_indices = list(range(len(entries)))
    else:
        selected_indices = list(range(min(args.num, len(entries))))

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(__file__).resolve().parent / "logs" / f"single_llm_eval_{run_ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    pred_path = run_root / "predictions.jsonl"
    summary_path = run_root / "summary.json"

    llm = Llama(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        temperature=args.temperature,
    )

    total = len(selected_indices)
    correct = 0
    parse_fail_or_empty = 0

    with pred_path.open("w", encoding="utf-8") as fp:
        for q_ord, entry_idx in enumerate(selected_indices, start=1):
            rec = entries[entry_idx]
            question = str(rec["question"])
            choices = list(rec["choices"])[:4]
            gold = normalize_gold(rec["gold"])
            subject = rec.get("subject", "unknown")
            index_in_split = rec.get("index_in_split", None)

            if len(choices) < 4 or gold not in LETTERS:
                parse_fail_or_empty += 1
                pred = ""
                reason = "Invalid pool entry format (choices<4 or gold label invalid)."
                raw = ""
                is_correct = False
            else:
                messages = format_messages(question, choices)
                seed = args.gen_seed0 + (q_ord * 100000)
                pred = ""
                reason = ""
                raw = ""
                is_correct = False

                for attempt in range(args.retries + 1):
                    raw = chat_complete(
                        llm=llm,
                        messages=messages,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        response_format=RESPONSE_FORMAT,
                        seed=seed + attempt,
                    )
                    reason, pred = parse_json_output(raw)
                    if pred in LETTERS:
                        break

                if pred not in LETTERS:
                    parse_fail_or_empty += 1
                is_correct = pred == gold

            if is_correct:
                correct += 1

            out = {
                "question_ord": q_ord,
                "pool_line_index": entry_idx,
                "index_in_split": index_in_split,
                "subject": subject,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "question": question,
                "choices": choices,
                "model_reason": reason,
                "model_output_raw": raw,
            }
            fp.write(json.dumps(out, ensure_ascii=False) + "\n")
            if q_ord % 50 == 0 or q_ord == total:
                print(f"[{q_ord}/{total}] acc={correct / q_ord:.4f} parse_fail={parse_fail_or_empty}")

    accuracy = correct / total if total else 0.0
    summary = {
        "run_dir": str(run_root),
        "pool_path": str(pool_path),
        "model_path": args.model_path,
        "params": {
            "num": args.num,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "n_gpu_layers": args.n_gpu_layers,
            "n_ctx": args.n_ctx,
            "gen_seed0": args.gen_seed0,
            "retries": args.retries,
        },
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "parse_fail_or_empty": parse_fail_or_empty,
        "outputs": {"predictions_jsonl": str(pred_path), "summary_json": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Accuracy: {correct}/{total} = {accuracy:.4f}")
    print(f"Run dir: {run_root}")


if __name__ == "__main__":
    main()
