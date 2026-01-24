from datasets import load_dataset
from llama_cpp import Llama
import re
import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# -------------------------------
# レスポンスフォーマット（json_object + schema）
# -------------------------------
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
    },
    "required": ["reason", "answer"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_object",
    "schema": JSON_SCHEMA,
}

LETTERS = ["A", "B", "C", "D"]


# -------------------------------
# プロンプト
# -------------------------------

def format_messages(question: str, choices: list[str]) -> list[dict[str, str]]:
    """ユーザー指定のプロンプト形式でメッセージを組み立てる。"""
    choice_lines = [f"{LETTERS[i]}. {ch}" for i, ch in enumerate(choices[:4])]
    topic = "\n".join([question, "", *choice_lines])

    system = {
        "role": "system",
        "content": "Follow the instructions strictly and return only valid JSON that matches the provided schema.",
    }

    user = {
        "role": "user",
        "content": f"""# Question
{topic}

# Instructions
- Derive your solution to the given question through step-by-step reasoning.
- Provide your answer and the reason behind it.
- Provide your response in the following Output format.

# Output format
Return strictly a JSON object only.
{{
    "reason": "Detailed reasoning for your choice (within 300 words).",
    "answer": "one of 'A', 'B', 'C', or 'D'"
}}"""
    }

    return [system, user]



# -------------------------------
# JSON パース（壊れた出力へのフォールバック）
# -------------------------------
def _try_extract_json_object(text: str) -> Optional[str]:
    """
    文字列中にJSONオブジェクトが混ざっている場合に { ... } を抜き出す。
    雑にやるが、ログ作成用途としては十分。
    """
    # 最初の { から最後の } までを抜く（過剰マッチ回避のため最短を試す）
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    if not candidates:
        return None
    # いちばん長い候補を優先（キー2つの想定なので短いことが多いが、念のため）
    candidates.sort(key=len, reverse=True)
    return candidates[0]


def parse_json_output(output: str) -> Tuple[str, str]:
    """
    (reason, answer) を返す。失敗したら (output, A/B/C/D抽出) へフォールバック。
    """
    reason, answer = "", ""
    raw = output.strip()

    def normalize_ans(a: str) -> str:
        a = (a or "").strip().upper()
        return a if a in LETTERS else ""

    # 1) そのままjson.loads
    try:
        obj = json.loads(raw)
        reason = str(obj.get("reason", "")).strip()
        answer = normalize_ans(obj.get("answer", ""))
        return reason, answer
    except Exception:
        pass

    # 2) 文字列中からJSON部分だけ抽出してjson.loads
    extracted = _try_extract_json_object(raw)
    if extracted:
        try:
            obj = json.loads(extracted)
            reason = str(obj.get("reason", "")).strip()
            answer = normalize_ans(obj.get("answer", ""))
            return reason, answer
        except Exception:
            pass

    # 3) フォールバック：A/B/C/D を拾う
    m = re.search(r"\b([ABCD])\b", raw.upper())
    answer = m.group(1) if m else ""
    reason = raw
    return reason, answer


# -------------------------------
# llama.cpp 呼び出し（seed対応が無い場合にも耐える）
# -------------------------------
def chat_complete(
    llm: Llama,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    response_format: Dict[str, Any],
    seed: Optional[int],
) -> str:
    kwargs = dict(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )

    # seed がサポートされていれば渡す。ダメなら外して実行。
    if seed is not None:
        kwargs["seed"] = seed

    try:
        resp = llm.create_chat_completion(**kwargs)
    except TypeError:
        # seed未対応など
        kwargs.pop("seed", None)
        resp = llm.create_chat_completion(**kwargs)

    return resp["choices"][0]["message"]["content"].strip()


# -------------------------------
# メイン
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build initial answer pool for MMLU (correct/wrong samples per question)")
    # Data
    parser.add_argument("--split", type=str, default="test", choices=["test", "dev", "validation"])
    parser.add_argument("--num", type=int, default=-1, help="Number of questions (-1 for all)")
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--data_seed", type=int, default=42)
    # Generation
    parser.add_argument("--k", type=int, default=5, help="Samples per question (default: 5)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--gen_seed0", type=int, default=1000, help="Base seed for generation")
    parser.add_argument("--gen_seed_step", type=int, default=1, help="Seed increment per trial")
    parser.add_argument("--retries", type=int, default=2, help="Retries when parsing fails or answer empty")
    # Filter
    parser.add_argument("--easy_threshold", type=float, default=0.8, help="Exclude if correct_rate >= this")
    parser.add_argument("--hard_threshold", type=float, default=0.2, help="Exclude if correct_rate <= this")
    parser.add_argument("--min_correct", type=int, default=2, help="Keep if correct_count >= this")
    parser.add_argument("--max_correct", type=int, default=3, help="Keep if correct_count <= this")
    # Selection
    parser.add_argument("--keep_correct", type=int, default=2)
    parser.add_argument("--keep_wrong", type=int, default=2)
    parser.add_argument("--selection_seed", type=int, default=0)
    # Model
    parser.add_argument("--model_path", type=str, required=True, help="Path to gguf model")
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--n_ctx", type=int, default=2000)

    args = parser.parse_args()

    # Run directory
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(__file__).resolve().parent / "logs" / f"initial_pool_{run_ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    samples_path = run_root / "samples.jsonl"
    pool_path = run_root / "initial_pool.jsonl"
    summary_path = run_root / "summary.json"

    # Model init
    llm = Llama(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        temperature=args.temperature,
    )

    # Load dataset
    ds = load_dataset("cais/mmlu", "all", split=args.split)
    total_available = len(ds)

    if args.num < 0:
        total = total_available
    else:
        total = min(args.num, total_available)

    indices = list(range(total_available))
    if args.shuffle:
        rnd = random.Random(args.data_seed)
        rnd.shuffle(indices)
    selected = indices[:total]

    # Stats
    excluded = {
        "easy": 0,
        "hard": 0,
        "correct_count_out_of_range": 0,
        "not_enough_correct_or_wrong": 0,
        "parse_fail_or_empty": 0,
    }
    kept = 0

    # Open output files
    fp_samples = samples_path.open("w", encoding="utf-8")
    fp_pool = pool_path.open("w", encoding="utf-8")

    try:
        for qi, idx in enumerate(selected, start=1):
            ex = ds[idx]
            question = ex["question"]
            choices = ex["choices"][:4]
            gold = LETTERS[ex["answer"]]
            subject = ex.get("subject", None) or ex.get("task", None) or "unknown"

            messages = format_messages(question, choices)

            trials: List[Dict[str, Any]] = []
            correct_trials: List[Dict[str, Any]] = []
            wrong_trials: List[Dict[str, Any]] = []

            # K samples
            for t in range(args.k):
                seed = args.gen_seed0 + (qi * 100000) + (t * args.gen_seed_step)
                out = ""
                reason, ans = "", ""
                ok = False

                for attempt in range(args.retries + 1):
                    out = chat_complete(
                        llm=llm,
                        messages=messages,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        response_format=RESPONSE_FORMAT,
                        seed=seed + attempt,  # retry時はseedもずらす
                    )
                    reason, ans = parse_json_output(out)
                    if ans in LETTERS:
                        ok = True
                        break

                if not ok:
                    excluded["parse_fail_or_empty"] += 1

                is_correct = (ans == gold)

                rec = {
                    "index_in_split": idx,
                    "question_ord": qi,          # this run's order
                    "trial": t,
                    "seed": seed,
                    "temperature": args.temperature,
                    "subject": subject,
                    "pred": ans,
                    "gold": gold,
                    "correct": is_correct,
                    "question": question,
                    "choices": choices,
                    "model_output_raw": out,
                    "model_reason": reason,
                    "model_answer": ans,
                }
                trials.append(rec)

                # trial log (always)
                fp_samples.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if ans in LETTERS:
                    if is_correct:
                        correct_trials.append(rec)
                    else:
                        wrong_trials.append(rec)

            fp_samples.flush()

            correct_count = len(correct_trials)
            wrong_count = len(wrong_trials)
            correct_rate = correct_count / args.k if args.k else 0.0

            # Filters
            if correct_rate >= args.easy_threshold:
                excluded["easy"] += 1
                if qi % 50 == 0:
                    print(f"[{qi}/{total}] excluded(easy) rate={correct_rate:.2f}")
                continue

            if correct_rate <= args.hard_threshold:
                excluded["hard"] += 1
                if qi % 50 == 0:
                    print(f"[{qi}/{total}] excluded(hard) rate={correct_rate:.2f}")
                continue

            if not (args.min_correct <= correct_count <= args.max_correct):
                excluded["correct_count_out_of_range"] += 1
                if qi % 50 == 0:
                    print(f"[{qi}/{total}] excluded(correct_count) {correct_count}/{args.k}")
                continue

            if correct_count < args.keep_correct or wrong_count < args.keep_wrong:
                excluded["not_enough_correct_or_wrong"] += 1
                if qi % 50 == 0:
                    print(f"[{qi}/{total}] excluded(not_enough) c={correct_count} w={wrong_count}")
                continue

            # Select 2 correct + 2 wrong (reproducible)
            sel_rnd = random.Random(args.selection_seed + idx)
            picked_correct = sel_rnd.sample(correct_trials, k=args.keep_correct)
            picked_wrong = sel_rnd.sample(wrong_trials, k=args.keep_wrong)

            pool_rec = {
                "index_in_split": idx,
                "subject": subject,
                "question": question,
                "choices": choices,
                "gold": gold,
                "k": args.k,
                "correct_rate": correct_rate,
                "picked": [
                    {
                        "label": "correct",
                        "answer": r["model_answer"],
                        "reason": r["model_reason"],
                        "seed": r["seed"],
                        "trial": r["trial"],
                        "raw": r["model_output_raw"],
                    }
                    for r in picked_correct
                ]
                + [
                    {
                        "label": "wrong",
                        "answer": r["model_answer"],
                        "reason": r["model_reason"],
                        "seed": r["seed"],
                        "trial": r["trial"],
                        "raw": r["model_output_raw"],
                    }
                    for r in picked_wrong
                ],
            }

            fp_pool.write(json.dumps(pool_rec, ensure_ascii=False) + "\n")
            fp_pool.flush()

            kept += 1
            if qi % 50 == 0:
                print(f"[{qi}/{total}] kept={kept} (latest rate={correct_rate:.2f}, c={correct_count}, w={wrong_count})")

        # Summary
        summary = {
            "run_dir": str(run_root),
            "dataset": {"name": "cais/mmlu", "config": "all", "split": args.split},
            "model_path": args.model_path,
            "params": {
                "k": args.k,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "easy_threshold": args.easy_threshold,
                "hard_threshold": args.hard_threshold,
                "min_correct": args.min_correct,
                "max_correct": args.max_correct,
                "keep_correct": args.keep_correct,
                "keep_wrong": args.keep_wrong,
                "shuffle": args.shuffle,
                "data_seed": args.data_seed,
                "gen_seed0": args.gen_seed0,
                "gen_seed_step": args.gen_seed_step,
                "selection_seed": args.selection_seed,
            },
            "total_questions_processed": total,
            "kept_questions": kept,
            "excluded_counts": excluded,
            "outputs": {
                "samples_jsonl": str(samples_path),
                "initial_pool_jsonl": str(pool_path),
            },
        }

        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nDone.")
        print(f"Kept questions: {kept}/{total}")
        print(f"Run dir: {run_root}")

    finally:
        fp_samples.close()
        fp_pool.close()


if __name__ == "__main__":
    main()
