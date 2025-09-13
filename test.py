from datasets import load_dataset
from llama_cpp import Llama
import re
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

# -------------------------------
# モデル設定
# -------------------------------
MODEL_PATH = "models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=2000,
    temperature=0.7,
)

# -------------------------------
# レスポンスフォーマット（JSON Schema で厳密化）
# -------------------------------
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "mmlu_reasoned_choice",
        "schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
            },
            "required": ["reason", "answer"],
            "additionalProperties": False,
        },
    },
}

# -------------------------------
# ユーティリティ
# -------------------------------
def format_messages(question: str, choices: list[str]):
    """chat completion 用のメッセージを組み立てる。"""
    letters = ["A", "B", "C", "D"]
    choice_lines = [f"{letters[i]}. {ch}" for i, ch in enumerate(choices[:4])]
    instruction_block = (
        "# Instruction\n"
        "- Derive your solution to the given question through step-by-step reasoning.\n"
        "- Provide your answer and the reasoning.\n"
        '- Output JSON only with two keys: "reason" and "answer".'
    )

    system = {
        "role": "system",
        "content": (
            "You are a character who is extroverted, conscientious, agreeable, open to experience, emotionally stable. "
            "Follow the instructions strictly and return only valid JSON that matches the provided schema."
        ),
    }
    user = {
        "role": "user",
        "content": "\n".join(
            [
                instruction_block,
                "# Question",
                f"Question: {question}",
                "",
                *choice_lines,
                "",
                "# Constraints\n"
                "Answer must be one of A, B, C, or D.",
                "Return only JSON. No prose, no markdown.",
            ]
        ),
    }
    return [system, user]

def parse_json_output(output: str):
    """
    モデル出力（JSON 文字列想定）から reason と answer を安全に取り出す。
    フォールバックとして A/B/C/D 抽出も実装。
    """
    reason, answer = "", ""
    try:
        obj = json.loads(output)
        reason = obj.get("reason", "")
        answer = obj.get("answer", "")
    except Exception:
        # フォールバック：A/B/C/D を拾う
        m = re.search(r'\b([ABCD])\b', output)
        answer = m.group(1) if m else ""
        # JSON でない場合は出力全体を理由として残す
        reason = output.strip()

    return reason, answer

# -------------------------------
# メイン処理
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run MMLU baseline with llama.cpp (JSON reasoning+answer logging)")
    parser.add_argument("--num", type=int, default=50,
                        help="Number of questions to evaluate (default: 50). Use -1 for all.")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "dev", "validation"], help="MMLU split (default: test)")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="Shuffle questions before sampling (default: True)")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                        help="Disable shuffling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens for chat completion (default: 256)")
    args = parser.parse_args()

    # ログ用ディレクトリ
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(__file__).resolve().parent / "logs" / f"baseline_{run_ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    result_path = run_root / "results.jsonl"

    # データ読み込み
    ds = load_dataset("cais/mmlu", "all", split=args.split)

    # 問題数の決定（-1 なら全問）
    total_available = len(ds)
    if args.num < 0:
        total = total_available
    else:
        total = min(args.num, total_available)

    # シャッフルして先頭 N 件を使用
    indices = list(range(total_available))
    if args.shuffle:
        rnd = random.Random(args.seed)
        rnd.shuffle(indices)
    selected = indices[:total]

    correct = 0
    with result_path.open("w", encoding="utf-8") as fp:
        for i, idx in enumerate(selected, start=1):
            ex = ds[idx]
            question = ex["question"]
            choices = ex["choices"]
            gold = ["A", "B", "C", "D"][ex["answer"]]

            messages = format_messages(question, choices)

            # llama_cpp chat completion API（JSON Schema で厳密な JSON 出力を要求）
            resp = llm.create_chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=args.max_tokens,
                response_format=RESPONSE_FORMAT,
            )
            output = resp["choices"][0]["message"]["content"].strip()

            # JSON 解析
            model_reason, model_answer = parse_json_output(output)
            pred = model_answer

            is_correct = (pred == gold)
            if is_correct:
                correct += 1

            # コンソール出力（理由は長いことがあるので省略／必要なら適宜表示）
            print(f"[Q{i:03}] Pred={pred or '∅'} | Gold={gold} | {'✅' if is_correct else '❌'}")

            # ログ（1行1レコード）
            fp.write(
                json.dumps(
                    {
                        "index_in_split": idx,
                        "question_id": i,
                        "pred": pred,
                        "gold": gold,
                        "correct": is_correct,
                        "question": question,
                        "choices": choices[:4],
                        "model_reason": model_reason, # JSON の reason
                        "model_answer": model_answer, # JSON の answer（A/B/C/D）
                        "input_messages": messages,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fp.flush()

        accuracy = correct / total if total else 0.0
        print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")

        # サマリも追記
        fp.write(
            json.dumps(
                {
                    "summary": "accuracy",
                    "correct": correct,
                    "total": total,
                    "accuracy": accuracy,
                    "split": args.split,
                    "shuffled": args.shuffle,
                    "seed": args.seed if args.shuffle else None,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

if __name__ == "__main__":
    main()
 