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
    temperature=0.0,
)

# -------------------------------
# ユーティリティ
# -------------------------------
def format_prompt(question: str, choices: list[str]) -> str:
    letters = ["A", "B", "C", "D"]
    lines = [f"Question: {question}", "Please think step by step."]
    for i, ch in enumerate(choices[:4]):  # 念のため4択に制限
        lines.append(f"{letters[i]}. {ch}")
    # 答えだけを出させるよう強制
    lines.append("Answer (only A, B, C, or D):")
    return "\n".join(lines)

def extract_answer(output: str) -> str:
    """モデル出力から最初に見つかった A/B/C/D を抽出"""
    match = re.search(r"\b([ABCD])\b", output)
    if match:
        return match.group(1)
    # バックアップ（句読点つき "A." や "Answer: C" にも対応）
    match = re.search(r"([ABCD])", output)
    return match.group(1) if match else ""

# -------------------------------
# メイン処理
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run MMLU baseline with llama.cpp")
    parser.add_argument("--num", type=int, default=50,
                        help="Number of questions to evaluate (default: 50). Use -1 for all.")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "dev", "validation"], help="MMLU split (default: test)")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="Shuffle questions before sampling (default: True)")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                        help="Disable shuffling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
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

            prompt = format_prompt(question, choices)

            # llama_cpp completion API
            resp = llm.create_completion(
                prompt=prompt,
                max_tokens=4,
                temperature=0.0,
                stop=["\n"],  # 一行で止める
            )
            output = resp["choices"][0]["text"].strip()
            pred = extract_answer(output)

            is_correct = (pred == gold)
            if is_correct:
                correct += 1

            # コンソール出力
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
                        "model_output_raw": output,
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
