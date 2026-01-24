from __future__ import annotations
from llama_cpp import Llama
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

"""
sub.py
----
stream=True で LLM の出力を受け取り、ストリームの“トークン相当チャンク”を 8 個ごとに束ねて順次出力します。
（llama.cpp のストリームは実装により厳密な「トークン」ではなくテキストチャンクの場合がありますが、
  本スクリプトでは 1 チャンク=1 トークン相当としてカウントします。）

使い方例:
  python sub.py --task "次の式を解いて、方針も簡潔に説明してください: 2x + 3 = 11"
  python sub.py --task "以下の英文を日本語で要約してください: Large language models can collaborate..." --chunk_size 8

モデルの設定は、提示いただいた baseline スクリプトに合わせています。
"""

# -------------------------------
# モデル設定（必要に応じて書き換えてください）
# -------------------------------
MODEL_PATH = "models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,  # 可能なら GPU に全レイヤーを載せる
    n_ctx=2000,
    temperature=0.7,
)

# -------------------------------
# メッセージ整形
# -------------------------------

LETTERS = ["A", "B", "C", "D"]

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
        "content": f"""
# Question
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
# ストリームからテキストを取り出すヘルパ
# -------------------------------

def extract_stream_text(chunk: Dict[str, Any]) -> str:
    """llama_cpp のストリームチャンクからテキストを安全に取り出す。"""
    try:
        choice = chunk.get("choices", [{}])[0]
        # OpenAI 互換の streaming では delta.content に入ることが多い
        delta = choice.get("delta", {})
        if isinstance(delta, dict) and "content" in delta and delta["content"] is not None:
            return str(delta["content"])
        # 一部の実装では text に入ることがある
        if "text" in choice and choice["text"] is not None:
            return str(choice["text"])
        # 稀に message.content に積まれるケース
        msg = choice.get("message", {})
        if isinstance(msg, dict) and "content" in msg and msg["content"] is not None:
            return str(msg["content"])
    except Exception:
        pass
    return ""

# -------------------------------
# 本体処理
# -------------------------------

def stream_aggregated_output(task: str, chunk_size: int = 8, max_tokens: int = 1024, temperature: float = 0.7) -> None:
    """
    LLM に task を与え、stream=True で受け取りながら、チャンクを chunk_size 個ごとに束ねて出力。
    出力は逐次 print されます。
    """
    messages = format_messages(task)

    # llama_cpp の chat completion をストリームで呼び出し
    stream = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,  # ★ 重要：ストリーミング有効化
    )

    buf: List[str] = []
    emitted_blocks = 0

    for chunk in stream:
        text = extract_stream_text(chunk)
        if not text:
            # finish_reason 等で空が来ることがある
            continue
        buf.append(text)
        if len(buf) >= chunk_size:
            block = "".join(buf)
            emitted_blocks += 1
            print(block, end="", flush=True)
            print("")
            # 行区切りを入れたい場合は下記のように変更
            # print("\n---\n", flush=True)
            buf.clear()

    # 端数が残っていれば最後に出力
    if buf:
        print("".join(buf), end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM のストリーム出力を 8 トークン(既定)ごとに束ねて表示")
    parser.add_argument("--task", type=str, default=None, help="LLM に与えるタスク（一文で OK）")
    parser.add_argument("--chunk_size", type=int, default=8, help="束ねるチャンク数（≒トークン数）")
    parser.add_argument("--max_tokens", type=int, default=1024, help="最大生成トークン数")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度")
    args = parser.parse_args()

    task = args.task
    if not task:
        # デフォルトの簡単タスク（サンプル）
        task = (
            "Question: This question refers to the following information.\nAlthough in Protestant Europe, [Peter the Great] was surrounded by evidence of the new civil and political rights of individual men embodied in constitutions, bills of rights and parliaments, he did not return to Russia determined to share power with his people. On the contrary, he returned not only determined to change his country but also convinced that if Russia was to be transformed, it was he who must provide both the direction and the motive force. He would try to lead; but where education and persuasion were not enough, he could drive—and if necessary flog—the backward nation forward.\n—Robert K. Massie, Peter the Great: His Life and World\nWhen Peter the Great ruled Russia, he continued the practice of which of the following?\nA: Decentralization of power\nB: Isolationism\nC: Serfdom\nD: Reform\n\nAnswer the question by choosing A, B, C, or D and explain your reasoning briefly."
        )

    # ログディレクトリ（任意）
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).resolve().parent / "logs"
    (log_dir / f"stream_{run_ts}").mkdir(parents=True, exist_ok=True)

    stream_aggregated_output(
        task=task,
        chunk_size=args.chunk_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl+C で即時終了
        sys.exit(130)
