"""
MMLU の問題を順次読み込み、エージェントにディベートさせた最終回答を
正解ラベルと照合して精度を算出しつつ、
各問題の正誤と最終 Accuracy を JSON Lines で記録するスクリプト
"""
from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import List

import yaml
from datasets import load_dataset

from src.agent import Agent
from src.llm_handler import LLMHandler
from src.manager import DiscussionManager
from src.prompt_logger import PromptLogger

LABELS: List[str] = ["A", "B", "C", "D", "E", "F"]  # 最大 6 択


# ---------- ユーティリティ ---------- #
def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_topic(question: str, choices: list[str]) -> str:
    lines = [question] + [f"{LABELS[i]}: {ch}" for i, ch in enumerate(choices)]
    return "\n".join(lines)


def majority_vote(ans_list: list[str]) -> str:
    return max(set(ans_list), key=ans_list.count)


def idx_to_label(idx: int | str) -> str:
    try:
        return LABELS[int(idx)]
    except (ValueError, TypeError, IndexError):
        return str(idx).strip().upper()


# ---------- メイン ---------- #
def main() -> None:
    base_cfg = load_config()

    # 実行フォルダ作成
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(__file__).resolve().parent / "logs" / f"run_{run_ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    # 結果を書き込む JSONL ファイル
    result_file = run_root / "accuracy_log.jsonl"
    result_fp = result_file.open("w", encoding="utf-8")

    # LLM は 1 度だけロード
    llm_handler = LLMHandler(base_cfg["llm"], prompt_logger=None)

    # データセット読み込み
    ds = load_dataset("cais/mmlu", "all", split="test")
    NUM_QUESTIONS = 50  # デモ用

    correct = 0
    for idx, ex in enumerate(ds, 1):
        if idx > NUM_QUESTIONS:
            break

        # ---- 問題フォルダ ----
        prob_dir = run_root / f"problem_{idx:03d}"
        prob_dir.mkdir(parents=True, exist_ok=True)

        # ---- PromptLogger ----
        prompt_logger = PromptLogger(prob_dir)
        llm_handler.logger = prompt_logger  # シングルトンに紐付け

        # ---- トピック ----
        topic = format_topic(ex["question"], ex["choices"])

        # ---- config 差し替え ----
        cfg = copy.deepcopy(base_cfg)
        cfg["discussion"]["topic"] = topic

        # ---- エージェント生成 ----
        agents = [Agent(a["name"], a["persona"], llm_handler) for a in cfg["agents"]]

        # ---- ディベート実行 ----
        manager = DiscussionManager(agents, cfg, log_dir=prob_dir)
        final = manager.run_discussion()

        # ---- 予測 ----
        preds = [ans.get("answer", "").strip().upper() for ans in final.values()]
        pred_label = majority_vote(preds)
        gold_label = idx_to_label(ex["answer"])

        is_correct = pred_label == gold_label
        if is_correct:
            correct += 1

        # ---- コンソール表示 ----
        print(
            f"[Q{idx:03}] Pred={pred_label} | Gold={gold_label} | "
            f"{'✅ 正解' if is_correct else '❌ 不正解'}"
        )

        # ---- 結果を JSON Lines に追記 ----
        result_fp.write(
            json.dumps(
                {
                    "question_id": idx,
                    "pred": pred_label,
                    "gold": gold_label,
                    "correct": is_correct,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        result_fp.flush()

    # ---- 最終 Accuracy ----
    accuracy = correct / NUM_QUESTIONS
    print(f"\nAccuracy: {correct}/{NUM_QUESTIONS} = {accuracy:.2%}")

    # ---- Accuracy を記録ファイルに追記 ----
    result_fp.write(
        json.dumps(
            {
                "summary": "accuracy",
                "correct": correct,
                "total": NUM_QUESTIONS,
                "accuracy": accuracy,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    result_fp.close()


if __name__ == "__main__":
    main()
