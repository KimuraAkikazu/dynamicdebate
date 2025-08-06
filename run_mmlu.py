"""
MMLU の問題を順次読み込み、エージェントにディベートさせた最終回答を
正解ラベルと照合して精度を算出するスクリプト
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml
from datasets import load_dataset

from src.agent import Agent
from src.llm_handler import LLMHandler
from src.manager import DiscussionManager

LABELS = ["A", "B", "C", "D", "E", "F"]  # 最大 6 択想定


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_topic(question: str, choices: list[str]) -> str:
    lines = [question] + [f"{LABELS[i]}: {ch}" for i, ch in enumerate(choices)]
    # lines.append("回答はA〜Dの中から1つ選んでください。")
    return "\n".join(lines)


def majority_vote(answers: list[str]) -> str:
    return max(set(answers), key=answers.count)


def idx_to_label(idx: int | str) -> str:
    """0/1/2/3 → A/B/C/D に変換（すでに文字列ならそのまま）"""
    try:
        return LABELS[int(idx)]
    except (ValueError, TypeError, IndexError):
        # 既に "A" などの文字列 or 範囲外 -> そのまま返す
        return str(idx).strip().upper()


def main() -> None:
    # ---------- 設定読み込み ----------
    base_cfg = load_config()

    # ---------- モデルを一度だけロード ----------
    llm_handler = LLMHandler(base_cfg["llm"])

    # ---------- データセット ----------
    ds = load_dataset("cais/mmlu", "all", split="test")

    NUM_QUESTIONS = 50  # デモ用
    correct = 0

    for idx, example in enumerate(ds):
        if idx >= NUM_QUESTIONS:
            break

        # --- 問題文をトピック化 ---
        topic = format_topic(example["question"], example["choices"])

        # --- config をコピーしてトピック差し替え ---
        cfg = copy.deepcopy(base_cfg)
        cfg["discussion"]["topic"] = topic
        # print(f"\n[Q{idx:02}] トピック: {topic}")

        # --- エージェント準備（同じ LLMHandler を共有） ---
        agents = [Agent(a["name"], a["persona"], llm_handler) for a in cfg["agents"]]

        # --- ディベート実行 ---
        manager = DiscussionManager(agents, cfg)
        final_answers = manager.run_discussion()  # {agent: {"answer": str, ...}}

        # --- 予測ラベルを決定（多数決） ---
        preds = [ans.get("answer", "").strip().upper() for ans in final_answers.values()]
        pred_label = majority_vote(preds)

        # --- ゴールドラベルを A/B/C/D へ変換 ---
        gold_label = idx_to_label(example["answer"])

        # --- 判定 ---
        is_correct = pred_label == gold_label
        if is_correct:
            correct += 1

        print(f"[Q{idx:02}] Pred={pred_label} | Gold={gold_label} | {'✅' if is_correct else '❌'}")

    print(f"\nAccuracy: {correct}/{NUM_QUESTIONS} = {correct / NUM_QUESTIONS:.2%}")


if __name__ == "__main__":
    main()
