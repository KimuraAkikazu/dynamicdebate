"""
MMLU の問題を順次読み込み、エージェントにディベートさせた最終回答を
正解ラベルと照合して精度を算出するスクリプト
"""
from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import yaml
from datasets import load_dataset

from src.agent import Agent
from src.llm_handler import LLMHandler
from src.manager import DiscussionManager
from src.prompt_logger import PromptLogger

LABELS = ["A", "B", "C", "D", "E", "F"]  # 最大 6 択


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


def main() -> None:
    # ---------- 設定 ----------
    base_cfg = load_config()

    # ---------- 実行 ID 直下フォルダ ----------
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(__file__).resolve().parent / "logs" / f"run_{run_ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    # ---------- LLM (logger なしで一度だけロード) ----------
    llm_handler = LLMHandler(base_cfg["llm"], prompt_logger=None)

    # ---------- データセット ----------
    ds = load_dataset("cais/mmlu", "all", split="test")
    NUM_QUESTIONS = 50  # デモ

    correct = 0
    for idx, ex in enumerate(ds, 1):
        if idx > NUM_QUESTIONS:
            break

        # ----- 問題フォルダ -----
        prob_dir = run_root / f"problem_{idx:03d}"
        prob_dir.mkdir(parents=True, exist_ok=True)

        # ----- PromptLogger を問題フォルダに作成 -----
        prompt_logger = PromptLogger(prob_dir)
        llm_handler.logger = prompt_logger  # シングルトンに紐付け

        # ----- トピック -----
        topic = format_topic(ex["question"], ex["choices"])

        # ----- config コピーしてトピック差し替え -----
        cfg = copy.deepcopy(base_cfg)
        cfg["discussion"]["topic"] = topic

        # ----- エージェント生成 -----
        agents = [Agent(a["name"], a["persona"], llm_handler) for a in cfg["agents"]]

        # ----- ディベート実行（logs は prob_dir 内） -----
        manager = DiscussionManager(agents, cfg, log_dir=prob_dir)
        final = manager.run_discussion()  # {agent: {...}}

        # ----- 多数決予測 -----
        preds = [ans.get("answer", "").strip().upper() for ans in final.values()]
        pred_label = majority_vote(preds)

        # ----- ゴールド -----
        gold_label = idx_to_label(ex["answer"])

        # ----- 精度集計 -----
        if pred_label == gold_label:
            correct += 1

        print(
            f"[Q{idx:03}] Pred={pred_label} | Gold={gold_label} | "
            f"{'✅ 正解' if pred_label == gold_label else '❌ 不正解'}"
        )

    print(
        f"\nAccuracy: {correct}/{NUM_QUESTIONS} = "
        f"{correct / NUM_QUESTIONS:.2%}"
    )


if __name__ == "__main__":
    main()
