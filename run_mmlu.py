"""
MMLU の問題をランダムに抽出した N 問について、
エージェントにディベートさせた最終回答を正解ラベルと照合して精度を算出しつつ、
各問題の正誤と最終 Accuracy を JSON Lines で記録するスクリプト
"""
from __future__ import annotations

import copy
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import yaml
from datasets import load_dataset

from src.agent import Agent
from src.llm_handler import LLMHandler
from src.manager import DiscussionManager
from src.prompt_logger import PromptLogger

LABELS: List[str] = ["A", "B", "C", "D", "E", "F"]  # 最大 6 択
SEED = 42  # 再現性のための乱数シード


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


def choose_adversary_target(gold_label: str | None) -> str:
    """A-D の範囲で gold と異なるラベルをランダムに選ぶ（gold が A-D 以外なら単純ランダム）"""
    pool = ["A", "B", "C", "D"]
    rnd = random.Random(SEED)
    if gold_label in pool:
        pool = [p for p in pool if p != gold_label]
    return rnd.choice(pool)


# ---------- メイン ---------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Run MMLU debate evaluation")
    parser.add_argument(
        "--num",
        type=int,
        default=50,
        help="Number of questions to sample (default: 50, use -1 for all)",
    )
    args = parser.parse_args()

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
    total_available = len(ds)

    # 使用する問題数を決定
    if args.num < 0:
        total = total_available
    else:
        total = min(args.num, total_available)

    # ランダムにシャッフルして total 問を抽出
    indices = list(range(total_available))
    rnd = random.Random(SEED)
    rnd.shuffle(indices)
    selected = indices[:total]

    # ---- adversary 設定 ----
    adv_cfg = base_cfg.get("adversary", {}) or {}
    adv_enabled: bool = bool(adv_cfg.get("enabled", False))
    adv_agent_name: str | None = adv_cfg.get("agent_name")
    adv_strategy: str = str(adv_cfg.get("target_strategy", "random_wrong"))
    adv_fixed_label: str | None = adv_cfg.get("fixed_label")

    correct = 0
    for run_id, ds_idx in enumerate(selected, start=1):
        ex = ds[ds_idx]

        # ---- 問題フォルダ ----
        prob_dir = run_root / f"problem_{run_id:03d}"
        prob_dir.mkdir(parents=True, exist_ok=True)

        # ---- PromptLogger ----
        prompt_logger = PromptLogger(prob_dir)
        llm_handler.logger = prompt_logger  # シングルトンに紐付け

        # ---- トピック ----
        topic = format_topic(ex["question"], ex["choices"])
        gold_label = idx_to_label(ex["answer"])  # E, F もあり得る

        # ---- config 差し替え ----
        cfg = copy.deepcopy(base_cfg)
        cfg["discussion"]["topic"] = topic

        # ---- エージェント生成 ----
        agents = [Agent(a["name"], a["persona"], llm_handler) for a in cfg["agents"]]

        # ---- adversary 指定（1 体） ----
        if adv_enabled and agents:
            # 対象エージェントを決定
            if adv_agent_name:
                target_agent = next((ag for ag in agents if ag.name == adv_agent_name), agents[-1])
            else:
                target_agent = agents[-1]  # デフォルトは最後のエージェント

            # ターゲットラベルを決定
            if adv_strategy == "fixed" and isinstance(adv_fixed_label, str):
                adv_target = adv_fixed_label.strip().upper()
                if adv_target not in {"A", "B", "C", "D"}:
                    adv_target = choose_adversary_target(gold_label)
            else:
                adv_target = choose_adversary_target(gold_label)

            target_agent.set_adversary(adv_target)

            # メタ情報を保存
            with open(prob_dir / "adversary_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "adversary_agent": target_agent.name,
                        "target_label": adv_target,
                        "gold_label": gold_label,
                        "strategy": adv_strategy,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        # ---- ディベート実行 ----
        manager = DiscussionManager(agents, cfg, log_dir=prob_dir)
        final = manager.run_discussion()

        # ---- 予測 ----
        preds = [ans.get("answer", "").strip().upper() for ans in final.values()]
        pred_label = majority_vote(preds)
        gold_label_ABCD = gold_label if gold_label in {"A", "B", "C", "D"} else gold_label  # そのまま
        is_correct = pred_label == gold_label_ABCD
        if is_correct:
            correct += 1

        # ---- コンソール表示 ----
        print(
            f"[Q{run_id:03}] (idx={ds_idx}) Pred={pred_label} | Gold={gold_label_ABCD} | "
            f"{'✅ 正解' if is_correct else '❌ 不正解'}"
        )

        # ---- 結果を JSON Lines に追記 ----
        result_fp.write(
            json.dumps(
                {
                    "question_id": run_id,
                    "index_in_split": ds_idx,
                    "pred": pred_label,
                    "gold": gold_label_ABCD,
                    "correct": is_correct,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        result_fp.flush()

    # ---- 最終 Accuracy ----
    accuracy = correct / total if total else 0.0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")

    # ---- Accuracy を記録ファイルに追記 ----
    result_fp.write(
        json.dumps(
            {
                "summary": "accuracy",
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "seed": SEED,
                "sampled": True,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    result_fp.close()


if __name__ == "__main__":
    main()
