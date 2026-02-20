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
from typing import List, Tuple, Dict, Any

import yaml
from datasets import load_dataset

from src.agent import Agent
from src.llm_handler import LLMHandler
from src.manager import DiscussionManager
from src.prompt_logger import PromptLogger

LABELS: List[str] = ["A", "B", "C", "D", "E", "F"]  # 最大 6 択
SEED = 42  # 再現性のための乱数シード
INITIAL_POOL_PATH = (
    Path(__file__).resolve().parent
    / "Initial_answer"
    / "initial_pool_20251219_152013"
    / "initial_pool.jsonl"
)


# ---------- ユーティリティ ---------- #
def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_topic(question: str, choices: list[str]) -> str:
    lines = [question] + [f"{LABELS[i]}: {ch}" for i, ch in enumerate(choices)]
    return "\n".join(lines)


def majority_vote(ans_list: list[str]) -> str:
    if not ans_list:
        return ""
    return max(set(ans_list), key=ans_list.count)


def idx_to_label(idx: int | str) -> str:
    try:
        return LABELS[int(idx)]
    except (ValueError, TypeError, IndexError):
        return str(idx).strip().upper()


def choose_adversary_target(gold_label: str | None) -> str:
    """A-D の範囲で gold と異なるラベルをランダムに選ぶ（gold が A-D 以外なら単純ランダム）"""
    pool = ["A", "B", "C", "D"]
    # 修正: 以前の rnd = random.Random(SEED) を削除し、グローバルの random を使用
    if gold_label in pool:
        pool = [p for p in pool if p != gold_label]
    return random.choice(pool)


def load_initial_pool(pool_path: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    pool_by_index: Dict[int, Dict[str, Any]] = {}
    pool_by_question: Dict[str, Dict[str, Any]] = {}
    if not pool_path.exists():
        print(f"[Warn] initial pool file not found: {pool_path}")
        return pool_by_index, pool_by_question

    with open(pool_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            index_in_split = entry.get("index_in_split")
            question = entry.get("question")
            if isinstance(index_in_split, int):
                if index_in_split not in pool_by_index:
                    pool_by_index[index_in_split] = entry
                else:
                    print(f"[Warn] Duplicate index_in_split at line {line_no}: {index_in_split}")
            if isinstance(question, str) and question:
                if question not in pool_by_question:
                    pool_by_question[question] = entry
                else:
                    print(f"[Warn] Duplicate question at line {line_no}")
    return pool_by_index, pool_by_question


# ---------- メイン ---------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Run MMLU debate evaluation")
    parser.add_argument(
        "--num",
        type=int,
        default=500,
        help="Number of questions to sample (default: 50, use -1 for all)",
    )
    parser.add_argument(
        "--order",
        choices=["sequential", "random"],
        default="sequential",
        help="Question order (default: sequential)",
    )
    args = parser.parse_args()

    # 修正: ここで乱数シードを固定し、以降はグローバルの random を使い回す
    random.seed(SEED)

    base_cfg = load_config()
    pool_by_index, pool_by_question = load_initial_pool(INITIAL_POOL_PATH)

    # 実行フォルダ作成
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(__file__).resolve().parent / "logs" / f"run_{run_ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    # 結果を書き込む JSONL ファイル
    result_file = run_root / "accuracy_log.jsonl"
    result_fp = result_file.open("w", encoding="utf-8")

    # LLM は 1 度だけロード
    llm_handler = LLMHandler(base_cfg["llm"], prompt_logger=None)

    # ---- adversary 設定 (Base) ----
    adv_cfg = base_cfg.get("adversary", {}) or {}
    adv_enabled: bool = bool(adv_cfg.get("enabled", False))
    
    # 複数エージェント対応: agent_names を取得
    adv_agent_names: List[str] = adv_cfg.get("agent_names", [])
    if not adv_agent_names and "agent_name" in adv_cfg:
        # 旧互換
        val = adv_cfg.get("agent_name")
        if val:
            adv_agent_names = [val]
            
    adv_strategy: str = str(adv_cfg.get("target_strategy", "random_wrong"))
    adv_fixed_label: str | None = adv_cfg.get("fixed_label")

    # データセット読み込み
    ds = load_dataset("cais/mmlu", "all", split="test")
    total_available = len(ds)

    # 使用する問題数を決定
    if adv_enabled and pool_by_index:
        # dict は挿入順を保持するため、initial_pool.jsonl の先頭から順に処理できる
        candidate_indices = list(pool_by_index.keys())
        if args.num < 0:
            total = len(candidate_indices)
        else:
            total = min(args.num, len(candidate_indices))
        if args.order == "random":
            random.shuffle(candidate_indices)
        selected = candidate_indices[:total]
    else:
        if adv_enabled and not pool_by_index:
            print("[Warn] Adversary is enabled but initial pool is empty. Falling back to full dataset.")
        if args.num < 0:
            total = total_available
        else:
            total = min(args.num, total_available)

        indices = list(range(total_available))
        if args.order == "random":
            # 修正: グローバルの random.shuffle を使用
            random.shuffle(indices)
        selected = indices[:total]

    correct = 0
    for run_id, ds_idx in enumerate(selected, start=1):
        ex = ds[ds_idx]

        # ---- 問題フォルダ ----
        prob_dir = run_root / f"problem_{run_id:03d}"
        prob_dir.mkdir(parents=True, exist_ok=True)

        # ---- PromptLogger ----
        prompt_logger = PromptLogger(prob_dir)
        llm_handler.logger = prompt_logger  # シングルトンに紐付け

        # ---- トピック & 正解 ----
        topic = format_topic(ex["question"], ex["choices"])
        gold_label = idx_to_label(ex["answer"])
        pool_entry = pool_by_index.get(ds_idx) or pool_by_question.get(ex["question"])
        if not pool_entry:
            print(f"[Warn] No initial pool entry found for idx={ds_idx}")
            if adv_enabled:
                # adversary 有効時は initial_pool を必須とする
                raise ValueError(
                    f"initial_pool_entry not found for idx={ds_idx}. "
                    "Use a pool that covers this question or restrict sampling to pool entries."
                )

        # ---- config の複製と動的設定 ----
        cfg = copy.deepcopy(base_cfg)
        cfg["discussion"]["topic"] = topic
        if adv_enabled:
            cfg["initial_pool_entry"] = pool_entry

        # Adversary のターゲット決定
        current_adv_target = None
        current_adversaries = []

        if adv_enabled:
            # 1. 今回のターゲット回答を決定
            if adv_strategy == "fixed" and adv_fixed_label in {"A", "B", "C", "D"}:
                current_adv_target = adv_fixed_label
            else:
                # random_wrong: Gold 以外から選択
                current_adv_target = choose_adversary_target(gold_label)
            
            # 2. Config を書き換えて Manager に渡す
            # Manager内でランダム抽選させるとGoldと被る可能性があるため、ここで固定化して渡す
            cfg["adversary"]["enabled"] = True
            cfg["adversary"]["target_strategy"] = "fixed"
            cfg["adversary"]["fixed_label"] = current_adv_target
            cfg["adversary"]["agent_names"] = adv_agent_names  # リストを渡す
            
            # メタデータ用にリスト保持
            current_adversaries = adv_agent_names

            # メタ情報を保存
            with open(prob_dir / "adversary_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "adversary_agents": current_adversaries,
                        "target_label": current_adv_target,
                        "gold_label": gold_label,
                        "strategy": adv_strategy,  # 元の設定値を記録
                        "actual_strategy_used": "fixed", # 内部的にはfixedとして動作
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        # ---- エージェント生成 ----
        # 役割設定は Manager の _initialize_discussion で config に基づき行われるため
        # ここではインスタンス化のみ行う
        agents = [Agent(a["name"], a["persona"], llm_handler) for a in cfg["agents"]]

        # ---- ディベート実行 ----
        manager = DiscussionManager(agents, cfg, log_dir=prob_dir)
        final = manager.run_discussion()

        # ---- 予測 ----
        preds = [ans.get("answer", "").strip().upper() for ans in final.values()]
        pred_label = majority_vote(preds)
        gold_label_ABCD = gold_label if gold_label in {"A", "B", "C", "D"} else gold_label
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
                    "adversary_target": current_adv_target
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
                "sampled": args.order == "random",
                "order": args.order,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    result_fp.close()


if __name__ == "__main__":
    main()
