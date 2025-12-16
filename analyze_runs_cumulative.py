#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
複数の run ログを比較し、「累積コスト（トークン数）」に対する Accuracy の推移を可視化するスクリプト。
prompt_log が存在する場合はそこからトークン数を推定・集計します。
"""

import argparse
import json
import os
import re
import math
from glob import glob
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- トークン計算用ライブラリ ---
try:
    import tiktoken
    ENCODING = tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("[INFO] tiktoken not found. Using character-length approximation.")

# ------------- ユーティリティ ------------- #

def count_tokens(text: str) -> int:
    """テキストのトークン数を計算（tiktoken または 簡易推定）"""
    if not text:
        return 0
    if HAS_TIKTOKEN:
        return len(ENCODING.encode(text))
    # 簡易推定: 英語メインなら文字数/4
    return math.ceil(len(text) / 4)

def resolve_run_dir(run_arg: str) -> str:
    if os.path.isdir(run_arg):
        return os.path.abspath(run_arg)
    candidate = os.path.join("logs", run_arg)
    if os.path.isdir(candidate):
        return os.path.abspath(candidate)
    raise FileNotFoundError(f"Run directory not found: {run_arg} or logs/{run_arg}")

def majority_vote(answers: List[str]) -> Optional[str]:
    filtered = [a for a in answers if a]
    if not filtered: return None
    counts = Counter(filtered)
    most = counts.most_common()
    if len(most) == 1: return most[0][0]
    if most[0][1] == most[1][1]: return None
    return most[0][0]

# ------------- データ読み込み ------------- #

def load_accuracy(run_dir: str) -> Dict[str, Dict[str, Any]]:
    candidates = [
        os.path.join(run_dir, "accuracy_log.jsonl"),
        os.path.join(run_dir, "accuracy_log.json"),
    ]
    acc_path = next((p for p in candidates if os.path.isfile(p)), None)
    if not acc_path:
        return {}

    data = []
    if acc_path.endswith(".jsonl"):
        with open(acc_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
    else:
        with open(acc_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                for k, v in loaded.items():
                    v["question_id"] = k
                    data.append(v)
            else:
                data = loaded

    acc_by_pid = {}
    for row in data:
        raw_id = row.get("question_id") or row.get("id") or row.get("problem_id")
        if raw_id is None: continue
        
        idx = int(str(raw_id).replace("problem_", ""))
        pid = f"problem_{idx:03d}"
        gold = row.get("gold") or row.get("gold_answer") or row.get("label")
        acc_by_pid[pid] = {"gold": gold, "index": idx}
    
    return acc_by_pid

def estimate_tokens_from_prompt_log(problem_dir: str) -> Dict[int, int]:
    """
    指定ディレクトリ内の prompt_log_*.jsonl を読み込み、
    {turn_id: このターンで発生した総トークン数(入力+出力)} の辞書を返す。
    """
    token_usage_by_turn = defaultdict(int)
    
    # ファイルを探す (prompt_log_*.jsonl)
    logs = glob(os.path.join(problem_dir, "prompt_log_*.jsonl"))
    if not logs:
        return {}
    
    # 複数ある場合は最新を使うなどの戦略があるが、通常1つと仮定
    # 全部読んで合算する（リトライなどで複数ファイルに分かれるケースも考慮）
    for log_file in logs:
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        entry = json.loads(line)
                    except: continue
                    
                    turn = entry.get("turn")
                    if turn is None: continue
                    
                    # turn が "final" などの文字列の場合の対応（必要なら）
                    # ここでは整数ターンのみを集計対象とする
                    if isinstance(turn, str) and not turn.isdigit():
                        continue
                    turn = int(turn)

                    phase = entry.get("phase", "")
                    cost = 0
                    
                    # # Prompt (入力)
                    # if "generated" not in phase:
                    #     sys = entry.get("system", "")
                    #     usr = entry.get("user", "")
                    #     cost += count_tokens((sys or "") + "\n" + (usr or ""))
                    
                    # Completion (出力)
                    if "_generated" in phase:
                        if "speaker_utterance_generated" not in phase and "final" not in phase and "initial" not in phase:
                            content = entry.get("content", "")
                            cost += count_tokens(content)
                    
                    token_usage_by_turn[turn] += cost
        except Exception as e:
            print(f"[WARN] Failed to parse {log_file}: {e}")
            
    return token_usage_by_turn

def load_discussion_details(run_dir: str) -> Dict[str, Any]:
    problems = {}
    prob_dirs = sorted(glob(os.path.join(run_dir, "problem_*")))
    
    for pdir in prob_dirs:
        pid = os.path.basename(pdir)
        log_path = os.path.join(pdir, "discussion_log.json")
        if not os.path.isfile(log_path): continue

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except:
            continue
        if not logs: continue

        # --- トークン情報の取得戦略 ---
        # 1. discussion_log 内に cumulative_token_usage があればそれを使う (最新版)
        # 2. なければ prompt_log をパースして集計する (過去データ)
        # 3. それもなければ文字数ベースで近似する
        
        has_embedded_tokens = any("cumulative_token_usage" in entry for entry in logs)
        token_usage_by_turn_from_log = {}
        
        if not has_embedded_tokens:
            token_usage_by_turn_from_log = estimate_tokens_from_prompt_log(pdir)

        # 初期状態 (Turn 0)
        init_log = next((l for l in logs if l.get("turn") == 0), None)
        current_answers = {}
        initial_answers = {}
        
        if init_log and "initial_answers" in init_log:
            for ag, val in init_log["initial_answers"].items():
                if isinstance(val, dict): ans = val.get("answer", "")
                elif isinstance(val, str): ans = val
                else: ans = ""
                if isinstance(ans, str) and ans.strip():
                    current_answers[ag] = ans.strip()
                    initial_answers[ag] = ans.strip()
        
        # 時系列データの構築
        events = []
        cumulative_cost = 0 # 文字数ではなくコスト(トークン)として扱う
        
        # Turn 0 のコスト加算 (初期回答生成など)
        if has_embedded_tokens and init_log:
            usage = init_log.get("cumulative_token_usage", {})
            cumulative_cost = usage.get("total_tokens", 0)
        elif token_usage_by_turn_from_log:
            cumulative_cost += token_usage_by_turn_from_log.get(0, 0)
        
        events.append((cumulative_cost, current_answers.copy()))

        sorted_logs = sorted(logs, key=lambda x: int(x.get("turn")) if isinstance(x.get("turn"), (int,str)) and str(x.get("turn")).isdigit() else -1)

        for entry in sorted_logs:
            turn_raw = entry.get("turn")
            if turn_raw == 0 or turn_raw == "final": continue
            
            try:
                turn = int(turn_raw)
            except:
                continue

            # --- コスト加算 ---
            step_cost = 0
            
            if has_embedded_tokens:
                # ログに記録された累積値があればそのまま採用（上書き）
                usage = entry.get("cumulative_token_usage")
                if usage:
                    # ログの値は「その時点までの累積」なので、cumulative_cost を更新
                    cumulative_cost = usage.get("total_tokens", cumulative_cost)
            elif token_usage_by_turn_from_log:
                # prompt_log から集計した「このターンのコスト」を加算
                # discussion_log は同じターンで複数回記録されることがあるため（発話フェーズ、計画フェーズ等）、
                # 二重加算しないよう注意が必要。
                # ここでは簡易的に「各ターンの最後のエントリでそのターンの全コストを加算」する、
                # あるいは「discussion_logのエントリごとに按分」は難しいので、
                # 単純化して「discussion_logのターンが変わるたび」に加算するロジックにするのが安全。
                # しかしループ構造上、同じターンのエントリが続く。
                # → 「このターンをまだ処理していなければ加算」フラグで管理するか、
                # prompt_logの集計は「ターンごと」なので、ここで加算する。
                # ただ、discussion_logの1エントリに対してprompt_logの全量を足すと多すぎる。
                # よって、「このターンの最初のイベント」でのみ加算する等の制御が必要。
                pass # 下記で処理
            else:
                # 簡易推定 (文字数ベース)
                content = entry.get("content", "") or ""
                step_cost = count_tokens(content) # 入力プロンプト分が含まれないので過小評価になる点に注意
                cumulative_cost += step_cost

            # prompt_log利用時の加算制御:
            # discussion_log上で "turn X" のレコードが複数ある場合、重複して足さないようにする。
            # ここでは「agent_actions」が含まれるレコード（＝ターンの終わり際のレコード）で加算するルールにする。
            if not has_embedded_tokens and token_usage_by_turn_from_log:
                # このレコードがそのターンの「締め」に近いか判定（agent_actionsがある、または発話がある）
                # 簡易的に、このターンで発生したコストを、そのターンのレコード数で割る...のは変。
                # 「イベントが発生した時点」でコストが増えたとみなす。
                # discussion_logの構造上、Turn X は通常1つか2つのレコード。
                # ここでは「レコードが来るたびにコストを加算せず、
                # 別途計算した『Turn X終了時の総コスト』を累積値として採用する」のが正確。
                
                # Turn X までの累積コストを計算
                # (Turn 0 ~ Turn X の合計)
                current_total = sum(token_usage_by_turn_from_log.get(t, 0) for t in range(turn + 1))
                cumulative_cost = current_total

            # 2. 回答状況の更新
            new_states = entry.get("agent_states")
            if new_states and isinstance(new_states, list):
                for st in new_states:
                    if not isinstance(st, dict): continue
                    ag = st.get("agent_name")
                    ans = st.get("current_answer")
                    if ag and isinstance(ans, str) and ans.strip():
                        current_answers[ag] = ans.strip()
            else:
                cons_state = entry.get("consensus_state")
                if cons_state and isinstance(cons_state, dict):
                    for ag, info in cons_state.items():
                        ans = ""
                        if isinstance(info, dict): ans = info.get("answer", "")
                        elif isinstance(info, str): ans = info
                        if isinstance(ans, str) and ans.strip():
                            current_answers[ag] = ans.strip()
            
            events.append((cumulative_cost, current_answers.copy()))
        
        problems[pid] = {
            "initial_answers": initial_answers,
            "timeline": events, 
            "final_cost": cumulative_cost
        }
    
    return problems

# ------------- 分析・集計 ------------- #

def filter_scenarios(problems: Dict, acc_data: Dict) -> List[str]:
    targets = []
    for pid, pdata in problems.items():
        if pid not in acc_data: continue
        gold = acc_data[pid]["gold"]
        initial = pdata["initial_answers"]
        if not gold or not initial: continue
        
        correct_count = sum(1 for a in initial.values() if a == gold)
        total = len(initial)
        
        if total == 3 and correct_count == 2:
            targets.append(pid)
    return targets

def compute_binned_accuracy(
    target_pids: List[str], 
    problems: Dict, 
    acc_data: Dict, 
    bin_size: int = 1000,
    max_cost: int = 30000
) -> Tuple[List[int], List[float]]:
    
    bins = np.arange(0, max_cost + bin_size, bin_size)
    bin_corrects = defaultdict(int)
    bin_counts = defaultdict(int)
    
    for pid in target_pids:
        timeline = problems[pid]["timeline"]
        gold = acc_data[pid]["gold"]
        
        current_idx = 0
        
        for t in bins:
            while current_idx < len(timeline) - 1:
                next_time = timeline[current_idx+1][0]
                if next_time <= t:
                    current_idx += 1
                else:
                    break
            
            current_ans_dict = timeline[current_idx][1]
            maj = majority_vote(list(current_ans_dict.values()))
            is_correct = 1 if maj == gold else 0
            
            bin_corrects[t] += is_correct
            bin_counts[t] += 1
            
    x_vals = []
    y_vals = []
    
    for t in bins:
        c = bin_counts[t]
        if c == 0: continue
        x_vals.append(t)
        y_vals.append(bin_corrects[t] / c)
        
    return x_vals, y_vals

# ------------- プロット ------------- #

def plot_cumulative_comparison(data_list: List[Dict], out_path: str, bin_size: int, has_tiktoken: bool):
    plt.figure(figsize=(10, 6))
    
    markers = ["o", "s", "^", "D"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    for i, d in enumerate(data_list):
        x = d["x"]
        y = d["y"]
        label = d["label"]
        plt.plot(x, y, marker=markers[i%len(markers)], label=label, 
                 color=colors[i%len(colors)], markersize=4, alpha=0.8, linestyle="-")

    unit = "Tokens (Estimated)" if has_tiktoken else "Approx. Cost (Tokens)"
    plt.xlabel(f"Cumulative {unit}")
    plt.ylabel("Accuracy (Majority Vote)")
    plt.title(f"Accuracy vs Cumulative Token Cost (Bin: {bin_size})\nTarget: 2 correct, 1 wrong initial state")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[INFO] Saved plot: {out_path}")
    plt.close()

# ------------- メイン ------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Run directories")
    parser.add_argument("--out-dir", default="analysis_outputs_cumulative")
    # トークン数は文字数より少ないため、bin-sizeのデフォルトを調整
    parser.add_argument("--bin-size", type=int, default=500, help="Bin size for token count (default: 500)")
    parser.add_argument("--max-cost", type=int, default=15000, help="Max token count on X-axis")
    args = parser.parse_args()

    run_dirs = [resolve_run_dir(r) for r in args.runs]
    os.makedirs(args.out_dir, exist_ok=True)
    
    comp_name = "__vs__".join([os.path.basename(r.rstrip(os.sep)) for r in run_dirs])
    out_dir = os.path.join(args.out_dir, comp_name[:100])
    os.makedirs(out_dir, exist_ok=True)

    all_run_data = []
    all_scenarios = []

    for r_dir in run_dirs:
        tag = os.path.basename(r_dir.rstrip(os.sep))
        print(f"[INFO] Loading {tag} ...")
        
        acc = load_accuracy(r_dir)
        probs = load_discussion_details(r_dir) # ここで prompt_log も読まれる
        
        scenarios = filter_scenarios(probs, acc)
        print(f"  -> Found {len(scenarios)} target scenarios.")
        
        all_run_data.append({
            "tag": tag,
            "acc": acc,
            "probs": probs,
            "scenarios": set(scenarios)
        })
        all_scenarios.append(set(scenarios))

    common_scenarios = set.intersection(*all_scenarios) if all_scenarios else set()
    common_list = sorted(list(common_scenarios))
    print(f"[INFO] Common scenarios count: {len(common_list)}")
    
    if not common_list:
        print("[ERROR] No common scenarios found. Exiting.")
        return

    plot_data = []
    
    for r_data in all_run_data:
        x, y = compute_binned_accuracy(
            common_list, 
            r_data["probs"], 
            r_data["acc"], 
            bin_size=args.bin_size, 
            max_cost=args.max_cost
        )
        plot_data.append({
            "label": r_data["tag"],
            "x": x,
            "y": y
        })
        
        avg_final_cost = np.mean([r_data["probs"][pid]["final_cost"] for pid in common_list])
        print(f"  [{r_data['tag']}] Avg Final Cost: {avg_final_cost:.1f} tokens")

    out_png = os.path.join(out_dir, "cumulative_token_accuracy_comparison.png")
    plot_cumulative_comparison(plot_data, out_png, args.bin_size, HAS_TIKTOKEN)

if __name__ == "__main__":
    main()