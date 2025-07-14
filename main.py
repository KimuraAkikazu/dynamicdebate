"""エントリーポイント"""
from pathlib import Path
import yaml

from src.agent import Agent
from src.llm_handler import LLMHandler
from src.manager import DiscussionManager
from src.prompt_logger import PromptLogger   # ★ 追加

def main() -> None:
    # ---------------- 設定読み込み ---------------- #
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---------------- PromptLogger ---------------- #
    logs_root = Path(__file__).resolve().parent / "logs"
    prompt_logger = PromptLogger(logs_root)          # ★ 追加

    # ---------------- LLMHandler (シングルトン) ---------------- #
    llm_handler = LLMHandler(config["llm"], prompt_logger=prompt_logger)  # ★ 修正

    # ---------------- Agent インスタンス生成 ---------------- #
    agents = [
        Agent(agent_cfg["name"], agent_cfg["persona"], llm_handler)
        for agent_cfg in config["agents"]
    ]

    # ---------------- DiscussionManager 実行 ---------------- #
    manager = DiscussionManager(agents, config)
    manager.run_discussion()

if __name__ == "__main__":
    main()
