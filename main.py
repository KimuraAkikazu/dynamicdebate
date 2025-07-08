"""エントリーポイント"""
from pathlib import Path
import yaml

from src.agent import Agent
from src.llm_handler import LLMHandler
from src.manager import DiscussionManager


def main() -> None:
    # ---------------- 設定読み込み ---------------- #
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---------------- LLMHandler (シングルトン) ---------------- #
    llm_handler = LLMHandler(config["llm"])

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