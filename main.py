"""エントリーポイント"""
from pathlib import Path
from datetime import datetime
import yaml

from src.agent import Agent
from src.llm_handler import LLMHandler
from src.manager import DiscussionManager
from src.prompt_logger import PromptLogger


def main() -> None:
    # ---------------- 設定読み込み ---------------- #
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---------------- 実行用ログディレクトリ作成 ---------------- #
    logs_root = Path(__file__).resolve().parent / "logs"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exec_dir = logs_root / f"run_{run_id}"
    exec_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- PromptLogger ---------------- #
    prompt_logger = PromptLogger(exec_dir)  # フォルダを直接渡す

    # ---------------- LLMHandler (シングルトン) ---------------- #
    llm_handler = LLMHandler(config["llm"], prompt_logger=prompt_logger)

    # ---------------- Agent インスタンス生成 ---------------- #
    agents = [
        Agent(agent_cfg["name"], agent_cfg["persona"], llm_handler)
        for agent_cfg in config["agents"]
    ]

    # ---------------- DiscussionManager 実行 ---------------- #
    manager = DiscussionManager(agents, config, log_dir=exec_dir)
    manager.run_discussion()


if __name__ == "__main__":
    main()
