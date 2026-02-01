"""Async entrypoint for streaming debate with barge-in support."""
from pathlib import Path
from datetime import datetime
import yaml

from src.agent import Agent
from src.llm_handler import LLMHandler
from src.manager import DiscussionManager
from src.prompt_logger import PromptLogger


def main() -> None:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logs_root = Path(__file__).resolve().parent / "logs"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exec_dir = logs_root / f"run_async_{run_id}"
    exec_dir.mkdir(parents=True, exist_ok=True)

    prompt_logger = PromptLogger(exec_dir)
    llm_handler = LLMHandler(config["llm"], prompt_logger=prompt_logger)
    agents = [
        Agent(agent_cfg["name"], agent_cfg["persona"], llm_handler)
        for agent_cfg in config["agents"]
    ]

    manager = DiscussionManager(agents, config, log_dir=exec_dir)
    manager.run_debate_async()


if __name__ == "__main__":
    main()
