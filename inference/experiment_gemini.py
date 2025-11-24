import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

import hydra
from omegaconf import DictConfig, OmegaConf

from inference.llm.llm_clients import GeminiClient
from inference.star_making.assets_utils import LEARNING_ORDERS, StarMakingRules
from inference.star_making.env_batch_text import StarMakingEnvBatchText

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _build_goal_and_rule_schedule(n_trials: int) -> tuple[list[str], list[str]]:
    """Create aligned goal-star and rule schedules."""
    learning_trials = n_trials // 2
    transfer_trials = n_trials - learning_trials
    block_size = max(1, n_trials // 16)
    transfer_labels = ["Star_1" if i % 2 == 0 else "Star_2" for i in range(8)]

    def assign_goals(total, size, labels):
        goals = []
        for label in labels:
            if len(goals) >= total:
                break
            goals.extend([label] * min(size, total - len(goals)))
        if len(goals) < total:
            goals.extend([goals[-1] if goals else labels[0]] * (total - len(goals)))
        return goals

    learning_goals = assign_goals(learning_trials, block_size, LEARNING_ORDERS)
    transfer_goals = assign_goals(transfer_trials, block_size, transfer_labels)
    return learning_goals + transfer_goals, ["learning"] * len(learning_goals) + ["transfer"] * len(transfer_goals)


def _run_encoding_group(
    participant_ids: Sequence[int],
    encoding: int,
    cfg: DictConfig,
    llm_client: GeminiClient,
    goal_schedule: list[str],
    rule_schedule: list[str],
) -> list[dict]:
    """Process all participants for a single action encoding."""
    rules = StarMakingRules(encoding=encoding)
    rules.set_transfer_rule_type(cfg.experiment.transfer_rule_type)

    n_agents = cfg.experiment.get("n_agents", 4)
    results = []

    for i in range(0, len(participant_ids), n_agents):
        batch_ids = participant_ids[i:i + n_agents]
        logger.info(f"Encoding {encoding} | Participants {batch_ids[0]}-{batch_ids[-1]}")

        env = StarMakingEnvBatchText(llm_client=llm_client, rules=rules, n_agents=len(batch_ids), rule_type="learning")
        batch_results = env.run_experiment_batch(
            n_trials=cfg.experiment.n_trials,
            goal_stars_per_agent=[list(goal_schedule) for _ in batch_ids],
            rule_schedule=rule_schedule,
        )

        for idx, agent_data in enumerate(batch_results["agents"]):
            agent_data["agent_id"] = batch_ids[idx]
            agent_data["encoding"] = encoding
        results.extend(batch_results["agents"])

    return results


@hydra.main(version_base=None, config_path="../config", config_name="experiment")
def main(cfg: DictConfig) -> None:
    """Run experiment with Gemini LLM."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    model_name = cfg.model.get("gemini_model", "gemini-2.5-flash")
    thinking_budget = cfg.model.get("thinking_budget", 256)
    logger.info(f"Initializing Gemini client with model: {model_name}, thinking_budget: {thinking_budget}")
    llm_client = GeminiClient(api_key=api_key, model=model_name, thinking_budget=thinking_budget)

    n_trials = cfg.experiment.n_trials
    num_participants = cfg.experiment.num_participants
    logger.info(f"Running {num_participants} participants, {n_trials} trials each")

    goal_schedule, rule_schedule = _build_goal_and_rule_schedule(n_trials)

    # Split participants between encodings
    participant_ids = list(range(num_participants))
    split_idx = num_participants // 2
    encoding_groups = [(participant_ids[:split_idx], 1), (participant_ids[split_idx:], 2)]

    all_agents = []
    for ids, encoding in encoding_groups:
        all_agents.extend(_run_encoding_group(ids, encoding, cfg, llm_client, goal_schedule, rule_schedule))

    # Build results
    participants = []
    for agent_data in sorted(all_agents, key=lambda x: x["agent_id"]):
        trials = agent_data["trials"]
        successes = sum(1 for t in trials if t["success"])
        participants.append({
            "participant_id": agent_data["agent_id"],
            "encoding": agent_data["encoding"],
            "conversation_history": agent_data["conversation_history"],
            "trials": trials,
            "summary": {"total_trials": len(trials), "successful_trials": successes, "success_rate": successes / len(trials) if trials else 0.0},
        })
        logger.info(f"Participant {agent_data['agent_id']} (enc {agent_data['encoding']}): {successes}/{len(trials)}")

    # Save results
    if cfg.output.save_json:
        output_dir = Path(cfg.output.dir) / f"gemini-{model_name}" / cfg.experiment.transfer_rule_type
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_experiment.json"
        with open(output_path, "w") as f:
            json.dump({
                "config": OmegaConf.to_container(cfg, resolve=True),
                "timestamp": datetime.now().isoformat(),
                "participants": participants,
                "experiment_schedule": {"goal_stars": goal_schedule, "rule_schedule": rule_schedule, "transfer_rule_type": cfg.experiment.transfer_rule_type},
            }, f, indent=2)
        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
