import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from inference.llm.local_vllm_clients import LocalVLLMClient
from inference.star_making.assets_utils import LEARNING_ORDERS, StarMakingRules
from inference.star_making.env_batch_text import StarMakingEnvBatchText

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _assign_block_goals(total_trials: int, block_size: int, labels: Sequence[str]) -> List[str]:
    """Assign goal stars to blocks with graceful handling of remainders."""
    if total_trials <= 0:
        return []
    size = max(1, block_size)
    goals: List[str] = []
    for label in labels:
        if len(goals) >= total_trials:
            break
        runs = min(size, total_trials - len(goals))
        goals.extend([label] * runs)
    if len(goals) < total_trials:
        filler = goals[-1] if goals else (labels[0] if labels else "Star_0")
        goals.extend([filler] * (total_trials - len(goals)))
    return goals


def _build_goal_and_rule_schedule(n_trials: int) -> tuple[list[str], list[str]]:
    """Create aligned goal-star and rule schedules for the whole experiment."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    learning_trials = (n_trials * 2) // 3
    transfer_trials = n_trials - learning_trials
    if n_trials % 3 != 0:
        logger.warning("n_trials=%s is not divisible by 3; learning=%s, transfer=%s", n_trials, learning_trials, transfer_trials)

    block_size = n_trials // 12
    if n_trials % 12 != 0:
        logger.warning("n_trials is not divisible by 12; using block size %s per requirements", max(1, block_size))
    transfer_labels = ["Star_1" if i % 2 == 0 else "Star_2" for i in range(4)]

    learning_goals = _assign_block_goals(learning_trials, block_size, LEARNING_ORDERS)
    transfer_goals = _assign_block_goals(transfer_trials, block_size, transfer_labels)
    rule_schedule = ["learning"] * len(learning_goals) + ["transfer"] * len(transfer_goals)
    goal_schedule = learning_goals + transfer_goals
    return goal_schedule, rule_schedule


def _chunked(iterable: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for idx in range(0, len(iterable), size):
        yield iterable[idx:idx + size]


def _run_encoding_group(
    participant_ids: Sequence[int],
    encoding_label: int,
    cfg: DictConfig,
    llm_client: LocalVLLMClient,
    goal_schedule: Sequence[str],
    rule_schedule: Sequence[str],
) -> list[dict]:
    """Process all participants for a single action encoding."""
    if not participant_ids:
        return []

    rules = StarMakingRules(encoding=encoding_label)
    rules.set_transfer_rule_type(cfg.experiment.transfer_rule_type)

    n_trials = cfg.experiment.n_trials
    n_agents = cfg.experiment.get("n_agents", 4)
    total_batches = (len(participant_ids) + n_agents - 1) // n_agents
    group_results: list[dict] = []

    for batch_num, batch_ids in enumerate(_chunked(participant_ids, n_agents), start=1):
        logger.info(f"\n{'='*60}")
        logger.info(
            "Encoding %s batch %s/%s | Participants %s-%s (%s agents)",
            encoding_label,
            batch_num,
            total_batches,
            batch_ids[0],
            batch_ids[-1],
            len(batch_ids),
        )
        logger.info(f"{'='*60}")

        env = StarMakingEnvBatchText(
            llm_client=llm_client,
            rules=rules,
            n_agents=len(batch_ids),
            rule_type="learning",
        )
        goal_stars_per_agent = [list(goal_schedule) for _ in batch_ids]
        batch_results = env.run_experiment_batch(
            n_trials=n_trials,
            goal_stars_per_agent=goal_stars_per_agent,
            rule_schedule=rule_schedule,
        )

        for idx, agent_data in enumerate(batch_results["agents"]):
            agent_data["agent_id"] = batch_ids[idx]
            agent_data["encoding"] = encoding_label
        group_results.extend(batch_results["agents"])

    return group_results


@hydra.main(version_base=None, config_path="../config", config_name="experiment")
def main(cfg: DictConfig) -> None:
    """Run hierarchical LLM action-sequence experiment."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared GPU cache")

    n_trials = cfg.experiment.n_trials
    num_participants = cfg.experiment.num_participants
    n_agents = cfg.experiment.get("n_agents", 4)
    logger.info("Initializing experiment with %s participants, %s trials each", num_participants, n_trials)
    logger.info("Transfer rule type: %s", cfg.experiment.transfer_rule_type)
    logger.info("Agents per batch: %s", n_agents)

    goal_schedule, rule_schedule = _build_goal_and_rule_schedule(n_trials)
    logger.info(
        "Trial split: %s learning / %s transfer (block size ~%s)",
        rule_schedule.count("learning"),
        rule_schedule.count("transfer"),
        max(1, n_trials // 12),
    )

    # Initialize vLLM client
    tensor_parallel_size = cfg.model.get("tensor_parallel_size", 1)
    logger.info("Loading vLLM model %s %s (%s GPU(s))", cfg.model.family, cfg.model.size, tensor_parallel_size)
    llm_client = LocalVLLMClient(
        family=cfg.model.family,
        size=cfg.model.size,
        dtype=cfg.model.dtype,
        tensor_parallel_size=tensor_parallel_size,
    )

    participant_ids = list(range(num_participants))
    if num_participants < 2:
        logger.warning("num_participants=%s < 2, assigning all to ACTION_ENCODE_1", num_participants)
        encoding_groups = [(participant_ids, 1)]
    else:
        split_idx = num_participants // 2
        encoding_groups = [
            (participant_ids[:split_idx], 1),
            (participant_ids[split_idx:], 2),
        ]
    encode1_count = len(encoding_groups[0][0])
    encode2_count = len(encoding_groups[1][0]) if len(encoding_groups) > 1 else 0
    logger.info(
        "Encoding split: %s participants -> ACTION_ENCODE_1, %s participants -> ACTION_ENCODE_2",
        encode1_count,
        encode2_count,
    )

    all_agents: list[dict] = []
    for ids, encoding in encoding_groups:
        all_agents.extend(_run_encoding_group(ids, encoding, cfg, llm_client, goal_schedule, rule_schedule))

    participants = []
    for agent_data in sorted(all_agents, key=lambda x: x["agent_id"]):
        trials = agent_data["trials"]
        successes = sum(1 for trial in trials if trial["success"])
        summary = {
            "total_trials": len(trials),
            "successful_trials": successes,
            "success_rate": successes / len(trials) if trials else 0.0,
        }
        participants.append({
            "participant_id": agent_data["agent_id"],
            "encoding": agent_data["encoding"],
            "conversation_history": agent_data["conversation_history"],
            "trials": trials,
            "summary": summary,
        })

    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"{'='*60}")
    for participant in participants:
        logger.info(
            "Participant %s (encoding %s): %s/%s (%.2f%%)",
            participant["participant_id"],
            participant["encoding"],
            participant["summary"]["successful_trials"],
            participant["summary"]["total_trials"],
            participant["summary"]["success_rate"] * 100,
        )

    output = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "timestamp": datetime.now().isoformat(),
        "participants": participants,
        "experiment_schedule": {
            "goal_stars": goal_schedule,
            "rule_schedule": rule_schedule,
            "transfer_rule_type": cfg.experiment.transfer_rule_type,
        },
        "batch_mode": True,
    }

    if cfg.output.save_json:
        model_name = f"{cfg.model.family}-{cfg.model.size}"
        output_dir = Path(cfg.output.dir) / model_name / cfg.experiment.transfer_rule_type
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{timestamp}_experiment.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to: {output_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
