"""
Batched inference module for star making task using vLLM.
This partitions participants across batches and processes each batch with a fixed number of agents.

Key differences from inference.py:
- Uses LocalVLLMClient (supports batch processing)
- Uses StarMakingEnvBatchText (processes multiple agents in parallel)
- Partitions participants into batches (default 4 agents per batch)
- Runs multiple batches sequentially to process all participants
- Only supports vLLM provider (Gemini doesn't support true batching)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from inference.llm.local_vllm_clients import LocalVLLMClient
from inference.star_making.assets_utils import StarMakingRules
from inference.star_making.env_batch_text import StarMakingEnvBatchText

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="inference_batch")
def main(cfg: DictConfig) -> None:
    """Run star making inference experiment in batch mode"""
    # Clear GPU memory at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared GPU cache")

    logger.info("Initializing batched star making inference...")
    logger.info(f"Model: {cfg.model.family} {cfg.model.size}")
    logger.info(f"Rule type: {cfg.experiment.rule_type}")
    logger.info(f"Trials per agent: {cfg.experiment.n_trials}")
    logger.info(f"Total participants: {cfg.experiment.num_participants}")

    # Get number of agents to run in parallel (default 4)
    n_agents = cfg.experiment.get("n_agents", 4)
    logger.info(f"Agents per batch: {n_agents}")

    # Calculate number of batches needed
    num_batches = (cfg.experiment.num_participants + n_agents - 1) // n_agents
    logger.info(f"Number of batches: {num_batches}")

    specify_star = cfg.experiment.get("specify_star", None)
    if specify_star:
        logger.info(f"Goal star: {specify_star} (all trials)")
    else:
        logger.info(f"Goal stars: Split evenly across Star_0, Star_1, Star_2, Star_3")

    # Initialize vLLM client
    tensor_parallel_size = cfg.model.get("tensor_parallel_size", 1)
    logger.info("Loading vLLM model...")
    logger.info(f"Tensor parallel size: {tensor_parallel_size} GPU(s)")
    llm_client = LocalVLLMClient(
        family=cfg.model.family,
        size=cfg.model.size,
        dtype=cfg.model.dtype,
        tensor_parallel_size=tensor_parallel_size,
    )
    logger.info(f"Using vLLM model: {cfg.model.family} {cfg.model.size}")

    # Initialize star making rules
    rules = StarMakingRules()

    # Process participants in batches
    all_agents = []
    for batch_idx in range(num_batches):
        start_participant = batch_idx * n_agents
        end_participant = min(start_participant + n_agents, cfg.experiment.num_participants)
        batch_size = end_participant - start_participant

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing batch {batch_idx + 1}/{num_batches}")
        logger.info(f"Participants {start_participant} to {end_participant - 1} ({batch_size} agents)")
        logger.info(f"{'='*60}")

        # Create batched environment for this batch
        env = StarMakingEnvBatchText(
            llm_client=llm_client,
            rules=rules,
            n_agents=batch_size,
            rule_type=cfg.experiment.rule_type,
        )

        # Run batched experiment for this batch
        results = env.run_experiment_batch(
            n_trials=cfg.experiment.n_trials,
            specify_star=cfg.experiment.get("specify_star", None)
        )

        # Update agent IDs to reflect global participant numbering
        for i, agent_data in enumerate(results["agents"]):
            agent_data["agent_id"] = start_participant + i

        # Aggregate results
        all_agents.extend(results["agents"])

        logger.info(f"Batch {batch_idx + 1} complete")

        # # Clear GPU memory between batches
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info("BATCH EXPERIMENT COMPLETE")
    logger.info(f"{'='*60}")
    for agent_data in all_agents:
        logger.info(
            f"Agent {agent_data['agent_id']}: "
            f"{sum(1 for t in agent_data['trials'] if t['success'])}/{len(agent_data['trials'])} "
            f"({agent_data['success_rate']:.2%})"
        )

    # Prepare output (convert agents to participants for consistency with inference.py)
    participants = []
    for agent_data in all_agents:
        participants.append({
            "participant_id": agent_data["agent_id"],
            "conversation_history": agent_data["conversation_history"],
            "trials": agent_data["trials"],
            "summary": {
                "total_trials": len(agent_data["trials"]),
                "successful_trials": sum(1 for t in agent_data["trials"] if t["success"]),
                "success_rate": agent_data["success_rate"],
            },
        })

    output = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "timestamp": datetime.now().isoformat(),
        "participants": participants,
        "batch_mode": True,
    }

    # Save results
    if cfg.output.save_json:
        model_name = f"{cfg.model.family}-{cfg.model.size}"
        star_dir = specify_star if specify_star else "None"

        # Create nested directory structure: results/{model}/{star}/
        output_dir = Path(cfg.output.dir) / model_name / star_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{timestamp}_inference_batch.json"

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
