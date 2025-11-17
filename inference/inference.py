"""
Inference module for star making task.
This runs the experiment of star making using LLM client and star making environment.
It uses the StarMakingEnvText to interact with the LLM client and simulate the star making task

Important addition:
Store and evaluation:
- Stores custom "conversation history" with LLM for context, which includes:
    - System prompt defining the star making task
    - User prompts reflecting start state and actions
    - Feedback prompts after each action
    - Prompts from previous trials
- Store the the number of trials completed and success rate in the result output
- Output conversation history and success rate

Config and param loading:
- We want to use hydra-core, and store configs in config/inference.yaml
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from inference.llm.local_llm_clients import LocalLLMClient
from inference.llm.llm_clients import GeminiClient
from inference.star_making.assets_utils import StarMakingRules
from inference.star_making.env_text import StarMakingEnvText

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="inference")
def main(cfg: DictConfig) -> None:
    """Run star making inference experiment"""
    # Clear GPU memory at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared GPU cache")

    logger.info("Initializing star making inference...")
    logger.info(f"Model: {cfg.model.family} {cfg.model.size}")
    logger.info(f"Rule type: {cfg.experiment.rule_type}")
    logger.info(f"Trials per participant: {cfg.experiment.n_trials}")
    logger.info(f"Number of participants: {cfg.experiment.num_participants}")

    specify_star = cfg.experiment.get("specify_star", None)
    if specify_star:
        logger.info(f"Goal star: {specify_star} (all trials)")
    else:
        logger.info(f"Goal stars: Split evenly across Star_0, Star_1, Star_2, Star_3")

    # Initialize LLM client (shared across participants)
    logger.info("Loading LLM model...")
    provider = cfg.model.get("provider", "local")

    if provider == "gemini":
        # Use Gemini API client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        model_name = cfg.model.get("model_name", "gemini-1.5-flash-latest")
        llm_client = GeminiClient(api_key=api_key, model=model_name)
        logger.info(f"Using Gemini model: {model_name}")
    else:
        # Use local model (default)
        llm_client = LocalLLMClient(
            family=cfg.model.family,
            size=cfg.model.size,
            dtype=cfg.model.dtype,
            device_map=cfg.model.device_map,
        )
        logger.info(f"Using local model: {cfg.model.family} {cfg.model.size}")

    # Initialize star making rules
    rules = StarMakingRules()

    # Run experiment for each participant
    all_participants = []
    for participant_id in range(cfg.experiment.num_participants):
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Participant {participant_id + 1}/{cfg.experiment.num_participants}")
        logger.info(f"{'='*60}")

        # Create fresh environment for each participant
        env = StarMakingEnvText(
            llm_client=llm_client,
            rules=rules,
            rule_type=cfg.experiment.rule_type,
        )

        # Run experiment
        results = env.run_experiment(
            n_trials=cfg.experiment.n_trials,
            specify_star=cfg.experiment.get("specify_star", None)
        )

        # Calculate success rate for this participant
        successes = sum(1 for trial in results["trials"] if trial["success"])
        success_rate = successes / len(results["trials"]) if results["trials"] else 0.0

        # Store participant data
        participant_data = {
            "participant_id": participant_id,
            "conversation_history": results["conversation_history"],
            "trials": results["trials"],
            "summary": {
                "total_trials": len(results["trials"]),
                "successful_trials": successes,
                "success_rate": success_rate,
            },
        }
        all_participants.append(participant_data)

        # Log this participant's results
        logger.info(f"Participant {participant_id} complete!")
        logger.info(f"  Successful trials: {successes}/{len(results['trials'])}")
        logger.info(f"  Success rate: {success_rate:.2%}")

        # # Clear GPU memory between participants
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

    # Prepare output
    output = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "timestamp": datetime.now().isoformat(),
        "participants": all_participants,
    }

    # Save results
    if cfg.output.save_json:
        # Construct model name for directory structure
        if provider == "gemini":
            model_name = cfg.model.get("model_name", "gemini")
        else:
            model_name = f"{cfg.model.family}-{cfg.model.size}"

        # Construct star directory name (use "None" if not specified)
        star_dir = specify_star if specify_star else "None"

        # Create nested directory structure: results/{model}/{star}/
        output_dir = Path(cfg.output.dir) / model_name / star_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{timestamp}_inference.json"

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")

    # Log summary of all participants
    logger.info(f"\n{'='*60}")
    logger.info("ALL PARTICIPANTS SUMMARY")
    logger.info(f"{'='*60}")
    for p in all_participants:
        logger.info(
            f"Participant {p['participant_id']}: "
            f"{p['summary']['successful_trials']}/{p['summary']['total_trials']} "
            f"({p['summary']['success_rate']:.2%})"
        )


if __name__ == "__main__":
    main()
