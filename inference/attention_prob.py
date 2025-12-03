"""
Attention Probability Extraction for Trial Replay Analysis

This script loads existing conversation histories from completed experiments,
replays specific trials (trial N) with conversation history up to trial N-1,
and extracts attention probabilities during the replay for analysis.

Goal:
- Load M participants from existing experiment results
- For each participant, replay trial N with history up to N-1
- Extract attention probabilities during generation (placeholder for now)
- Compare replay outcomes with original outcomes
- Save attention data for downstream analysis
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from inference.llm.local_llm_clients import LocalLLMClient
from inference.star_making.assets_utils import (
    ACTION_PROMPT_INSTRUCTION,
    ConversationHistoryManager,
    StarMakingRules,
)
from inference.star_making.env_text import StarMakingSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AttentionLLMClient(LocalLLMClient):
    """
    Extended LLM client with attention probability extraction capability.

    Currently uses placeholder functions for attention extraction.
    Future implementation will integrate with model-specific attention APIs.
    """

    def extract_attention_probabilities(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Generate a single token and extract attention probabilities without allocating
        full seq_len x seq_len attention maps.

        The prompt is run once to build the KV cache (no attentions). We sample the
        next token from the prompt logits, then run a single cached forward pass with
        output_attentions=True to get a [layers, heads, 1, seq_len] tensor. We drop
        the self-position, renormalize over prompt tokens so the vector sums to 1.0,
        and average over heads then layers.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Step 1: Prepare input
        prompt = self._apply_chat_template(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.model.device)
        else:
            attention_mask = attention_mask.to(self.model.device)
        input_length = input_ids.shape[-1]
        if input_length > 90000:
            logger.warning(f"Attention extraction on very long prompt ({input_length} tokens) may still be memory heavy.")

        # Chunked prefill to avoid quadratic memory spikes on very long prompts
        prefill_chunk_size = 512
        with torch.no_grad():
            past_key_values = None
            logits = None

            total_len = input_ids.shape[1]
            for start in range(0, total_len, prefill_chunk_size):
                end = min(start + prefill_chunk_size, total_len)
                chunk_ids = input_ids[:, start:end]
                # Attention mask must cover all tokens seen so far
                chunk_mask = attention_mask[:, :end]

                prompt_outputs = self.model(
                    input_ids=chunk_ids,
                    attention_mask=chunk_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True,
                )

                past_key_values = prompt_outputs.past_key_values
                logits = prompt_outputs.logits[:, -1, :]

            # Sample next token (temperature/top-p) from prompt logits
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    cutoff_mask = cumulative > top_p
                    cutoff_mask[..., 0] = False
                    sorted_probs = sorted_probs.masked_fill(cutoff_mask, 0.0)
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = torch.gather(sorted_indices, -1, next_token)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)

            # Extend attention mask for generated token
            extended_attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)],
                dim=1,
            )

            # Pass 2: single-step forward with cache to obtain attentions for the generated token
            step_outputs = self.model(
                input_ids=next_token,
                attention_mask=extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )

        # Decode generated token/content
        content = self.tokenizer.decode(next_token[0], skip_special_tokens=True).strip()
        generated_token_text = self.tokenizer.decode(next_token[0].item())

        # Aggregate attentions: [layers, heads, 1, seq_len+1] -> [seq_len] probability vector
        layer_attns = step_outputs.attentions  # tuple per layer
        num_layers = len(layer_attns)
        num_heads = layer_attns[0].shape[1] if num_layers > 0 else 0
        layer_distributions = []

        for layer_attn in layer_attns:
            # layer_attn: [batch=1, heads, q_len=1, kv_len=input_length+1]
            attn_to_prompt = layer_attn[0, :, 0, :input_length]  # exclude self position
            head_sums = attn_to_prompt.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            normalized = attn_to_prompt / head_sums  # head-level distributions sum to 1 over prompt tokens
            layer_distributions.append(normalized.mean(dim=0))  # average over heads -> [seq_len]

        averaged_attention = torch.stack(layer_distributions, dim=0).mean(dim=0)  # average over layers

        # Decode input tokens and build result dictionary
        input_token_ids = input_ids[0].tolist()
        attention_weights = {
            pos: (self.tokenizer.decode([token_id]), float(attn_score))
            for pos, (token_id, attn_score) in enumerate(zip(input_token_ids, averaged_attention.tolist()))
        }

        return {
            "content": content,
            "attention_weights": attention_weights,
            "input_length": input_length,
            "generated_token_text": generated_token_text,
            "layer_count": num_layers,
            "head_count": num_heads,
            "implementation_status": "complete",
        }


def load_experiment_data(experiment_path: Path) -> Dict:
    """
    Load existing experiment results from JSON file.
    """
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {experiment_path}")

    logger.info(f"Loading experiment data from: {experiment_path}")

    with open(experiment_path, "r") as f:
        data = json.load(f)

    # Validate required fields
    required_fields = ["participants", "experiment_schedule"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field in experiment data: {field}")

    logger.info(f"Loaded data for {len(data['participants'])} participants")
    return data


def select_participants_for_replay(
    experiment_data: Dict,
    num_participants: int,
    specific_ids: Optional[List[int]] = None,
    target_encoding: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Select M participants from experiment data for replay analysis.
    """
    participants = experiment_data["participants"]

    # Filter by encoding if specified
    if target_encoding is not None:
        participants = [p for p in participants if p["encoding"] == target_encoding]
        logger.info(f"Filtered to {len(participants)} participants with encoding {target_encoding}")

    # Use specific IDs if provided
    if specific_ids is not None:
        participants = [p for p in participants if p["participant_id"] in specific_ids]
        logger.info(f"Selected {len(participants)} participants by specific IDs: {specific_ids}")
    else:
        # Random selection
        if seed is not None:
            random.seed(seed)

        if len(participants) > num_participants:
            participants = random.sample(participants, num_participants)
            logger.info(f"Randomly selected {num_participants} participants")
        else:
            logger.info(f"Using all {len(participants)} available participants")

    return participants


def determine_replay_trial_index(
    experiment_data: Dict,
    replay_trial_index: Optional[int] = None,
) -> int:
    """
    Determine which trial index to replay.
    """
    if replay_trial_index is not None:
        trial_idx = replay_trial_index
        logger.info(f"Using specified replay trial index: {trial_idx}")
    else:
        # Use transfer start trial from experiment schedule
        rule_schedule = experiment_data["experiment_schedule"]["rule_schedule"]
        trial_idx = rule_schedule.index("transfer") if "transfer" in rule_schedule else 0
        logger.info(f"Using transfer start trial as replay index: {trial_idx}")

    return trial_idx


def reconstruct_conversation_history(
    participant_data: Dict,
    trial_index: int,
) -> Tuple[ConversationHistoryManager, Dict]:
    """
    Reconstruct conversation history up to trial N-1 and extract trial N state.

    We slice the conversation_history so it contains all messages through the end of
    trial_index-1 plus the initial user state message for trial_index.
    """
    def slice_history_by_trial(history: List[Dict[str, str]], target_trial: int, rounds_per_trial: int = 4) -> List[Dict[str, str]]:
        """Keep trials [0, target_trial-1] fully and the first user message(s) of target_trial."""
        if target_trial <= 0:
            return history

        sliced = []
        trial = 0
        assistants_in_trial = 0
        awaiting_target_start = False

        for idx, msg in enumerate(history):
            role = msg.get("role")

            if awaiting_target_start:
                if role == "user":
                    sliced.append(msg)
                    # Guard check: if the next message is also a user message, include it too
                    if idx + 1 < len(history) and history[idx + 1].get("role") == "user":
                        sliced.append(history[idx + 1])
                    break
                # Skip any non-user chatter between trials
                continue

            sliced.append(msg)

            if role == "assistant":
                assistants_in_trial += 1
                if assistants_in_trial == rounds_per_trial:
                    trial += 1
                    assistants_in_trial = 0
                    if trial == target_trial:
                        awaiting_target_start = True

        return sliced

    # Get conversation history up to trial N-1
    conv_history = slice_history_by_trial(participant_data["conversation_history"], trial_index)

    # Reconstruct conversation manager
    manager = ConversationHistoryManager()
    for msg in conv_history:
        manager.add_message(msg["role"], msg["content"])

    # Get the state at trial N
    trials = participant_data["trials"]
    if trial_index >= len(trials):
        raise ValueError(f"Trial index {trial_index} out of range (max: {len(trials) - 1})")

    trial_state = trials[trial_index]

    logger.debug(f"Reconstructed history with {len(conv_history)} messages for trial {trial_index}")

    return manager, trial_state


def replay_trial_with_attention(
    llm_client: AttentionLLMClient,
    conversation_manager: ConversationHistoryManager,
    trial_state: Dict,
    rules: StarMakingRules,
) -> Dict:
    """
    Replay a single trial and extract attention probabilities.

    This function now focuses solely on collecting attention over the full prompt.
    """
    # Prepare messages for generation
    messages = conversation_manager.prepare_for_llm(ACTION_PROMPT_INSTRUCTION)

    # Generate with attention extraction (single call; no parsing/retries)
    result = llm_client.extract_attention_probabilities(
        messages=messages,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )

    attention_data = {
        "attention_weights": result["attention_weights"],
        "input_length": result["input_length"],
        "generated_token_text": result["generated_token_text"],
        "layer_count": result["layer_count"],
        "head_count": result["head_count"],
        "implementation_status": result["implementation_status"],
    }

    return {
        "response_content": result["content"],
        "attention_data": attention_data,
    }


def save_attention_analysis_results(
    results: List[Dict],
    cfg: DictConfig,
    experiment_metadata: Dict,
) -> Path:
    """
    Save attention analysis results to disk.
    """
    # Extract input experiment filename
    input_experiment_path = Path(cfg.attention_analysis.input_experiment_path)
    experiment_name = input_experiment_path.stem

    # Extract trial_index from first result (all participants share the same trial_index)
    trial_index = results[0]["trial_index"] if results else 0

    # Create output directory with experiment name as subdirectory
    model_name = f"{cfg.model.family}-{cfg.model.size}"
    output_dir = Path(cfg.output.dir) / model_name / cfg.attention_analysis.output_subdir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare output data
    output_data = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "timestamp": datetime.now().isoformat(),
        "source_experiment": str(cfg.attention_analysis.input_experiment_path),
        "replay_results": results,
        "metadata": {
            "num_participants_replayed": len(results),
            "original_experiment_config": experiment_metadata.get("config"),
            "original_experiment_timestamp": experiment_metadata.get("timestamp"),
        },
    }

    # Save main JSON file (without timestamp in filename)
    output_path = output_dir / f"trial{trial_index}_attention_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved attention analysis results to: {output_path}")

    # Save attention weights separately if configured
    if cfg.attention_analysis.save_attention_weights:
        weights_dir = output_dir / f"trial{trial_index}_attention_weights"
        weights_dir.mkdir(exist_ok=True)

        # Create manifest for attention weight files
        manifest = []
        for result in results:
            participant_id = result["participant_id"]
            attention_weights = result["attention_weights"]

            # Save attention weights as JSON
            # Format: {position: [token_text, attention_score]}
            # Convert tuple values to lists for JSON serialization
            attention_weights_json = {
                str(pos): [token, score]
                for pos, (token, score) in attention_weights.items()
            }

            weight_file = f"participant_{participant_id}_attention.json"
            weight_path = weights_dir / weight_file
            with open(weight_path, "w") as f:
                json.dump(attention_weights_json, f, indent=2)

            manifest.append({
                "participant_id": participant_id,
                "trial_index": result["trial_index"],
                "weight_file": weight_file,
                "input_length": result["attention_data_summary"]["input_length"],
                "generated_token": result["attention_data_summary"]["generated_token_text"],
                "format": "json",
                "structure": "position -> [token_text, attention_score]",
            })

        manifest_path = weights_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Attention weights saved to: {weights_dir}")
        logger.info(f"Attention weights manifest saved to: {manifest_path}")

    return output_path


@hydra.main(version_base=None, config_path="../config", config_name="attention_prob")
def main(cfg: DictConfig) -> None:
    """
    Main function for attention probability extraction analysis.

    Workflow:
    1. Load existing experiment results
    2. Select M participants for replay
    3. Determine which trial N to replay
    4. For each participant:
        a. Reconstruct conversation history up to trial N-1
        b. Extract trial N state
        c. Replay trial N with attention extraction (no action parsing)
    5. Save aggregated results and attention data
    """
    logger.info("="*60)
    logger.info("ATTENTION PROBABILITY EXTRACTION ANALYSIS")
    logger.info("="*60)

    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared GPU cache")

    # Load existing experiment data
    experiment_path = Path(cfg.attention_analysis.input_experiment_path)
    experiment_data = load_experiment_data(experiment_path)

    # Select participants for replay
    selected_participants = select_participants_for_replay(
        experiment_data=experiment_data,
        num_participants=cfg.attention_analysis.num_replay_participants,
        specific_ids=cfg.attention_analysis.get("specific_participant_ids"),
        target_encoding=cfg.attention_analysis.get("target_encoding"),
        seed=42,  # For reproducibility
    )

    # Determine replay trial index
    replay_trial_idx = determine_replay_trial_index(
        experiment_data=experiment_data,
        replay_trial_index=cfg.attention_analysis.get("replay_trial_index"),
    )

    logger.info(f"Will replay trial {replay_trial_idx} for {len(selected_participants)} participants")

    # Initialize attention-enhanced LLM client
    logger.info(f"Loading model: {cfg.model.family} {cfg.model.size}")
    llm_client = AttentionLLMClient(
        family=cfg.model.family,
        size=cfg.model.size,
        dtype=cfg.model.dtype,
        device_map="auto",
    )

    # Process each participant
    replay_results = []

    for idx, participant in enumerate(selected_participants, start=1):
        participant_id = participant["participant_id"]
        encoding = participant["encoding"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Replaying participant {participant_id} ({idx}/{len(selected_participants)})")
        logger.info(f"Encoding: {encoding}")
        logger.info(f"{'='*60}")

        # Reconstruct conversation history up to trial N-1
        conversation_manager, trial_state = reconstruct_conversation_history(
            participant_data=participant,
            trial_index=replay_trial_idx,
        )

        # Set up rules for this participant's encoding
        rules = StarMakingRules(encoding=encoding)
        # Note: transfer_rule_type should match original experiment
        transfer_rule_type = experiment_data["experiment_schedule"].get("transfer_rule_type", "transfer_low")
        rules.set_transfer_rule_type(transfer_rule_type)

        # Replay trial with attention extraction
        replay_result = replay_trial_with_attention(
            llm_client=llm_client,
            conversation_manager=conversation_manager,
            trial_state=trial_state,
            rules=rules,
        )

        # Log top attention weights
        attention_weights = replay_result["attention_data"]["attention_weights"]
        sorted_attention = sorted(attention_weights.items(), key=lambda x: x[1][1], reverse=True)
        logger.info(f"Top 5 attention weights:")
        for pos, (token, score) in sorted_attention[:5]:
            logger.info(f"  Position {pos}: '{token}' -> {score:.4f}")

        # Aggregate results
        replay_results.append({
            "participant_id": participant_id,
            "encoding": encoding,
            "trial_index": replay_trial_idx,
            "response_content": replay_result["response_content"],
            "attention_weights": replay_result["attention_data"]["attention_weights"],
            "attention_data_summary": {
                "input_length": replay_result["attention_data"]["input_length"],
                "generated_token_text": replay_result["attention_data"]["generated_token_text"],
                "layer_count": replay_result["attention_data"]["layer_count"],
                "head_count": replay_result["attention_data"]["head_count"],
                "implementation_status": replay_result["attention_data"]["implementation_status"],
            },
        })

    # Save results
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"{'='*60}")

    # Summary statistics
    total_replayed = len(replay_results)
    logger.info(f"Total participants replayed: {total_replayed}")

    # Save results to disk
    output_path = save_attention_analysis_results(
        results=replay_results,
        cfg=cfg,
        experiment_metadata=experiment_data,
    )

    logger.info(f"Results saved to: {output_path}")

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
