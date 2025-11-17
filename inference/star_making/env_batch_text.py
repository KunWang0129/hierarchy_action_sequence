"""
The is the batched version of the inference/star_making/env_text.py module. Most modules are similar and can be shared.
The main difference are given below:

Idea: given inputs:
- vllm client
- star making simulators
- number of agents

This runs the star making environment in batch mode, where we store conversation history for each agent
and process multiple LLM requests in batches (to vllm client that supports batching)

The process starts at each trial, and we keep track of which agents are still active in the current trial. When trial ends for all agents we move to next trial.

Note: Use a variable that index the agent in the batch of prompts/responses to vllm client, as there can be descrepancy on the progress of each agent
- e.g. some agents might take more completions due to parsing errors, while others can finish earlier.


Related files:
inference/llm/local_vllm_clients.py - local vllm client that supports batch processing (revise as needed)
inference/inference_batch.py - main inference module that runs batched star making env with multiple agents (with vllm clients)
"""
from tqdm import tqdm
from .env_text import StarMakingSimulator
from .assets_utils import (
    get_system_prompt,
    format_state_prompt,
    format_feedback_prompt,
    parse_action,
    ACTION_PROMPT_INSTRUCTION,
)


class StarMakingEnvBatchText:
    """Batched text-based environment for star making with LLM interaction."""

    def __init__(self, llm_client, rules, n_agents, rule_type='learning'):
        self.llm_client = llm_client
        self.rules = rules
        self.n_agents = n_agents
        self.rule_type = rule_type
        self.simulators = [StarMakingSimulator(rules, rule_type) for _ in range(n_agents)]
        self.conversation_histories = [[] for _ in range(n_agents)]
        self.awaiting_action_prompts = [False] * n_agents

    def _parse_action(self, response):
        """Parse action from LLM response.

        Returns:
            tuple: (action, parse_success)
        """
        return parse_action(response)

    def get_actions_batch(self, agent_indices):
        """Get actions for multiple agents in one batch request.

        Args:
            agent_indices: List of agent IDs that need actions

        Returns:
            List of (response_content, action, parse_success) tuples in same order as input
        """
        message_batches = [self._prepare_conversation_for_agent(i) for i in agent_indices]
        responses = self.llm_client.generate_batch(message_batches)

        # Parse each response (but don't add to conversation history yet)
        results = []
        for i, response in enumerate(responses):
            action, parse_success = self._parse_action(response)
            results.append((response["content"], action, parse_success))

        return results

    def _prepare_conversation_for_agent(self, agent_idx):
        """Append action instruction for an agent without mutating stored history."""
        if not (self.awaiting_action_prompts[agent_idx] and self.conversation_histories[agent_idx]):
            return self.conversation_histories[agent_idx]

        last_message = self.conversation_histories[agent_idx][-1]
        if last_message.get("role") != "user":
            return self.conversation_histories[agent_idx]

        prepared_history = list(self.conversation_histories[agent_idx][:-1])
        last_copy = last_message.copy()
        last_copy["content"] = f"{last_copy['content']}{ACTION_PROMPT_INSTRUCTION}"
        prepared_history.append(last_copy)
        return prepared_history

    def get_actions_batch_with_retry(self, agent_indices, max_retries=50):
        """Get actions for multiple agents with retry for parsing failures.

        Args:
            agent_indices: List of agent IDs that need actions
            max_retries: Maximum number of retry attempts for parsing failures

        Returns:
            List of actions in same order as input agent_indices
        """
        actions = [None] * len(agent_indices)
        pending = list(range(len(agent_indices)))  # Indices into agent_indices list

        for attempt in range(max_retries):
            if not pending:
                break

            # Get batch of agents that still need actions
            pending_agent_indices = [agent_indices[i] for i in pending]
            results = self.get_actions_batch(pending_agent_indices)

            # Process results and only add successful responses to history
            new_pending = []
            for i, (response_content, action, parse_success) in enumerate(results):
                idx_in_actions = pending[i]
                agent_idx = agent_indices[idx_in_actions]

                if parse_success:
                    # Only add to conversation history if parsing succeeded
                    self.conversation_histories[agent_idx].append({"role": "assistant", "content": response_content})
                    self.awaiting_action_prompts[agent_idx] = False
                    actions[idx_in_actions] = action
                else:
                    # Don't add to history - retry with same conversation state
                    new_pending.append(idx_in_actions)

            pending = new_pending

        # Default remaining failures to action 0
        for idx in pending:
            actions[idx] = 0
            print(f"Warning: Agent {agent_indices[idx]} parse failed after {max_retries} attempts. Defaulting to action 0.")
            self.awaiting_action_prompts[agent_indices[idx]] = False

        return actions

    def run_trial_batch(self, goal_stars):
        """Run single trial for all N agents in batch mode.

        Args:
            goal_stars: List of goal stars, one per agent

        Returns:
            List of success booleans
        """
        # Reset all simulators
        for i, goal_star in enumerate(goal_stars):
            self.simulators[i].reset(goal_star)

        # Initial state prompts for all agents
        for i in range(self.n_agents):
            state = self.simulators[i].get_state()
            prompt = format_state_prompt(state)
            self.conversation_histories[i].append({"role": "user", "content": prompt})
            self.awaiting_action_prompts[i] = True

        # 4 rounds
        all_agent_indices = list(range(self.n_agents))
        for round_idx in range(4):
            # Get actions for all agents in batch
            actions = self.get_actions_batch_with_retry(all_agent_indices)

            # Execute actions and provide feedback
            for i, action in enumerate(actions):
                self.simulators[i].step(action)
                state = self.simulators[i].get_state()
                is_last_round = (round_idx == 3)
                is_complete = self.simulators[i].is_complete()
                feedback = format_feedback_prompt(action, state, is_complete, trial_end=is_last_round)
                self.conversation_histories[i].append({"role": "user", "content": feedback})
                self.awaiting_action_prompts[i] = not is_last_round

        # Check success for all agents
        return [sim.is_complete() for sim in self.simulators]

    def run_experiment_batch(self, n_trials=20, goal_stars_per_agent=None, specify_star=None):
        """Run full experiment for all agents in batch mode.

        Args:
            n_trials: Number of trials to run
            goal_stars_per_agent: Optional list of lists - goal stars for each agent and trial
                                 Shape: [n_agents][n_trials]
            specify_star: Optional star to use for all agents and all trials

        Returns:
            Dict with results for each agent
        """
        # Initialize conversation histories with system prompt
        system_prompt = get_system_prompt(n_trials)
        for i in range(self.n_agents):
            self.conversation_histories[i] = [{"role": "system", "content": system_prompt}]
            self.awaiting_action_prompts[i] = False

        # Initialize results storage
        agent_results = [[] for _ in range(self.n_agents)]

        # Run trials
        for trial in tqdm(range(n_trials), desc="Running trials (batched)"):
            # Determine goal stars for each agent
            goal_stars = []
            for agent_idx in range(self.n_agents):
                if goal_stars_per_agent and goal_stars_per_agent[agent_idx]:
                    goal_star = goal_stars_per_agent[agent_idx][trial]
                elif specify_star is not None:
                    goal_star = specify_star
                else:
                    goal_star = f"Star_{(trial // (n_trials // 8)) % 4}"
                goal_stars.append(goal_star)

            # Run trial for all agents
            successes = self.run_trial_batch(goal_stars)

            # Store results
            for agent_idx in range(self.n_agents):
                agent_results[agent_idx].append({
                    'trial': trial,
                    'goal_star': goal_stars[agent_idx],
                    'success': successes[agent_idx]
                })

        # Aggregate results
        return {
            "agents": [
                {
                    "agent_id": i,
                    "trials": agent_results[i],
                    "conversation_history": self.conversation_histories[i],
                    "success_rate": sum(1 for t in agent_results[i] if t['success']) / n_trials
                }
                for i in range(self.n_agents)
            ]
        }
