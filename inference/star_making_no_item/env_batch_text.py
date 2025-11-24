from tqdm import tqdm
from .env_text import StarMakingSimulator
from .assets_utils import (
    get_system_prompt,
    format_state_prompt,
    format_feedback_prompt,
    parse_action,
    ACTION_PROMPT_INSTRUCTION,
    ConversationHistoryManager,
)


class StarMakingEnvBatchText:
    """Batched text-based environment for star making with LLM interaction."""

    def __init__(self, llm_client, rules, n_agents, rule_type='learning'):
        self.llm_client = llm_client
        self.rules = rules
        self.n_agents = n_agents
        self.rule_type = rule_type
        self.simulators = [StarMakingSimulator(rules, rule_type) for _ in range(n_agents)]
        self.conversation_managers = [ConversationHistoryManager() for _ in range(n_agents)]

    def _parse_action(self, response):
        """Parse action from LLM response.

        Returns:
            tuple: (action, parse_success)
        """
        return parse_action(response, self.rules)

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
        return self.conversation_managers[agent_idx].prepare_for_llm(ACTION_PROMPT_INSTRUCTION)

    def get_actions_batch_with_retry(self, agent_indices, max_retries=10):
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
                    self.conversation_managers[agent_idx].add_message("assistant", response_content)
                    self.conversation_managers[agent_idx].set_awaiting_action(False)
                    actions[idx_in_actions] = action
                else:
                    # Don't add to history - retry with same conversation state
                    new_pending.append(idx_in_actions)

            pending = new_pending

        # Default remaining failures to action 0
        for idx in pending:
            actions[idx] = 0
            print(f"Warning: Agent {agent_indices[idx]} parse failed after {max_retries} attempts. Defaulting to action 0.")
            self.conversation_managers[agent_indices[idx]].set_awaiting_action(False)

        return actions

    def run_trial_batch(self, goal_stars, rule_type=None):
        """Run single trial for all N agents in batch mode.

        Args:
            goal_stars: List of goal stars, one per agent
            rule_type: Rule type to use for this trial ("learning" or "transfer")

        Returns:
            List of success booleans
        """
        active_rule = rule_type or self.rule_type
        for simulator in self.simulators:
            simulator.rule_type = active_rule

        # Reset all simulators
        for i, goal_star in enumerate(goal_stars):
            self.simulators[i].reset(goal_star)

        # Initial state prompts for all agents
        for i in range(self.n_agents):
            state = self.simulators[i].get_state()
            prompt = format_state_prompt(state, self.rules)
            self.conversation_managers[i].add_message("user", prompt)
            self.conversation_managers[i].set_awaiting_action(True)

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
                feedback = format_feedback_prompt(action, state, is_complete, self.rules, trial_end=is_last_round)
                self.conversation_managers[i].add_message("user", feedback)
                self.conversation_managers[i].set_awaiting_action(not is_last_round)

        # Check success for all agents
        return [sim.is_complete() for sim in self.simulators]

    def run_experiment_batch(self, n_trials=20, goal_stars_per_agent=None, specify_star=None, rule_schedule=None):
        """Run full experiment for all agents in batch mode.

        Args:
            n_trials: Number of trials to run
            goal_stars_per_agent: Optional list of lists - goal stars for each agent and trial
                                 Shape: [n_agents][n_trials]
            specify_star: Optional star to use for all agents and all trials
            rule_schedule: Optional list of rule types per trial

        Returns:
            Dict with results for each agent
        """
        if rule_schedule and len(rule_schedule) != n_trials:
            raise ValueError("rule_schedule length must match n_trials")

        # Initialize conversation histories - use in-context for transfer, standard prompt for learning
        if rule_schedule:
            should_use_transfer_prompt = all(rule == 'transfer' for rule in rule_schedule)
        else:
            should_use_transfer_prompt = (self.rule_type == 'transfer')

        for i in range(self.n_agents):
            if should_use_transfer_prompt:
                self.conversation_managers[i].initialize_with_in_context(n_trials)
            else:
                self.conversation_managers[i].initialize(get_system_prompt(n_trials))

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

            current_rule_type = rule_schedule[trial] if rule_schedule else self.rule_type

            # Run trial for all agents
            successes = self.run_trial_batch(goal_stars, rule_type=current_rule_type)

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
                    "conversation_history": self.conversation_managers[i].get_history(),
                    "success_rate": sum(1 for t in agent_results[i] if t['success']) / n_trials
                }
                for i in range(self.n_agents)
            ]
        }
