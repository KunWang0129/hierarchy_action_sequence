"""
This is the class for star making task interact with LLM client

StarMakingSimulator:
    - Simulates the star making environment using provided rules
    - Maintains current state including actions taken and items made
    - Provides methods to reset state, make actions, and check if goal star is made

StarMakingEnvText:
    - Interfaces with LLM client to process actions and provide feedback
    - Constructs prompts based on current state and actions
    - Stores custom "conversation history" with LLM for context, which includes:
        - System prompt defining the star making task
        - User prompts reflecting start state and actions
        - Feedback prompts after each action
        - Prompts from previous trials
    - Handles interaction loop for multiple trials and rounds
"""
from tqdm import tqdm
from .assets_utils import (
    get_system_prompt,
    format_state_prompt,
    format_feedback_prompt,
    parse_action,
    ACTION_PROMPT_INSTRUCTION,
)


class StarMakingSimulator:
    """Simulates the star making game environment."""

    def __init__(self, rules, rule_type='learning'):
        self.rules = rules
        self.rule_type = rule_type
        self.reset()

    def reset(self, goal_star=None):
        """Reset state for new trial."""
        self.actions = []
        self.items = []
        self.goal_star = goal_star

    def step(self, action):
        """Take action and update state."""
        self.actions.append(action)

        # Check for item creation at positions [0,1] or [2,3]
        if len(self.actions) == 2 and len(self.items) == 0:
            item = self.rules.get_item(self.actions, self.rule_type)
            if item:
                self.items.append(item)
        elif len(self.actions) == 4 and len(self.items) == 1:
            item = self.rules.get_item(self.actions[2:4], self.rule_type)
            if item:
                self.items.append(item)

    def is_complete(self):
        """Check if goal star is achieved."""
        if len(self.items) == 2:
            star = self.rules.get_star(self.items, self.rule_type)
            return star == self.goal_star
        return False

    def get_state(self):
        """Return current state dict."""
        # Ensure items array is always length 2, padding with None
        items = self.items + [None] * (2 - len(self.items))
        return {
            'goal_star': self.goal_star,
            'items': items,
            'actions': self.actions
        }


class StarMakingEnvText:
    """Text-based environment for star making with LLM interaction."""

    def __init__(self, llm_client, rules, rule_type='learning'):
        self.llm_client = llm_client
        self.simulator = StarMakingSimulator(rules, rule_type)
        self.conversation_history = []
        self.awaiting_action_prompt = False

    def _prepare_conversation_for_llm(self):
        """Append action instruction to latest user message without mutating history."""
        if not (self.awaiting_action_prompt and self.conversation_history):
            return self.conversation_history

        last_message = self.conversation_history[-1]
        if last_message.get("role") != "user":
            return self.conversation_history

        prepared_history = list(self.conversation_history[:-1])
        last_copy = last_message.copy()
        last_copy["content"] = f"{last_copy['content']}{ACTION_PROMPT_INSTRUCTION}"
        prepared_history.append(last_copy)
        return prepared_history

    def get_action(self):
        """Get action from LLM.

        Returns:
            tuple: (action, parse_success) where action is 0-3 and parse_success is bool
        """
        response = self.llm_client.get_response(self._prepare_conversation_for_llm())
        return parse_action(response)

    def get_action_with_retry(self, max_retries=50):
        """Get action from LLM with retry on parse failure.

        Args:
            max_retries: Maximum number of retry attempts on parse failure

        Returns:
            int: The parsed action (0-3)
        """
        for attempt in range(max_retries):
            action, parse_success = self.get_action()

            if parse_success:
                return action

            # Parse failed - retry with same input (no error feedback)
            print(f"Warning: Parse failure on attempt {attempt + 1}/{max_retries}")

        # Max retries exceeded - default to action 0
        print(f"Warning: Parse failed after {max_retries} attempts. Defaulting to action 0.")
        return 0

    def reset_trial(self, goal_star):
        """Reset simulator for new trial."""
        self.simulator.reset(goal_star)

    def run_trial(self, goal_star):
        """Run single trial (4 rounds)."""
        self.reset_trial(goal_star)

        # Initial state prompt
        state = self.simulator.get_state()
        user_prompt = format_state_prompt(state)
        self.conversation_history.append({"role": "user", "content": user_prompt})
        self.awaiting_action_prompt = True

        # 4 rounds
        for round_idx in range(4):
            action = self.get_action_with_retry()
            self.simulator.step(action)
            state = self.simulator.get_state()

            is_last_round = (round_idx == 3)
            is_complete = self.simulator.is_complete()
            feedback = format_feedback_prompt(action, state, is_complete, trial_end=is_last_round)
            self.conversation_history.append({"role": "user", "content": feedback})
            self.awaiting_action_prompt = not is_last_round

        return self.simulator.is_complete()

    def run_experiment(self, n_trials=20, goal_stars=None, specify_star=None):
        """Run multiple trials.

        Args:
            n_trials: Number of trials to run
            goal_stars: Optional list of goal stars for each trial (overrides other options)
            specify_star: Optional star to use for all trials (e.g., "Star_0").
                         If None, trials are split evenly across Star_0, Star_1, Star_2, Star_3
        """
        # Initialize conversation history with system prompt containing n_trials
        self.conversation_history = [{"role": "system", "content": get_system_prompt(n_trials)}]
        self.awaiting_action_prompt = False

        results = []
        for trial in tqdm(range(n_trials), desc="Running trials"):
            # Determine goal star:
            # 1. Use goal_stars list if provided
            # 2. Else use specify_star if provided
            # 3. Else divide N trials into 8 blocks, cycling through 4 stars twice
            if goal_stars:
                goal_star = goal_stars[trial]
            elif specify_star is not None:
                goal_star = specify_star
            else:
                goal_star = f"Star_{(trial // (n_trials // 8)) % 4}"

            success = self.run_trial(goal_star)
            results.append({'trial': trial, 'goal_star': goal_star, 'success': success})
        return {
            "trials": results,
            "conversation_history": self.conversation_history
        }
