
import json
import re
from pathlib import Path

ACTION_PROMPT_INSTRUCTION = "\n\nWhat action do you take next?"

ACTION_ENCODE_1 = {
    "I": 0,
    "O": 1,
    "U": 2,
    "P": 3
}

ACTION_ENCODE_2 = {
    "U": 0,
    "P": 1,
    "O": 2,
    "I": 3
}

LEARNING_ORDERS = [
    "Star_0",
    "Star_1",
    "Star_2",
    "Star_3",
    "Star_1",
    "Star_3",
    "Star_0",
    "Star_2"
]

class StarMakingRules:
    """Stores and manages star making rules for learning and transfer conditions."""

    def __init__(self, encoding=1):
        self.encoding = ACTION_ENCODE_1 if encoding == 1 else ACTION_ENCODE_2
        # Flattened rules: 4-action sequences directly produce stars
        self.learning_rules = {
            (0,1,2,3): 'Star_0',  # (0,1)→A + (2,3)→B → Star_0
            (1,2,3,0): 'Star_1',  # (1,2)→C + (3,0)→D → Star_1
            (2,3,1,2): 'Star_2',  # (2,3)→B + (1,2)→C → Star_2
            (3,0,0,1): 'Star_3'   # (3,0)→D + (0,1)→A → Star_3
        }
        self.transfer_low_rules = {
            (0,1,2,3): 'Star_0',  # (0,1)→A + (2,3)→B → Star_0
            (0,2,3,1): 'Star_1',  # (0,2)→C + (3,1)→D → Star_1
            (2,3,0,2): 'Star_2',  # (2,3)→B + (0,2)→C → Star_2
            (3,1,0,1): 'Star_3'   # (3,1)→D + (0,1)→A → Star_3
        }

        self.transfer_high_rules = {
            (0,1,2,3): 'Star_0',  # (0,1)→A + (2,3)→B → Star_0
            (1,2,2,3): 'Star_1',  # (1,2)→C + (2,3)→B → Star_1
            (3,0,2,3): 'Star_2',  # (3,0)→D + (2,3)→B → Star_2
            (3,0,0,1): 'Star_3'   # (3,0)→D + (0,1)→A → Star_3
        }
        self.transfer_rules = self.transfer_low_rules

    def set_transfer_rule_type(self, rule_type):
        """Choose which transfer rule table to use."""
        mapping = {
            "transfer_low": self.transfer_low_rules,
            "transfer_high": self.transfer_high_rules,
        }
        if rule_type not in mapping:
            raise ValueError(f"Unsupported transfer rule type: {rule_type}")
        self.transfer_rules = mapping[rule_type]

    def encode_action(self, action):
        """Convert numeric action (0,1,2,3) to letter (U,I,O,P)"""
        decode_map = {v: k for k, v in self.encoding.items()}
        return decode_map[action]

    def decode_action(self, action):
        """Convert letter action (U,I,O,P) to numeric (0,1,2,3)"""
        return self.encoding[action.upper()]

    def get_star(self, actions, rule_type='learning'):
        """Check if 4-action sequence creates a star."""
        rules = self.learning_rules if rule_type == 'learning' else self.transfer_rules
        if len(actions) == 4:
            return rules.get(tuple(actions))
        return None



class ConversationHistoryManager:
    """Manages conversation history and state for LLM interactions.

    This class centralizes conversation history management to avoid code duplication
    across single-agent and batch-agent implementations.
    """

    def __init__(self):
        """Initialize empty conversation history."""
        self.history = []
        self.awaiting_action = False

    def initialize(self, system_prompt):
        """Initialize conversation with system prompt.

        Args:
            system_prompt: System message content to start the conversation
        """
        self.history = [{"role": "system", "content": system_prompt}]
        self.awaiting_action = False

    def initialize_with_in_context(self, n_trials, json_path=None):
        """Initialize conversation with in-context examples from JSON file.

        This loads a pre-existing conversation history that demonstrates the game mechanics,
        useful for transfer learning scenarios where the agent needs prior knowledge.

        Args:
            n_trials: Number of trials for current experiment (updates system prompt)
            json_path: Optional path to in_context.json. If None, uses default location
                      (same directory as this file)
        """
        if json_path is None:
            json_path = Path(__file__).parent / "in_context.json"

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Load conversation history from JSON
        self.history = data["conversation_history"].copy()

        # Update system prompt to reflect current n_trials
        if self.history and self.history[0].get("role") == "system":
            # Extract the current system prompt template and update n_trials
            system_content = self.history[0]["content"]
            # Replace the number in "You will play X trials" with current n_trials
            updated_content = re.sub(
                r'You will play \d+ trials',
                f'You will play {n_trials} trials',
                system_content
            )
            self.history[0]["content"] = updated_content

        # Set awaiting_action to False (in-context ends with a successful trial)
        self.awaiting_action = False

    def add_message(self, role, content):
        """Add a message to the conversation history.

        Args:
            role: Message role ("user", "assistant", or "system")
            content: Message content
        """
        self.history.append({"role": role, "content": content})

    def prepare_for_llm(self, action_instruction=None):
        """Prepare conversation for LLM call, optionally appending action instruction.

        Non-mutating: returns a copy with modifications, original history unchanged.

        Args:
            action_instruction: Optional instruction to append to last user message
                              (e.g., ACTION_PROMPT_INSTRUCTION)

        Returns:
            List of message dicts ready for LLM
        """
        # If not awaiting action or no instruction provided, return as-is
        if not (self.awaiting_action and action_instruction and self.history):
            return self.history

        # Only append to user messages
        last_message = self.history[-1]
        if last_message.get("role") != "user":
            return self.history

        # Create non-mutating copy with instruction appended
        prepared_history = list(self.history[:-1])
        last_copy = last_message.copy()
        last_copy["content"] = f"{last_copy['content']}{action_instruction}"
        prepared_history.append(last_copy)
        return prepared_history

    def set_awaiting_action(self, value):
        """Set the awaiting_action flag.

        Args:
            value: Boolean indicating if we're awaiting an action from LLM
        """
        self.awaiting_action = value

    def get_history(self):
        """Get the current conversation history.

        Returns:
            List of message dicts
        """
        return self.history


# Prompt formatting functions
def get_system_prompt(n_trials):
    """Generate system prompt with specified number of trials."""
    return f"""You are an expert agent in a star making game.

# Trials And Rounds
You will play {n_trials} trials. Each trial has 4 rounds where you choose an action from available actions (U, I, O, P).

# State
Each round shows:
- Goal star: One of Star_0, Star_1, Star_2, Star_3
- Actions: Actions taken so far in this trial

# Your Goal
Make the goal star by the end of each trial.

# Instruction
End every reply with the final line "action".
Example: "(some thoughts...), thus we take action O.
O"
"""


def format_state_prompt(state, rules):
    """Format current state as prompt."""
    actions = [rules.encode_action(a) for a in state['actions']]
    return f"Current state:\nGoal star: {state['goal_star']}.\nActions: {actions}."


def format_feedback_prompt(action, state, is_complete, rules, trial_end=False):
    """Format feedback after action.

    Args:
        action: The action taken (0-3)
        state: Current state dict with goal_star, items, actions
        is_complete: Boolean indicating if goal star was successfully made
        rules: StarMakingRules instance for encoding
        trial_end: Boolean indicating if this is the last round

    Returns:
        Formatted feedback string
    """
    state_prompt = format_state_prompt(state, rules)
    action_letter = rules.encode_action(action)
    feedback = f"You press {action_letter}\n{state_prompt}"

    if trial_end:
        if is_complete:
            feedback += "\n\nCongratulations! You successfully made the goal star!"
        else:
            feedback += "\n\nTrial ended. You did not make the goal star."

    return feedback


def parse_action(response, rules):
    """Parse action from LLM response.

    Looks for any occurrence of action letters (U,I,O,P) in the response and returns the last one found.
    This is flexible and works with various response formats.

    Args:
        response: Response dict or string from LLM
        rules: StarMakingRules instance for decoding

    Returns:
        tuple: (action, parse_success) where action is 0-3 and parse_success is bool
    """
    import re

    # Extract content from response dict or use string directly
    response_text = response.get("content", "") if isinstance(response, dict) else response

    # Find all occurrences of action letters in the response (case insensitive)
    matches = re.findall(r"[UIOPuiop]", response_text)

    if matches:
        # If multiple actions found, pick the last one
        return rules.decode_action(matches[-1]), True

    return 0, False  # Default action if parsing fails
