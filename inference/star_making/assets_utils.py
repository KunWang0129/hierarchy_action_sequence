
ACTION_PROMPT_INSTRUCTION = "\n\nWhat action do you take next?"


class StarMakingRules:
    """Stores and manages star making rules for learning and transfer conditions."""

    def __init__(self):
        self.learning_rules = {
            'low': {(0,1): 'A', (2,3): 'B', (1,2): 'C', (3,0): 'D'},
            'high': {('A','B'): 'Star_0', ('C','D'): 'Star_1', ('B','C'): 'Star_2', ('D','A'): 'Star_3'}
        }
        self.transfer_rules = {
            'low': {(0,1): 'A', (2,3): 'B', (0,2): 'C', (3,1): 'D'},
            'high': {('A','B'): 'Star_0', ('C','B'): 'Star_1', ('D','B'): 'Star_2', ('D','A'): 'Star_3'}
        }

    def get_item(self, actions, rule_type='learning'):
        """Check if action sequence creates an item. Only checks positions [0,1] or [2,3]."""
        rules = self.learning_rules if rule_type == 'learning' else self.transfer_rules
        if len(actions) >= 2:
            return rules['low'].get((actions[0], actions[1]))
        return None

    def get_star(self, items, rule_type='learning'):
        """Check if item pair creates a star."""
        rules = self.learning_rules if rule_type == 'learning' else self.transfer_rules
        if len(items) == 2:
            return rules['high'].get((items[0], items[1]))
        return None


# Prompt formatting functions
def get_system_prompt(n_trials):
    """Generate system prompt with specified number of trials."""
    return f"""You are an expert agent in a star making game.

# Trials And Rounds
You will play {n_trials} trials. Each trial has 4 rounds where you choose an action from available actions (0, 1, 2, 3).

# State
Each round shows:
- Goal star: One of Star_0, Star_1, Star_2, Star_3
- Items: Created items (A, B, C, D or None)
- Actions: Actions taken so far in this trial

# Your Goal
Make the goal star by the end of each trial.

# Instruction
End every reply with the final line "action". 
Example: "(some thoughts...), thus we take action 2.
2"
"""


def format_state_prompt(state):
    """Format current state as prompt."""
    items = [item if item else 'None' for item in (state['items'] if state['items'] else [None, None])]
    return f"Current state:\nGoal star: {state['goal_star']}.\nItems: [{items[0]}, {items[1]}].\nActions: {state['actions']}."


def format_feedback_prompt(action, state, is_complete, trial_end=False):
    """Format feedback after action.

    Args:
        action: The action taken (0-3)
        state: Current state dict with goal_star, items, actions
        is_complete: Boolean indicating if goal star was successfully made
        trial_end: Boolean indicating if this is the last round

    Returns:
        Formatted feedback string
    """
    state_prompt = format_state_prompt(state)
    feedback = f"You press {action}\n{state_prompt}"

    if trial_end:
        if is_complete:
            feedback += "\n\nCongratulations! You successfully made the goal star!"
        else:
            feedback += "\n\nTrial ended. You did not make the goal star."

    return feedback


def parse_action(response):
    """Parse action from LLM response.

    Looks for any occurrence of digits 0-3 in the response and returns the last one found.
    This is flexible and works with various response formats.

    Args:
        response: Response dict or string from LLM

    Returns:
        tuple: (action, parse_success) where action is 0-3 and parse_success is bool
    """
    import re

    # Extract content from response dict or use string directly
    response_text = response.get("content", "") if isinstance(response, dict) else response

    # Find all occurrences of digits 0-3 in the response
    matches = re.findall(r"[0-3]", response_text)

    if matches:
        # If multiple actions found, pick the last one
        return int(matches[-1]), True

    return 0, False  # Default action if parsing fails
