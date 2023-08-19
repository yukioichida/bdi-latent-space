from typing import NamedTuple, Dict


class State(NamedTuple):
    """
    Current beliefs seeing by the agent
    Means-Ends Reasoning - pag 19
    ... The agent’s current beliefs about the *state* of the environment. ...
    """
    goal: str
    observation: str
    look: list[str] # objects that agent are seeing in the current state
    inventory: str

    def sentence_list(self):
        return [self.inventory] + self.look


class BeliefBase:

    def __init__(self):
        self.memory = []

    def belief_addition(self, new_state: State):
        self.memory.append(new_state)

    def get_current_beliefs(self) -> State:
        """
        Retrieve current beliefs
        :return: List of sentences corresponding to the current beliefs
        """
        return self.memory[-1] if len(self.memory) > 0 else []
