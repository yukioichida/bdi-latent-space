from typing import NamedTuple


class State(NamedTuple):
    """
    Environment state perceived by the agent
    Means-Ends Reasoning - pag 19
    ... The agent’s current beliefs about the *state* of the environment. ...
    """
    goal: str = ""  # main goal
    task_description: str = ""  # task description
    observation: str = ""  # observation perceived
    look: list[str] = []  # objects that agent are seeing in the current state
    inventory: str = ""  # agent's inventory
    valid_actions: list[str] = []  # valid actions that the agent can perform in the current state
    score: float = 0  # reward received by the environment
    error: bool = False  # flag indicating an error occurred by the action executed
    completed: bool = False # whether the agent could finish the task
    metadata: dict = {} # additional info collected by the agent

    def sentence_list(self):
        return [self.inventory] + self.look


class BeliefBase:

    def __init__(self):
        """
        Agent's Belief base that contains perceived environment state
        """
        self.memory = []

    def belief_addition(self, new_state: State):
        self.memory.append(new_state)

    def get_current_beliefs(self) -> State:
        """
        Retrieve current state stored in the belief base
        :return: List of sentences corresponding to the current state perceived in the belief base
        """
        return self.memory[-1] if len(self.memory) > 0 else []
