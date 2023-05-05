from typing import NamedTuple


class BeliefBase(NamedTuple):
    """
    Belief base information
    """
    observation: str
    goal: str
    inventory: str
    look_around: str

    def string_representation(self):
        # TODO: workaround, look around termina com um \n no final da string
        if self.look_around[:-1] == self.observation:
            return self.goal + ' ' + self.observation + ' ' + self.inventory
        else:
            return self.goal + ' ' + self.observation + ' ' + self.inventory + ' ' + self.look_around


class Plan(NamedTuple):
    task: str
    context: str
    body: list[str]
    idx: int

    def plan_header(self):
        return 'if your task is to ' + self.task + ' and ' + self.context
