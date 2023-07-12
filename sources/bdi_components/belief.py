class BeliefBase:

    def __init__(self):
        self.memory = []

    def belief_addition(self, observations: list[str]):
        self.memory.append(observations)

    def get_current_beliefs(self) -> list[str]:
        """
        Retrieve current beliefs
        :return: List of sentences corresponding to the current beliefs
        """
        return self.memory[-1] if len(self.memory) > 0 else []
