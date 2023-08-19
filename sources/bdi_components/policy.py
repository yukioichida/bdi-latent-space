
from sources.bdi_components.belief import State
from sources.drrn.drrn_agent import DRRN_Agent
from sources.drrn.util import sanitizeObservation

"""
class DefaultPolicy:

    def act(self, state: State, available_actions: list[str]) -> list[str]:
        pass
"""

class DRRNDefaultPolicy:

    def __init__(self, spm_path: str, trained_model_path: str, trained_model_id: str):
        """
        Wrapper class for the defaultpolicy equipped with a DRRN agent
        :param spm_path: DRRN SentencePiece model
        """
        policy = DRRN_Agent(spm_path)
        policy.load(trained_model_path, trained_model_id)
        self.policy = policy

    def act(self, observation, goal, look, inventory, available_actions: list[str]) -> list[str]:
        sanitize_obs = sanitizeObservation(obsIn=observation, taskDesc=goal)
        state = self.policy.build_state(obs=sanitize_obs,
                                        inv=inventory,
                                        look=look)
        valid_ids = self.policy.encode(available_actions)
        action_ids, action_idxs, q_values = self.policy.act([state], [valid_ids], sample=False)
        action_idx = action_idxs[0]
        # Sanitize input
        action_str = available_actions[action_idx]
        action_str = action_str.lower().strip()
        return [action_str]