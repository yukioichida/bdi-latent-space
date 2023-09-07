from sources.bdi_components.belief import State
from sources.drrn.drrn_agent import DRRN_Agent
from sources.drrn.util import sanitizeObservation


class DefaultPolicy:

    def select_action(self, state: State) -> str:
        pass


class DRRNDefaultPolicy(DefaultPolicy):

    def __init__(self, spm_path: str, trained_model_path: str):
        """
        Wrapper class for the defaultpolicy equipped with a DRRN agent
        :param spm_path: DRRN SentencePiece model
        """
        policy = DRRN_Agent(spm_path)
        policy.load(trained_model_path)
        self.policy = policy

    def select_action(self, state: State) -> str:
        sanitize_obs = sanitizeObservation(obsIn=state.observation, taskDesc=state.task_description)
        state = self.policy.build_state(obs=sanitize_obs,
                                        inv=state.inventory,
                                        look=state.look)
        valid_ids = self.policy.encode(state.valid_actions)
        action_ids, action_idxs, q_values = self.policy.act([state], [valid_ids], sample=False)
        action_idx = action_idxs[0]
        # Sanitize input
        action_str = state.valid_actions[action_idx]
        action_str = action_str.lower().strip()
        return action_str
