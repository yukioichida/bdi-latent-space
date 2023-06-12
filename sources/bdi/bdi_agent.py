from sources.bdi.models import NLIModel
from sources.bdi.plans import PlanLibrary

# TODO: watch out for cyclic dependencies
from sources.drrn.drrn_agent import DRRN_Agent
from sources.drrn.util import sanitizeObservation

from typing import NamedTuple, Dict

from sources.bdi.entities import BeliefBase, Plan


class DefaultPolicy:

    def act(self, belief_base: BeliefBase, available_actions: list[str]) -> list[str]:
        pass


class DRRNDefaultPolicy(DefaultPolicy):

    def __init__(self, spm_path: str, trained_model_path: str, trained_model_id: str):
        """
        Wrapper class for the defaultpolicy equipped with a DRRN agent
        :param spm_path: DRRN SentencePiece model
        """
        policy = DRRN_Agent(spm_path)
        policy.load(trained_model_path, trained_model_id)
        self.policy = policy

    def act(self, belief_base: BeliefBase, available_actions: list[str]) -> list[str]:
        observation = sanitizeObservation(obsIn=belief_base.observation, taskDesc=belief_base.goal)
        state = self.policy.build_state(obs=observation,
                                        inv=belief_base.inventory,
                                        look=belief_base.look_around)
        valid_ids = self.policy.encode(available_actions)
        action_ids, action_idxs, q_values = self.policy.act([state], [valid_ids], sample=False)
        action_idx = action_idxs[0]
        # Sanitize input
        action_str = available_actions[action_idx]
        action_str = action_str.lower().strip()
        return [action_str]


class BDIAgent:

    def __init__(self, nli_model: NLIModel, default_policy: DefaultPolicy, plan_file: str):
        self.plan_library = PlanLibrary(nli_model, plan_file)
        self.default_policy = default_policy
        self.num_selected_plans = 0
        self.num_default_actions = 0

    def act(self, goal: str, observation: str, inventory: str, look_around: str, valid_actions: list[str]) -> list[str]:
        belief_base = BeliefBase(goal=goal, observation=observation, inventory=inventory, look_around=look_around)
        self.belief_base = belief_base
        plan = self.plan_library.select_plan(belief_base)
        if plan is None:
            self.num_default_actions += 1
            print("There is no candidate plan. Using a default policy")
            default_actions = self.default_policy.act(belief_base, valid_actions)
            return default_actions
        else:
            self.num_selected_plans += 1
            print(f"plan selected {plan.idx} - belief base {belief_base.string_representation()} - plan header {plan.plan_header()}")
            plan_actions = self.plan_library.get_actions(plan, valid_actions)
            return plan_actions
