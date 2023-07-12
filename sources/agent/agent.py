from sources.bdi_components.plans import Plan, PlanLibrary
from sources.bdi_components.belief import BeliefBase
from sources.bdi_components.inference import NLIModel


class BDIAgent:

    def __init__(self, plan_files: str, nli_model: NLIModel):
        self.belief_base = BeliefBase()
        self.plan_library = PlanLibrary(plan_files)
        self.nli_model = nli_model

    def perceive_and_act(self, goal: str, observations_perceived: list[str]) -> list[str]:
        """
        Perceive new observations (belief addition) and goal (goal addition event), deliberates over them and act.
        :param goal: Goal perceived from environment
        :param observations_perceived: Observations perceived from environment
        :return: set of actions contained in the plan body whether there is a candidate plan
        """
        candidate_plan = self.select_options(goal, observations_perceived)
        if candidate_plan is not None:
            # decompose plan to get actions
            actions = self.plan_library.get_actions(candidate_plan)
            return actions
        else:
            return []

    def select_options(self, goal: str, new_observations: list[str]) -> Plan:
        """

        :param goal: Goal received - analogous to goal addition event
        :param new_observations: observation perceived by the agent
        :return: Plan to be executed whether there is a candidate one
        """
        candidate_plans = []

        # TODO: use all memory instead of relying only on the current
        self.belief_base.belief_addition(new_observations)  # updating belief base
        current_beliefs = self.belief_base.get_current_beliefs()

        # TODO: plans can be processed on GPU in parallel by generating a multi-dimensional tensor containing all plans (BATCHED inference)
        all_plans = self.plan_library.plans
        for trigger_condition, plan in all_plans.items():
            # the selected plan should have the same goal
            same_goal, _ = self.nli_model.check_goal(goal, plan.task)
            entailment, confidence = self.nli_model.check_context_entailment(beliefs=current_beliefs,
                                                                             plan_contexts=plan.context)
            if entailment and same_goal:
                candidate_plans.append((confidence, plan))

        # verify if there is a candidate plan before sort it
        if len(candidate_plans) > 0:
            candidate_plans.sort(key=lambda x: x[0])
            return candidate_plans[0][1]
        else:
            return None
