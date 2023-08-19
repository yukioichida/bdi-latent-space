from sources.bdi_components.belief import BeliefBase, State
from sources.bdi_components.inference import NLIModel
from sources.bdi_components.plans import Plan, PlanLibrary


class BDIAgent:

    def __init__(self, plan_file: str, nli_model: NLIModel):
        self.belief_base = BeliefBase()
        self.plan_library = PlanLibrary(plan_file)
        self.nli_model = nli_model

    def act(self, current_state: State, available_actions: list[str]) -> list[str]:
        """
        Perceive new observations (belief addition) and goal (goal addition event), deliberates over them and act.
        :param current_state: Current agent state in environment
        :param available_actions: valid actions to be selected given the current environment state
        :return: set of actions contained in the plan body whether there is a candidate plan
        """
        candidate_plan = self.select_options(current_state)
        if candidate_plan is not None:
            # decompose plan to get actions
            actions = self.plan_library.get_actions(candidate_plan, valid_actions=available_actions)
            return actions
        else:
            # if there are no plans, use the default policy to decide which action should be selected
            #return self.fallback_policy.act(state=current_state, available_actions=available_actions)
            return []

    def select_options(self, state: State) -> Plan:
        """

        :param state: current environment state perceived
        :return: Plan to be executed whether there is a candidate one
        """
        candidate_plans = []

        # TODO: use all memory instead of relying only on the current
        self.belief_base.belief_addition(state)  # updating belief base
        current_beliefs = self.belief_base.get_current_beliefs()

        # TODO: plans can be processed on GPU in parallel by generating a multi-dimensional tensor containing all plans (BATCHED inference)
        all_plans = self.plan_library.plans
        for trigger_condition, plan in all_plans.items():
            # the selected plan should have the same goal
            same_goal = self.nli_model.check_goal(state.goal, plan.task)
            entailment, confidence = self.nli_model.check_context_entailment(beliefs=current_beliefs.sentence_list(),
                                                                             plan_contexts=plan.context)
            if entailment and same_goal:
                candidate_plans.append((confidence, plan))

        # verify if there is a candidate plan before sort it
        if len(candidate_plans) > 0:
            candidate_plans.sort(key=lambda x: x[0])
            return candidate_plans[0][1]
        else:
            return None
