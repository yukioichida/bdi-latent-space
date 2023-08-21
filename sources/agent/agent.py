from typing import List, Callable

from sources.bdi_components.belief import BeliefBase, State
from sources.bdi_components.inference import NLIModel
from sources.bdi_components.plans import Plan, PlanLibrary


class BDIAgent:

    def __init__(self, plan_library: PlanLibrary, nli_model: NLIModel):
        self.belief_base = BeliefBase()
        self.plan_library = plan_library
        self.nli_model = nli_model

    def act(self,
            current_state: State,
            step_function: Callable[[str], State]) -> list[str]:
        """
        Perceive new observations and goal, deliberates over them and act.
        :param current_state: Current agent state in environment
        :param step_function: function that executes an step in the environment and returns the new state
        :return: set of actions contained in the plan body whether there is a candidate plan
        """

        # root plan
        visited_events = []
        self.reasoning_cycle(current_state, current_state.goal, visited_events, step_function)

    def reasoning_cycle(self,
                        state: State,
                        triggering_event: str,
                        visited_events: list[str],
                        step_function: Callable[[str], State]) -> State:

        plan = self.get_plan(state, triggering_event)

        if plan is not None:
            plan_body = plan.body
            current_state = state
            for event in plan_body:
                print(f"-> Event {event}")
                if event in self.plan_library.plans.keys():
                    # it is a subgoal
                    if event not in visited_events:
                        print(f"drill down to event: {event}")
                        visited_events.append(event)
                        current_state = self.reasoning_cycle(current_state, event, visited_events, step_function)
                elif event in current_state.valid_actions:  # is an action
                    # env.step
                    print(f"Executing action {event}")
                    current_state = step_function(event)
                else:
                    # invalid token
                    print(f"event {event} not recognized as a subgoal neither an action")
                    return None # automatically break as a plan failure
            return current_state

        else:
            print(f"No plan found for event ({triggering_event}) with beliefs ({state.sentence_list()})")

    def get_plans_from_event(self, triggering_event: str) -> List[Plan]:
        return self.plan_library.plans[triggering_event]

    def get_plan(self, state: State, triggering_event: str) -> Plan:
        """

        :param state: current environment state perceived
        :param triggering_event: plan triggering event
        :return: Plan to be executed whether there is a candidate one
        """
        candidate_plans = []
        all_plans = self.plan_library.plans[triggering_event]
        all_beliefs = state.sentence_list()
        for plan in all_plans:
            if len(plan.context) > 0:
                entailment, confidence = self.nli_model.check_context_entailment(beliefs=all_beliefs,
                                                                                 plan_contexts=plan.context)
            else:
                entailment, confidence = True, 1  # assumes True whether a plan does not have context

            if entailment:
                candidate_plans.append((confidence, plan))

        if len(candidate_plans) > 0:
            candidate_plans.sort(key=lambda x: x[0])  # order by confidence
            return candidate_plans[0][1]  # plan with the highest confidence score
        else:
            return None
