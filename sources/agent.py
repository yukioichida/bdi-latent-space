from typing import Callable

from sources.bdi_components.belief import BeliefBase, State
from sources.bdi_components.inference import NLIModel
from sources.bdi_components.plans import Plan, PlanLibrary


class BDIAgent:

    def __init__(self, plan_library: PlanLibrary, nli_model: NLIModel):
        """
        BDI agent main class
        :param plan_library: Structure storing the plans to be executed by the agent.
        :param nli_model: Model to make logical inference over natural language information
        """
        self.belief_base = BeliefBase()
        self.plan_library = plan_library
        self.nli_model = nli_model
        self.event_trace = []
        self.action_trace = []

    def act(self,
            current_state: State,
            step_function: Callable[[str], State]) -> State:
        """
        Perceive new observations and goal, deliberates, and act.
        :param current_state: Current agent state in environment
        :param step_function: function that executes an step in the environment and returns the new state
        """
        # root plan
        visited_events = []
        plan_state = self.reasoning_cycle(current_state, current_state.goal, visited_events, step_function, 0)
        return plan_state

    def reasoning_cycle(self,
                        state: State,
                        triggering_event: str,
                        visited_events: list[str],
                        step_function: Callable[[str], State],
                        depth: int) -> State:
        """
        Reasoning cycle based on BDI event-driven plan selection.
        This implementation uses a deep-first search to decompose subtasks and execute actions.
        :param state: current state
        :param triggering_event: event perceived in the reasoning cycle, which can be a goal addition or an action
        :param visited_events: track events already visited
        :param step_function: function that calls the environment.step() and returns the environment state updated
        :param depth: depth number
        :return: state modified by the action executed
        """

        # TODO: rever termo "state", talvez mudar para "belief base"
        plan = self.get_plan(state, triggering_event)
        if plan is not None:
            self.event_trace.append(triggering_event)
            plan_body = plan.body
            current_state = state
            for breadth, event in enumerate(plan_body):
                if event in self.plan_library.plans.keys():  # goal addition, proceed in decomposition
                    event_id = f"{depth}-{breadth}-{event}"
                    if event_id not in visited_events:
                        visited_events.append(event_id)
                        current_state = self.reasoning_cycle(current_state, event, visited_events, step_function,
                                                             depth + 1)
                        if current_state.error:
                            break  # action failure must raise a plan failure
                else:  # event in current_state.valid_actions:  # action, primitive task
                    self.action_trace.append(event)
                    current_state = step_function(event)
            return current_state
        else:
            # print(f"No plan found for event ({triggering_event}) with beliefs ({state.sentence_list()})")
            return State(error=True,
                         score=state.score,
                         goal=state.goal,
                         observation=state.observation,
                         look=state.look,
                         valid_actions=state.valid_actions)

    def get_plan(self, state: State, triggering_event: str) -> Plan:
        """
        Find a Plan compatible with the current belief base
        :param state: current environment state perceived
        :param triggering_event: plan triggering event
        :return: Plan to be executed whether there is a candidate one
        """
        candidate_plans = []
        # get plans triggered by the event (new goal)
        all_plans = self.plan_library.plans[triggering_event]
        # print(f"Goal {triggering_event} - {all_plans}")
        all_beliefs = state.sentence_list()
        # TODO: remove this loop to avoid unnecessary nli model calls and do the inference
        for plan in all_plans:
            if len(plan.context) > 0:
                entailment, confidence = self.nli_model.check_context_entailment(beliefs=all_beliefs,
                                                                                 plan_contexts=plan.context)
            else:
                entailment, confidence = True, 1  # it assumes True whether a plan does not have context

            if entailment:
                candidate_plans.append((confidence, plan))

        if len(candidate_plans) > 0:
            candidate_plans.sort(key=lambda x: x[0])  # order by confidence
            return candidate_plans[-1][1]  # plan with the highest confidence score
        else:
            return None
