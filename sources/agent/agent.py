from typing import List, Callable
import networkx as nx

from sources.bdi_components.belief import BeliefBase, State
from sources.bdi_components.inference import NLIModel
from sources.bdi_components.plans import Plan, PlanLibrary


class BDIAgent:

    def __init__(self, plan_library: PlanLibrary, nli_model: NLIModel):
        self.belief_base = BeliefBase()
        self.plan_library = plan_library
        self.nli_model = nli_model
        self.plan_tree = nx.Graph()

    def act(self,
            current_state: State,
            step_function: Callable[[str], State]):
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
        """
        Reasoning cycle based on BDI event-driven plan selection.
        This implementation uses a deep-first search to decompose subtasks and execute actions.
        :param state: current state
        :param triggering_event: event perceived in the reasoning cycle, which can be a goal addition or an action
        :param visited_events: track events already visited
        :param step_function: function that calls the environment.step() and returns the environment state updated
        :return: state modified by the action executed
        """

        # TODO: rever termo "state", talvez mudar para "belief base"
        # TODO: usar networkx para pegar a árvore de planos executado
        plan = self.get_plan(state, triggering_event)
        self.plan_tree.add_node(triggering_event)
        if plan is not None:
            plan_body = plan.body
            current_state = state
            for event in plan_body:
                if event in self.plan_library.plans.keys():  # goal addition, proceed in decomposition
                    if event not in visited_events:
                        self.plan_tree.add_edge(triggering_event, event)
                        visited_events.append(event)
                        current_state = self.reasoning_cycle(current_state, event, visited_events, step_function)
                elif event in current_state.valid_actions:  # action, primitive task
                    self.plan_tree.add_edge(triggering_event, event)
                    current_state = step_function(event)  # executes the action and receives the updated state
                else:  # invalid token
                    print(f"Event {event} is not a subgoal nor an action.")
                    return None  # automatically break as a plan failure
            return current_state
        else:
            print(f"No plan found for event ({triggering_event}) with beliefs ({state.sentence_list()})")

    def get_plan(self, state: State, triggering_event: str) -> Plan:
        """
        Find a Plan compatible with the current belief base
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
