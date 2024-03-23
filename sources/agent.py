import typing
from typing import Callable, Deque
import collections
from copy import deepcopy
from enum import Enum
from typing import NamedTuple

from sources.bdi_components.belief import BeliefBase, State
from sources.bdi_components.inference import NLIModel
from sources.bdi_components.plans import Plan, PlanLibrary


class EventType(Enum):
    GOAL_ADDITION = 1
    BELIEF_UPDATE = 2


class Event(NamedTuple):
    type: EventType
    content: str = ""


class IntentionType(Enum):
    END = 1
    ACTION = 2
    SUB_GOAL = 3


class Intention(NamedTuple):
    type: IntentionType
    content: str = ""


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
        self.event_queue = Deque[Event]()
        self.intention_structure = Deque[Intention]()
        self.event_trace = []
        self.action_trace = []

    def perceive(self,
                 goal: str,
                 state: State,
                 step_function: Callable[[str], State]):

        # update belief base
        self._update_beliefs(state)

        # Plan Adoption for the main goal
        init_event = Event(EventType.GOAL_ADDITION, goal)
        self.event_queue.appendleft(init_event)

        # BDI reasoning cycle
        self.reasoning_cycle(step_function=step_function)

    def _update_beliefs(self, state):
        self.belief_base.belief_update(state)
        self.current_state = state
        return state

    def reasoning_cycle(self, step_function: Callable[[str], State]):
        while True:
            has_candidate_plan = self._select_plan()
            if len(self.intention_structure) > 0:
                self._execute_intention(step_function=step_function)
            else:
                # all intentions were already executed
                break

    def _select_plan(self) -> bool:
        # SELECT PLAN
        if len(self.event_queue) > 0:
            event = self.event_queue.popleft()
            option = self.get_plan(event.content)
            # if there is a plan, include its body into the intention structure
            if option is not None:
                self.event_trace.append(event)
                for step in reversed(option.body):
                    int_type = IntentionType.SUB_GOAL if step in self.plan_library.plans.keys() else IntentionType.ACTION
                    intention = Intention(content=step, type=int_type)
                    self.intention_structure.appendleft(intention)
                return True
            else:
                print(f"No plans for goal: {event.content}")
                # TODO: fallback rl
                return False

    def _execute_intention(self, step_function: Callable[[str], State]):
        current_intention = self.intention_structure.popleft()
        if current_intention.type == IntentionType.SUB_GOAL:
            # includes a new goal addition event to deal with the subgoal
            self.event_queue.appendleft(Event(EventType.GOAL_ADDITION, current_intention.content))
        elif current_intention.type == IntentionType.ACTION:
            # execute action and retrieve the updated state
            self.action_trace.append(current_intention.content)
            updated_state = step_function(current_intention.content)  # TODO: include belief base update event
            self._update_beliefs(updated_state)
        else:
            raise Exception(f"Invalid Intention Type:{current_intention}")

    def get_plan(self, triggering_event: str) -> typing.Optional[Plan]:
        """
        Find a Plan compatible with the current belief base given an triggering event
        :param triggering_event: plan triggering event
        :return: Plan to be executed whether there is a candidate one
        """
        candidate_plans = []
        # get plans triggered by the event (goal addition)
        if triggering_event in self.plan_library.plans:
            all_plans = self.plan_library.plans[triggering_event]
            current_beliefs = self.belief_base.get_current_beliefs()
            # TODO: remove this loop to avoid unnecessary nli model calls and do the inference
            for plan in all_plans:
                if len(plan.context) > 0:
                    entailment, confidence = self.nli_model.check_context_entailment(beliefs=current_beliefs.beliefs,
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
