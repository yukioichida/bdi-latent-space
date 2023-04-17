from sources.bdi.models import NLIModel
from sources.bdi.plans import PlanLibrary


class BDIAgent:

    def __init__(self, nli_model: NLIModel):
        self.plan_library = PlanLibrary(nli_model)


    def plan(self, observation: str):
        belief_base = observation
        plan = self.plan_library.select_plan(belief_base)

        if plan is None:
            print("There is no candidate plan")
        else:
            plan_actions = self.plan_library.get_actions(plan)
            return plan_actions