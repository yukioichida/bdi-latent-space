import re

from typing import NamedTuple, Dict

from sources.bdi.models import NLIModel


class Plan(NamedTuple):
    task: str
    context: str
    body: list[str]


class PlanParser:

    def parse(self, plan_content: str) -> Plan:
        """
        Extract plan task, plan context and plan body from natural language plan
        :param plan_content: natural language plan
        """
        x = re.search(r"(?<=IF your goal is to)([\S\s]*?)(?:CONSIDERING)([\S\s]*)(?:THEN)([\S\s]*)", plan_content)
        groups = x.groups()
        if len(groups) == 3:
            task = self.preprocess_text(groups[0])
            context = self.preprocess_text(groups[1])
            body = [self.preprocess_text(text) for text in groups[2].split(',')]
            return Plan(task, context, body)
        else:
            print(f"Parse error: Plan {plan_content} malformed")
            return None

    @staticmethod
    def preprocess_text(txt: str):
        """
        Preprocessing text removing special tokens
        :param txt: text to be processed
        :return: preprocessed text
        """
        return txt.replace('\n', ' ').replace('\t', ' ').strip()


class PlanLibrary:

    def __init__(self, nli_model: NLIModel):
        """

        :param nli_model: natural language inference model to detect whether the belief base entails a plan context
        """
        self.nli_model = nli_model
        self.plans: Dict[str, Plan] = dict()  # dict key with goal -> plan
        self.subtasks = []

    def load_plans(self, input_plans: list[Plan]):
        for plan in input_plans:
            self.plans[plan.task] = plan
            self.subtasks.append(plan.task)

    def select_plan(self, belief_base: str) -> Plan:
        """
        Find a plan
        :param belief_base:
        :return: List of atomic actions to agent execute it
        """
        candidate_plans = []

        # TODO: plans can be processed on GPU in parallel by generating a single tensor containing all plans
        for plan in self.plans:
            entailment, confidence = self.nli_model.entails(p=belief_base, h=plan.plan_context)
            if entailment:
                candidate_plans.append((confidence, plan))

        # verify if there is a candidate plan before sort it
        if len(candidate_plans) > 0:
            candidate_plans.sort(key=lambda x: x[0])
            return candidate_plans[0][0]
        else:
            return None

    def get_actions(self, plan: Plan):
        """
        Drill down sub-task to collect all actions from a hierarchical plan
        :return:
        """
        plan_actions = []
        for term in plan.body:
            self._dfs(term, plan_actions)
        return plan_actions

    def _dfs(self, term, actions) -> str:
        # TODO: ver como lidar com goals e ações com nomes nomes iguais pois causa loop infinito
        if term not in self.subtasks:
            actions.append(term)  # is an action
            return
        elif term in self.plans:
            # plan that satisfies the goa
            subtask = self.plans[term]
            for term in subtask.body:
                self._dfs(term, actions)
        else:
            print(f"Term {term} is not an action or plan")
            return
