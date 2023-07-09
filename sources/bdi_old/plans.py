import re

from typing import NamedTuple, Dict

from sources.bdi_old.models import NLIModel
# TODO: botar em um pacote separado a classe abaixo para evitar dependências circulares
from sources.bdi_old.entities import BeliefBase, Plan



class PlanParser:

    def parse(self, plan_content: str, idx: int) -> Plan:
        """
        Extract plan task, plan context and plan body from natural language plan
        :param idx: plan identifier
        :param plan_content: natural language plan
        :return Plan parsed
        """
        x = re.search(r"(?<=IF your goal is to)([\S\s]*?)(?:CONSIDERING)([\S\s]*)(?:THEN)([\S\s]*)", plan_content)
        groups = x.groups()
        if len(groups) == 3:
            task = self.preprocess_text(groups[0])
            context = self.preprocess_text(groups[1])
            body = [self.preprocess_text(text) for text in groups[2].split(',')]
            return Plan(task, context, body, idx)
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


def load_plans_from_file(file: str):
    """
    Load plans from a .plan file
    :param file: path containing the .plan file
    :return: list of plans
    """
    parser = PlanParser()
    with open(file) as f:
        plan_str = f.read()

    plans = plan_str.split("\n;\n")
    return [parser.parse(plan, idx) for idx, plan in enumerate(plans)]


class PlanLibrary:

    def __init__(self, nli_model: NLIModel, plans_file: str):
        """

        :param nli_model: natural language inference model to detect whether the belief base entails a plan context
        :param plans_file: File containing the natural language plans
        """
        self.nli_model = nli_model
        self.plans: Dict[str, Plan] = dict()  # dict key with goal -> plan
        self.subtasks = []
        plans_from_file = load_plans_from_file(plans_file)
        self.load_plans(plans_from_file)

    def load_plans(self, input_plans: list[Plan]):
        for plan in input_plans:
            self.plans[plan.task] = plan
            self.subtasks.append(plan.task)

    def select_plan(self, belief_base: BeliefBase) -> Plan:
        """
        Find a plan
        :param belief_base:
        :return: List of atomic actions to agent execute it
        """
        candidate_plans = []

        # TODO: plans can be processed on GPU in parallel by generating a single tensor containing all plans
        for trigger_condition, plan in self.plans.items():
            # the selected plan should have the same goal
            same_goal, _ = self.nli_model.entails(p=belief_base.goal, h=plan.task)
            entailment, confidence = self.nli_model.entails(p=belief_base.observation, h=plan.context)
            if entailment and same_goal:
                candidate_plans.append((confidence, plan))

        # verify if there is a candidate plan before sort it
        if len(candidate_plans) > 0:
            candidate_plans.sort(key=lambda x: x[0])
            return candidate_plans[0][1]
        else:
            return None

    def get_actions(self, plan: Plan, valid_actions: list[str]):
        """
        Drill down sub-task to collect all actions from a hierarchical plan
        :return:
        """
        plan_actions = []
        for term in plan.body:
            self._dfs(term, plan_actions, valid_actions)
        return plan_actions

    def _dfs(self, term, actions, valid_actions: list[str]) -> str:
        # TODO: ver como lidar com goals e ações com nomes nomes iguais pois causa loop infinito
        if term not in self.subtasks or term in valid_actions:
            actions.append(term)  # is an action
            return
        elif term in self.plans:
            # plan that satisfies the goa
            subtask = self.plans[term]
            for term in subtask.body:
                self._dfs(term, actions, valid_actions)
        else:
            print(f"Term {term} is not an action or plan")
            return
