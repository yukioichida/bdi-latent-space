import re
from typing import NamedTuple, Dict

from sources.bdi_components.inference import NLIModel


class Plan(NamedTuple):
    task: str
    context: list[str]
    body: list[str]
    idx: int
    
    def plan_header(self):
        return 'if your task is to ' + self.task + ' and ' + self.context


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
            
            all_context = context.split("and")
            
            body = [self.preprocess_text(text) for text in groups[2].split(',')]
            return Plan(task, all_context, body, idx)
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
    
    def __init__(self, plans_file: str):
        self.plans: Dict[str, Plan] = dict()  # dict key with trigger event -> plan
        self.subtasks = []
        plans_from_file = load_plans_from_file(plans_file)
        self.load_plans(plans_from_file)
    
    def load_plans(self, input_plans: list[Plan]):
        for plan in input_plans:
            self.plans[plan.task] = plan
            self.subtasks.append(plan.task)
    
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
        """
        Decompose a
        :param term:
        :param actions:
        :param valid_actions:
        :return:
        """
        # TODO: ver como lidar com goals e ações com nomes nomes iguais pois causa loop infinito
        if term not in self.subtasks or term in valid_actions:  # primitive task
            actions.append(term)  # is an action (primitive task)
            return
        elif term in self.plans:  # network task
            # decompose operation
            subtask = self.plans[term]
            for term in subtask.body: # TODO: check whether the plan precondition is entailed by the current belief base
                self._dfs(term, actions, valid_actions)
        else:
            print(f"Term {term} is not an action or plan")
            return
