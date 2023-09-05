from scienceworld import ScienceWorldEnv
from sources.agent import BDIAgent
from sources.scienceworld import parse_observation, load_step_function

from sources.bdi_components.belief import State
from sources.bdi_components.inference import NLIModel
from sources.bdi_components.plans import PlanLibrary

import pandas as pd
import random
import numpy
import torch


def load_plan_library(plan_file: str):
    pl = PlanLibrary()
    pl.load_plans_from_file(plan_file)
    pl.load_plans_from_file("plans/plans_nl/plan_common.plan")
    pl.load_plans_from_file("notebooks/plans_navigation.txt")
    print(pl.plans.keys())
    return pl


def random_seed(seed):
    torch.random.manual_seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    random_seed(42)
    task = 'melt'
    hg_nli_model = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli" # best
    #hg_nli_model = "MoritzLaurer/MiniLM-L6-mnli"
    # hg_nli_model = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    nli_model = NLIModel(hg_nli_model, device='cuda')
    env = ScienceWorldEnv("", "", envStepLimit=100)
    env.load(task, 0)
    num_episode = 10

    plan_files = ["plans/plans_nl/plan_1_melt.plan",
                  "plans/plans_nl/plan_1_melt_75.plan",
                  "plans/plans_nl/plan_1_melt_50.plan",
                  "plans/plans_nl/plan_1_melt_25.plan"]

    statistics = []
    for file in plan_files:
        print(f"Loading plan file: {file}")
        pl = load_plan_library(file)
        pl_size = len(pl.plans.keys())
        all_scores = []

        test_variations = env.getVariationsTest()
        for var in test_variations:
            env.load(task, var)

            main_goal = env.getTaskDescription().split('.')[0].replace("Your task is to", "").strip()
            env.reset()
            step_function = load_step_function(env, main_goal)

            # initial state
            observation, reward, isCompleted, info = env.step('look around')
            current_state = parse_observation(observation=observation, inventory=info['inv'], look_around=info['look'],
                                              task=main_goal, valid_actions=info['valid'])
            print(f"================== {main_goal} - {var} ==================")

            agent = BDIAgent(plan_library=pl, nli_model=nli_model)
            last_state = agent.act(current_state, step_function=step_function)
            plan_found = 1 if len(agent.trace) > 0 else 0
            statistics.append({
                'plan_found': plan_found,
                'variation': var,
                'error': last_state.error,
                'score': last_state.score,
                'complete': last_state.complete,
                'num_plans': len(agent.trace),
                'plan_library_size': pl_size
            })

            all_scores.append(last_state.score)
            print(f"Finish = {isCompleted} - Score {last_state.score} - Variation {var}")

        avg_score = sum(all_scores) / len(all_scores)
        print(f"score = {avg_score}")
        print(f"all_scores = {all_scores}")

    pd.DataFrame(statistics).to_csv("results.csv", index=False)
