from scienceworld import ScienceWorldEnv
from sources.agent import BDIAgent
from sources.bdi_components.policy import DRRNDefaultPolicy
from sources.scienceworld import parse_observation, load_step_function

from sources.bdi_components.belief import State
from sources.bdi_components.inference import NLIModel
from sources.bdi_components.plans import PlanLibrary
from sources.drrn.drrn_agent import DRRN_Agent

import pandas as pd
import random
import numpy
import torch
import re

from os import listdir
from os.path import isfile, join


def load_plan_library(plan_file: str):
    pl = PlanLibrary()
    pl.load_plans_from_file(plan_file)
    pl.load_plans_from_file("plans/plans_nl/plan_common.plan")
    pl.load_plans_from_file("notebooks/plans_navigation.txt")
    print(pl.plans.keys())
    return pl


def get_drrn_trained_models(path: str):
    # = '../models/model_task1melt/'
    model_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".pt")]
    print(model_files)
    metadata = []

    for file in model_files:
        x = re.findall("model-steps(\d*)-eps(\d*).pt", file)[0]
        steps, eps = x
        metadata.append({
            'model_file': path + file,
            'eps': eps,
            'steps': steps
        })

    models_df = pd.DataFrame(metadata).sort_values("eps")
    return models_df


def get_plan_files():
    task = "plan_1_melt"
    task = "plan_3_focus_non_living_thing"
    plan_files = [f"plans/plans_nl/{task}_100.plan",
                  f"plans/plans_nl/{task}_75.plan",
                  f"plans/plans_nl/{task}_50.plan",
                  f"plans/plans_nl/{task}_25.plan"]
    rows = []
    for file in plan_files:
        x = re.findall(f"plans/plans_nl/{task}_(\d*).plan", file)[0]
        pct = x[0]
        rows.append({
            'plan_file': file,
            'pct_plans': pct
        })

    plans_df = pd.DataFrame(rows).sort_values("pct_plans")
    return plans_df


def random_seed(seed):
    torch.random.manual_seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    random_seed(42)
    # task = 'melt'
    task = 'find-non-living-thing'
    hg_nli_model = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"  # best
    # hg_nli_model = "MoritzLaurer/MiniLM-L6-mnli"
    # hg_nli_model = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    nli_model = NLIModel(hg_nli_model, device='cuda')
    env = ScienceWorldEnv("", "", envStepLimit=100)
    env.load(task, 0)

    plans_df = get_plan_files()
    plans_df['id'] = 0
    # models_df = get_drrn_trained_models("models/model_task1melt/")
    models_df = get_drrn_trained_models("models/models_task13-overfit/")
    models_df['id'] = 0
    experiment_df = plans_df.merge(models_df, on='id', how='outer')

    results = []

    for i, row in experiment_df.iterrows():
        print(f"Loading plan file: {row['plan_file']}")
        pl = load_plan_library(row['plan_file'])
        all_scores = []

        test_variations = env.getVariationsTest()
        for var in test_variations:
            env.load(task, var, simplificationStr="easy")

            # main_goal = env.getTaskDescription().split('.')[0].replace("Your task is to", "").strip()
            # main_goal = env.getTaskDescription() \
            #    .replace(". First, focus on the thing. Then,", "") \
            #    .replace("move", "Your task is to", "") \
            #    .replace(".", "") \
            #    .strip()
            main_goal = env.getTaskDescription().replace(". First, focus on the thing. Then,", "").replace("move",
                                                                                                           "by moving").replace(
                "Your task is to", "").replace(".", "").strip()

            # print(main_goal)
            env.reset()
            step_function = load_step_function(env, main_goal)

            # initial state
            observation, reward, isCompleted, info = env.step('look around')
            current_state = parse_observation(observation=observation,
                                              inventory=info['inv'],
                                              look_around=info['look'],
                                              task=main_goal,
                                              valid_actions=info['valid'],
                                              task_description=env.getTaskDescription())
            print(f"== {main_goal} - {var} ==")
            bdi_agent = BDIAgent(plan_library=pl, nli_model=nli_model)
            bdi_state = bdi_agent.act(current_state, step_function=step_function)
            score = bdi_state.score

            # initial state
            rl_actions = []
            if bdi_state.error:  # TODO: maybe I should incorporate this code into the BDI agent
                print(f"BDIScore = {score} - Starting RL agent ...")
                drrn_agent = DRRN_Agent(spm_path="models/spm_models/unigram_8k.model")
                drrn_agent.load(row['model_file'])
                observation, reward, isCompleted, info = env.step('look around')
                for _ in range(100):  # stepLimits
                    drrn_state = drrn_agent.build_state(obs=observation, inv=info['inv'], look=info['look'])
                    valid_ids = drrn_agent.encode(info['valid'])
                    _, action_idx, action_values = drrn_agent.act([drrn_state], [valid_ids], sample=False)
                    action_idx = action_idx[0]
                    action_values = action_values[0]
                    action_str = info['valid'][action_idx]
                    rl_actions.append(action_str)
                    observation, reward, isCompleted, info = env.step(action_str)
                    if isCompleted:
                        break
                score = info['score']

            plan_found = 1 if len(bdi_agent.event_trace) > 0 else 0
            results.append({
                'num_bdi_actions': len(bdi_agent.action_trace),
                'num_rl_actions': len(rl_actions),
                'plan_found': plan_found,
                'variation': var,
                'error': bdi_state.error,
                'score': score,
                'bdi_score': bdi_state.score,
                'complete': isCompleted,
                'num_plans': len(bdi_agent.event_trace),
                'plan_library_size': len(pl.plans.keys()),
                'plans_pct': row['pct_plans'],
                'eps': row['eps']
            })

            all_scores.append(score)
            print(f"Finish = {isCompleted} - Score {score} - Variation {var}")

        avg_score = sum(all_scores) / len(all_scores)
        print(f"score = {avg_score}")
        print(f"all_scores = {all_scores}")

    pd.DataFrame(results).to_csv("results.csv", index=False)
