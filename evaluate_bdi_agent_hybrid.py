import random
import re
import argparse
from os import listdir
from os.path import isfile, join

import pandas as pd
import torch
from scienceworld import ScienceWorldEnv

from sources.agent import BDIAgent
from sources.bdi_components.belief import State
from sources.bdi_components.inference import NLIModel
from sources.bdi_components.plans import PlanLibrary
from sources.drrn.drrn_agent import DRRN_Agent
from sources.scienceworld import parse_observation, load_step_function


def load_plan_library(plan_file: str):
    pl = PlanLibrary()
    pl.load_plans_from_file(plan_file)
    pl.load_plans_from_file("plans/plans_nl/plan_common.plan")
    pl.load_plans_from_file("notebooks/plans_navigation.txt")
    print(pl.plans.keys())
    return pl


def get_drrn_pertrained_info(path: str) -> pd.DataFrame:
    # = '../models/model_task1melt/'
    model_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".pt")]
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


def get_plan_files(task: str) -> pd.DataFrame:
    # task = "plan_1_melt"
    # task = "plan_3_focus_non_living_thing"
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


def bdi_phase(plan_library: PlanLibrary, nli_model: NLIModel, env: ScienceWorldEnv) -> (State, BDIAgent):
    """
    Executes the experiment phase where the BDI agent reasons over the environment state and call plans.
    :param plan_library: Plan Library containing plans
    :param nli_model: Natural Language Inference model
    :param env:
    :return: Last state achieved by the BDI agent with its own instance.
    """
    main_goal = env.getTaskDescription() \
        .replace(". First, focus on the thing. Then,", "") \
        .replace("move", "by moving") \
        .replace("Your task is to", "") \
        .replace(".", "").strip()

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
    bdi_agent = BDIAgent(plan_library=plan_library, nli_model=nli_model)
    bdi_state = bdi_agent.act(current_state, step_function=step_function)
    return bdi_state, bdi_agent


def drrn_phase(env: ScienceWorldEnv) -> (State, list[str]):
    """
    Executed the experiment phase where the agent using a RL trained policy tries to reach the goal
    given the current environment state.
    :param env: Current state of environment
    :return: last state achieved by the Policy-driven agent with its list of action perfomed.
    """
    drrn_agent = DRRN_Agent(spm_path="models/spm_models/unigram_8k.model")
    drrn_agent.load(row['model_file'])
    observation, reward, isCompleted, info = env.step('look around')
    rl_actions = []
    for _ in range(100):  # stepLimits
        drrn_state = drrn_agent.build_state(obs=observation, inv=info['inv'], look=info['look'])
        valid_ids = drrn_agent.encode(info['valid'])
        _, action_idx, action_values = drrn_agent.act([drrn_state], [valid_ids], sample=False)
        action_idx = action_idx[0]
        action_str = info['valid'][action_idx]
        rl_actions.append(action_str)
        observation, reward, isCompleted, info = env.step(action_str)
        if isCompleted:
            break
    return State(completed=isCompleted, score=info['score']), rl_actions


def load_experiment_info(args: argparse.Namespace) -> pd.DataFrame:
    """
    Loads all test scenarios to be executed in experiments.
    :return: Dataframe containing information of each test scenario.
    """
    task = ""
    plans_df = get_plan_files(task)
    plans_df['id'] = 0
    # models_df = get_drrn_trained_models("models/model_task1melt/")
    models_df = get_drrn_pertrained_info(args.drrn_pretrained_file)
    models_df['id'] = 0
    experiment_df = plans_df.merge(models_df, on='id', how='outer')
    return experiment_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='melt')
    parser.add_argument('--drrn_pretrained_file', type=str, default='models/models_task13-overfit/')
    parser.add_argument('--nli_model', type=str, default='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli')
    # parser.add_argument('--nli_model', type=str, default='MoritzLaurer/MiniLM-L6-mnli')
    # parser.add_argument('--nli_model', type=str, default='ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli')

    return parser.parse_args()


if __name__ == '__main__':
    random_seed(42)
    args = parse_args()
    nli_model = NLIModel(args.nli_model, device='cuda')
    env = ScienceWorldEnv("", "", envStepLimit=100)
    env.load(args.task, 0)

    experiment_df = load_experiment_info(args)
    results = []

    for i, row in experiment_df.iterrows():
        print(f"Loading plan file: {row['plan_file']}")
        pl = load_plan_library(row['plan_file'])
        all_scores = []

        for var in env.getVariationsTest():
            env.load(args.task, var, simplificationStr="easy")
            # BDI Phase
            bdi_state, bdi_agent = bdi_phase(plan_library=pl, nli_model=nli_model, env=env)
            rl_actions = []
            last_state = bdi_state
            rl_score = 0

            if bdi_state.error:  # TODO: maybe I should incorporate this code into the BDI agent
                # RL trained Policy Phase
                rl_state, rl_actions = drrn_phase(env)
                last_state = rl_state
                rl_score = rl_state.score - bdi_state.score  # score acquired exclusively from DRRN (RL)

            plan_found = 1 if len(bdi_agent.event_trace) > 0 else 0
            results.append({
                'num_bdi_actions': len(bdi_agent.action_trace),
                'num_rl_actions': len(rl_actions),
                'plan_found': plan_found,
                'variation': var,
                'error': bdi_state.error,
                'bdi_score': bdi_state.score,
                'rl_score': rl_score,
                'final_score': last_state.score,
                'complete': last_state.completed,
                'num_plans': len(bdi_agent.event_trace),
                'plan_library_size': len(pl.plans.keys()),
                'plans_pct': row['pct_plans'],
                'eps': row['eps']
            })

    pd.DataFrame(results).to_csv(f"results_{args.task}.csv", index=False)
