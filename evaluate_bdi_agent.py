from scienceworld import ScienceWorldEnv
from sources.agent import BDIAgent
from sources.scienceworld import parse_observation

from sources.bdi_components.belief import State
from sources.bdi_components.inference import NLIModel
from sources.bdi_components.plans import PlanLibrary


def load_plan_library():
    pl = PlanLibrary()
    pl.load_plans_from_file("plans/plans_nl/plan_1_boil.plan")  # load plans from strings above
    pl.load_plans_from_file("notebooks/plans_navigation.txt")  # load plans from file
    print(pl.plans.keys())
    return pl

def load_step_function(env: ScienceWorldEnv, goal: str):
    def step_function(action: str) -> State:
        observation, reward, isCompleted, info = env.step(action)
        updated_state = parse_observation(observation=observation,
                                          inventory=info['inv'],
                                          look_around=info['look'],
                                          task=goal,
                                          valid_actions=info['valid'],
                                          score=info['score'])
        return updated_state

    return step_function

if __name__ == '__main__':
    task = 'boil'
    pl = load_plan_library()
    nli_model = NLIModel("alisawuffles/roberta-large-wanli")
    env = ScienceWorldEnv("", "", envStepLimit=100)
    env.load(task, 0)
    num_episode = 5
    all_scores = []

    for episode in range(num_episode):
        randVariationIdx = env.getRandomVariationTest()
        env.load(task, randVariationIdx)

        main_goal = env.getTaskDescription().split('.')[0].replace("Your task is to", "").strip()
        observation, info = env.reset()
        step_function = load_step_function(env, main_goal)
        # initial state
        observation, reward, isCompleted, info = env.step('look around')
        current_state = parse_observation(observation=observation, inventory=info['inv'], look_around=info['look'],
                                          task=main_goal, valid_actions=info['valid'])

        agent = BDIAgent(plan_library=pl, nli_model=nli_model)
        last_state = agent.act(current_state, step_function=step_function)
        all_scores.append(last_state.reward)

    print(f"score = {sum(all_scores) / len(all_scores)}")
    print(f"all_scores = {all_scores}")
