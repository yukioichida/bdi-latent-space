from scienceworld import ScienceWorldEnv
from sources.agent import BDIAgent
from sources.scienceworld import parse_observation

from sources.bdi_components.belief import State
from sources.bdi_components.inference import NLIModel
from sources.bdi_components.plans import PlanLibrary


def load_plan_library():
    # MAIN GOAL
    green_plan = """
        IF your goal is to use chemistry to create green paint THEN
            move to art studio,
            pour cup containing blue paint in art studio into jug,
            pour cup containing yellow paint in art studio into jug,
            mix jug,
            focus on green paint
    """

    orange_plan = """
        IF your goal is to use chemistry to create orange paint THEN
            move to art studio,
            pour cup containing red paint in art studio into jug,
            pour cup containing yellow paint in art studio into jug,
            mix jug,
            focus on orange paint
    """

    violet_plan = """
        IF your goal is to use chemistry to create violet paint THEN
            move to art studio,
            pour cup containing red paint in art studio into jug,
            pour cup containing blue paint in art studio into jug,
            mix jug,
            focus on violet paint
    """

    all_plans = [green_plan, orange_plan, violet_plan]

    pl = PlanLibrary()
    pl.load_plans_from_strings(all_plans)  # load plans from strings above
    pl.load_plans_from_file("notebooks/plans.txt")  # load plans from file
    print(pl.plans.keys())
    return pl


if __name__ == '__main__':
    pl = load_plan_library()
    nli_model = NLIModel("alisawuffles/roberta-large-wanli")
    env = ScienceWorldEnv("", "", envStepLimit=100)

    # root_event = 'use chemistry to create green paint'

    env.load('chemistry-mix-paint-secondary-color', 0)

    num_episode = 5
    all_scores = []
    for episode in range(num_episode):

        randVariationIdx = env.getRandomVariationTest()
        env.load('chemistry-mix-paint-secondary-color', randVariationIdx)

        goal = env.getTaskDescription().split('.')[0].replace("Your task is to", "").strip()

        #print(f"Task Name: " + 'boil' + " variation " + str(randVariationIdx))
        #print("Task Description: " + str(env.getTaskDescription()))

        # Reset the environment
        observation, info = env.reset()
        # initial state
        observation, reward, isCompleted, info = env.step('look around')
        current_state = parse_observation(observation=observation, inventory=info['inv'], look_around=info['look'],
                                          task=goal, valid_actions=info['valid'])


        def step_function(action: str) -> State:
            observation, reward, isCompleted, info = env.step(action)
            updated_state = parse_observation(observation=observation,
                                              inventory=info['inv'],
                                              look_around=info['look'],
                                              task=goal,
                                              valid_actions=info['valid'],
                                              score=info['score'])
            return updated_state


        agent = BDIAgent(plan_library=pl, nli_model=nli_model)
        last_state = agent.act(current_state, step_function=step_function)

        #print(env.getGoalProgressStr())
        all_scores.append(last_state.reward)

    print(f"score = {sum(all_scores)/len(all_scores)}")
    print(f"all_scores = {all_scores}")
