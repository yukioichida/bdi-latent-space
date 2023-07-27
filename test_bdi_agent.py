import argparse
import random
import time

from scienceworld import ScienceWorldEnv

from sources.agent.agent import BDIAgent
from sources.agent.scienceworld import parse_observation
from sources.bdi_components.belief import State
from sources.bdi_components.inference import NLIModel
from sources.bdi_components.policy import DRRNDefaultPolicy


def load_rl_policy(path: str = "models/drrn-task0/") -> DRRNDefaultPolicy:
    drrn_policy = DRRNDefaultPolicy(spm_path="models/spm_models/unigram_8k.model",
                                    trained_model_path=path,
                                    trained_model_id="-steps80000-eps562")
    print("DRRN pretrained policy initialized")
    return drrn_policy


def load_bdi_agent():
    nli_model = NLIModel(hg_model_name='alisawuffles/roberta-large-wanli')
    bdi_agent = BDIAgent(nli_model=nli_model, plan_file="plans/boil_water-easy.plan")
    print("BDI agent initialized")
    return bdi_agent


def run_bdi_agent(args):
    """ Example bdi_old agent """
    taskIdx = args['task_num']
    simplificationStr = args['simplification_str']
    numEpisodes = args['num_episodes']

    # Keep track of the agent's final scores
    finalScores = []
    # Initialize environment
    env = ScienceWorldEnv("", args['jar_path'], envStepLimit=args['env_step_limit'])

    taskNames = env.getTaskNames()
    print("Task Names: " + str(taskNames))

    # Choose task
    taskName = taskNames[taskIdx]  # Just get first task
    env.load(taskName, 0,
             "")  # Load the task, so we have access to some extra accessors e.g. getRandomVariationTrain() )
    maxVariations = env.getMaxVariations(taskName)
    print(f"Simplification: {simplificationStr}")
    time.sleep(2)

    # loading agent
    bdi_agent = load_bdi_agent()
    drrn_policy = load_rl_policy()

    # Start running episodes
    for episodeIdx in range(0, numEpisodes):
        # Pick a random task variation
        #randVariationIdx = env.getRandomVariationTest()
        randVariationIdx = 0
        # randVariationIdx = 0
        env.load(taskName, randVariationIdx, simplificationStr)
        # Reset the environment
        observation, info = env.reset()

        print(f"Task Name: " + taskName + " variation " + str(randVariationIdx))
        print("Task Variation: " + str(randVariationIdx) + " / " + str(maxVariations))
        print("Task Description: " + str(env.getTaskDescription()))

        score = 0.0
        curIter = 0

        num_default_actions = 0
        num_selected_plans = 0


        # Run one episode until we reach a stopping condition (including exceeding the maximum steps)
        action_str = "look around"  # First action
        observation, reward, isCompleted, info = env.step(action_str)
        while not isCompleted and curIter < 20:
            print("----------------------------------------------------------------")
            print("Step: " + str(curIter))
            print("\n>>> " + observation)
            print("Reward: " + str(reward))
            print("Score: " + str(score))
            print("isCompleted: " + str(isCompleted))

            # The environment will make isCompleted `True` when a stop condition has happened, or the maximum number of steps is reached.
            if (isCompleted):
                break

            obs_sentences = parse_observation(observation=info['look'], inventory=info['inv'])
            current_state = State(goal=info['taskDesc'], observation=" ".join(observation.split()).lower(),
                                  inventory=info['inv'], look=obs_sentences)

            plan_actions = bdi_agent.act(current_state=current_state, available_actions=info['valid'])
            if len(plan_actions) > 0:
                num_selected_plans += 1
                for action in plan_actions:
                    print("Executing action: " + str(action))
                    observation, reward, isCompleted, info = env.step(action)
                    print(f"{observation} - {isCompleted}")
                    score = info['score']
                    # Keep track of the number of commands sent to the environment in this episode
                    curIter += 1
            else:
                num_default_actions += 1
                # no plan has been found
                rl_action = drrn_policy.act(observation, goal=info['taskDesc'], look=info['look'],
                                            inventory=info['inv'], available_actions=info['valid'])
                for action in rl_action:
                    print("Executing action: " + str(action))
                    observation, reward, isCompleted, info = env.step(action)
                    curIter += 1
        print("Goal Progress:")
        print(env.getGoalProgressStr())
        time.sleep(1)

        # Episode finished -- Record the final score
        finalScores.append(score)

        # Report progress of model
        print("Final score: " + str(score))
        print("isCompleted: " + str(isCompleted))

        # Save history -- and when we reach maxPerFile, export them to file
        filenameOutPrefix = args['output_path_prefix'] + str(taskIdx)
        env.storeRunHistory(episodeIdx, notes={'text': 'my notes here'})
        env.saveRunHistoriesBufferIfFull(filenameOutPrefix, maxPerFile=args['max_episode_per_file'])

    # Episodes are finished -- manually save any last histories still in the buffer
    env.saveRunHistoriesBufferIfFull(filenameOutPrefix, maxPerFile=args['max_episode_per_file'], forceSave=True)

    # Show final episode scores to user:
    avg = sum([x for x in finalScores if x >= 0]) / len(
        finalScores)  # Clip negative scores to 0 for average calculation
    print("")
    print("---------------------------------------------------------------------")
    print(" Summary (Random Agent)")
    print(" Task " + str(taskIdx) + ": " + taskName)
    print(" Simplifications: " + str(simplificationStr))
    print("---------------------------------------------------------------------")
    print(" Episode scores: " + str(finalScores))
    print(" Average episode score: " + str(avg))
    print(" Num plans selected: " + str(num_selected_plans))
    print(" Num default actions: " + str(num_default_actions))
    print("---------------------------------------------------------------------")
    print("")

    print("Completed.")


def build_simplification_str(args):
    """ Build simplification_str from args. """
    simplifications = list()
    if args["teleport"]:
        simplifications.append("teleportAction")

    if args["self_watering_plants"]:
        simplifications.append("selfWateringFlowerPots")

    if args["open_containers"]:
        simplifications.append("openContainers")

    if args["open_doors"]:
        simplifications.append("openDoors")

    if args["no_electrical"]:
        simplifications.append("noElectricalAction")

    return "easy"  # args["simplifications_preset"] or ",".join(simplifications)


#
#   Parse command line arguments
#
def parse_args():
    desc = "Run bdi_old agent."
    parser = argparse.ArgumentParser(desc)
    parser.add_argument("--jar_path", type=str,
                        help="Path to the ScienceWorld jar file. Default: use builtin.")
    parser.add_argument("--task-num", type=int, default=0,
                        help="Specify the task number to play. Default: %(default)s")
    parser.add_argument("--var-num", type=int, default=0,
                        help="Specify the task variation number to play. Default: %(default)s")
    parser.add_argument("--env-step-limit", type=int, default=100,
                        help="Maximum number of steps per episode. Default: %(default)s")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Number of episodes to play. Default: %(default)s")
    parser.add_argument("--seed", type=int,
                        help="Seed the random generator used for sampling random actions.")
    parser.add_argument("--output-path-prefix", default="save-histories",
                        help="Path prefix to use for saving episode transcripts. Default: %(default)s")
    parser.add_argument("--max-episode-per-file", type=int, default=1000,
                        help="Maximum number of episodes per transcript file. Default: %(default)s")

    simplification_group = parser.add_argument_group('Game simplifications')
    simplification_group.add_argument("--simplifications-preset", choices=['easy'],
                                      help="Choose a preset among: 'easy' (apply all possible simplifications).")
    simplification_group.add_argument("--teleport", action="store_true",
                                      help="Lets agents instantly move to any location.", default=True)
    simplification_group.add_argument("--self-watering-plants", action="store_true",
                                      help="Plants do not have to be frequently watered.")
    simplification_group.add_argument("--open-containers", action="store_true",
                                      help="All containers are opened by default.")
    simplification_group.add_argument("--open-doors", action="store_true",
                                      help="All doors are opened by default.")
    simplification_group.add_argument("--no-electrical", action="store_true",
                                      help="Remove the electrical actions (reduces the size of the action space).")

    args = parser.parse_args()
    params = vars(args)
    return params


def main():
    print("ScienceWorld 1.0 API Examples - Random Agent")
    # Parse command line arguments
    args = parse_args()
    random.seed(args["seed"])
    args["simplification_str"] = build_simplification_str(args)
    run_bdi_agent(args)


if __name__ == "__main__":
    main()
