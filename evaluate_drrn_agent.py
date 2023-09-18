import torch
import argparse
import random
import torch
from scienceworld import ScienceWorldEnv

from sources.drrn.drrn_agent import DRRN_Agent


def sanitizeInfo(infoIn):
    # Convert from py4j.java_collections.JavaList to python list
    recastList = []
    for elem in infoIn['valid']:
        recastList.append(elem)

    # print("SanitizeInfo:" + str(infoIn))

    info = {'moves': infoIn['moves'],
            'reward': infoIn['reward'],
            'score': infoIn['score'],
            'look': infoIn['look'],
            'inv': infoIn['inv'],
            'valid': recastList,
            'taskDesc': infoIn['taskDesc']
            }

    return info


def sanitizeObservation(obsIn, infoIn):
    obs = infoIn['taskDesc'] + " OBSERVATION " + obsIn
    return obs


def clean(strIn):
    charsToFilter = ['\t', '\n', '*', '-']
    for c in charsToFilter:
        strIn = strIn.replace(c, ' ')
    return strIn.strip()


def resetWithVariationDev(env, taskName, simplificationStr):
    variationIdx = env.getRandomVariationDev()  ## Random variation on dev
    env.load(taskName, variationIdx, simplificationStr)
    print("Simplifications: " + env.getSimplificationsUsed())
    initialObs, initialDict = env.reset()
    return initialObs, initialDict


def resetWithVariationTest(env, taskName, simplificationStr):
    variationIdx = env.getRandomVariationTest()  ## Random variation on test
    env.load(taskName, variationIdx, simplificationStr)
    print("Simplifications: " + env.getSimplificationsUsed())
    initialObs, initialDict = env.reset()
    return initialObs, initialDict


# Initialize a ScienceWorld environment directly from the API
def initializeEnv(threadNum, args):

    env = ScienceWorldEnv("", None, args.env_step_limit)

    taskNames = env.getTaskNames()
    taskName = taskNames[args.task_idx]

    # Just reset to variation 0, as another call (e.g. resetWithVariation...) will setup an appropriate variation (train/dev/test)
    env.load(taskName, 0, args.simplification_str)
    return env, taskName


def evaluate(agent, args, env_step_limit, nb_episodes=10):
    # Initialize a ScienceWorld thread for serial evaluation
    env, taskName = initializeEnv(threadNum=args.num_envs + 10,
                        args=args)  # A threadNum (and therefore port) that shouldn't be used by any of the regular training workers

    scoresOut = []
    with torch.no_grad():
        for ep in range(nb_episodes):
            total_score = 0
            print("Starting evaluation episode {}".format(ep))
            print("Starting evaluation episode " + str(ep) + " / " + str(nb_episodes))
            score = evaluate_episode(agent, taskName, env, env_step_limit, args.simplification_str, args.eval_set)
            print("Evaluation episode {} ended with score {}\n\n".format(ep, score))
            total_score += score
            scoresOut.append(total_score)
            print("")

        avg_score = total_score / nb_episodes

        #env.shutdown()

        return scoresOut, avg_score


def evaluate_episode(agent, taskName, env, env_step_limit, simplificationStr, evalSet):
    step = 0
    done = False
    numSteps = 0
    ob = ""
    info = {}
    if (evalSet == "dev"):
        ob, info = resetWithVariationDev(env, taskName, simplificationStr)
        info = sanitizeInfo(info)
        ob = sanitizeObservation(ob, info)

    elif (evalSet == "test"):
        ob, info = resetWithVariationTest(env, taskName, simplificationStr)
        info = sanitizeInfo(info)
        ob = sanitizeObservation(ob, info)

    else:
        print("evaluate_episode: unknown evaluation set (expected 'dev' or 'test', found: " + str(evalSet) + ")")
        env.shutdown()

        exit(1)

    state = agent.build_state(obs=ob, inv=info['inv'], look=info['look'])
    print('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']), clean(info['look'])))
    while not done:
        # print("numSteps: " + str(numSteps))
        valid_acts = info['valid']
        valid_ids = agent.encode(valid_acts)
        _, action_idx, action_values = agent.act([state], [valid_ids], sample=False)
        action_idx = action_idx[0]
        action_values = action_values[0]
        action_str = valid_acts[action_idx]
        #print('Action{}: {}, Q-Value {:.2f}'.format(step, action_str, action_values[action_idx].item()))
        s = ''

        maxToDisplay = 10  # Max Q values to display, to limit the log size
        numDisplayed = 0
        for idx, (act, val) in enumerate(sorted(zip(valid_acts, action_values), key=lambda x: x[1], reverse=True), 1):
            s += "{}){:.2f} {} ".format(idx, val.item(), act)
            numDisplayed += 1
            if (numDisplayed > maxToDisplay):
                break

        #print('Q-Values: {}'.format(s))
        ob, rew, done, info = env.step(action_str)
        info = sanitizeInfo(info)
        ob = sanitizeObservation(ob, info)

        #print("Reward{}: {}, Score {}, Done {}".format(step, rew, info['score'], done))
        step += 1
        #print('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']), clean(info['look'])))
        state = agent.build_state(obs=ob, inv=info['inv'], look=info['look'])

        numSteps += 1
        if (numSteps > env_step_limit):
            print("Maximum number of evaluation steps reached (" + str(env_step_limit) + ").")
            break

    print("Completed one evaluation episode")

    return info['score']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--spm_path', default='../spm_models/unigram_8k.model')
    parser.add_argument('--rom_path', default='zork1.z5')
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_envs', default=16, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--checkpoint_freq', default=500, type=int)
    parser.add_argument('--eval_freq', default=500, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--memory_size', default=5000000, type=int)
    parser.add_argument('--priority_fraction', default=0.0, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)

    parser.add_argument('--task_idx', default=0, type=int)
    parser.add_argument('--maxHistoriesPerFile', default=1000, type=int)
    parser.add_argument('--historySavePrefix', default='saveout', type=str)

    parser.add_argument('--eval_set', default='dev', type=str)  # 'dev' or 'test'

    parser.add_argument('--simplification_str', default='', type=str)

    return parser.parse_args()


def load_drrn_agent(path: str = "models/drrn-task0/"):
    agent = DRRN_Agent(spm_path="models/spm_models/unigram_8k.model")
    #agent.load(path, "-steps80000-eps562")
    print("DRRN agent initialized")
    return agent


def main():
    ## assert jericho.__version__ == '2.1.0', "This code is designed to be run with Jericho version 2.1.0."
    args = parse_args()
    print(args)
    agent = load_drrn_agent()
    eval_scores, avg_eval_score = evaluate(agent, args, args.env_step_limit)

    for eval_score in eval_scores:
        print("EVAL EPISODE SCORE: " + str(eval_score))


if __name__ == "__main__":
    main()
