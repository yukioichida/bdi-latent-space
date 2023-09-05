import re

from typing import Callable

from scienceworld import ScienceWorldEnv

from sources.bdi_components.belief import State

"""
Scienceworld environment functions 
"""

error_messages = ["No known action matches that input.",
                  "The door is not open.",
                  "The stove appears broken, and can't be activated or deactivated."]


def load_step_function(env: ScienceWorldEnv, goal: str) -> Callable[[str], State]:
    def step_function(action: str) -> State:
        """
        Wrapper function that executes a natural language action in ScienceWorld environment
        :param action: action to be performed by the agent in the environment
        :return: state updated given the action performed
        """
        observation, reward, isCompleted, info = env.step(action)

        error = False
        if observation in error_messages:
            print(info['look'])
            print(f"Action: {action} - obs: {observation}")
            error = True

        updated_state = parse_observation(observation=observation,
                                          inventory=info['inv'],
                                          look_around=info['look'],
                                          task=goal,
                                          valid_actions=info['valid'],
                                          score=info['score'],
                                          complete=isCompleted,
                                          error=error)
        return updated_state

    return step_function


def parse_observation(observation: str,
                      inventory: str,
                      look_around: str,
                      task: str,
                      valid_actions: list[str],
                      score: float = 0,
                      complete: bool = False,
                      error: bool = False) -> State:
    """
    ScienceWorld environment specific function to convert environment information into a State object.

    :param observation: agent observation
    :param inventory:  itens in inventory
    :param look_around: agent observations in the current environment state
    :param task: agent's goal
    :param valid_actions: actions allowed to perform given the current environment state
    :param score: score received by the environment
    :return:
    """
    x = re.search(r"([\S\s]*?)(?:In it, you see:)([\S\s]*?)(?:You also see:)([\S\s]*)", look_around)
    if x == None:
        x = re.search(r"([\S\s]*?)(?:Here you see:)([\S\s]*?)(?:You also see:)([\S\s]*)", look_around)
    groups = x.groups()

    location = groups[0]
    objects = groups[1]
    doors = groups[2]

    loc_split = [location.strip()]
    obs_split = [obs.strip() for obs in objects.split('\n') if len(obs.strip()) > 0]
    obs_split = [f"You see {obs}" for obs in obs_split]
    doors_split = [door.strip() for door in doors.split('\n') if len(door.strip()) > 0]
    env_state_sentences = loc_split + obs_split + doors_split
    inventory = inventory.replace('\n', ' ').replace('\t', '')

    return State(goal=task,
                 observation=observation,
                 look=env_state_sentences,
                 inventory=inventory,
                 valid_actions=valid_actions,
                 score=score,
                 complete=complete,
                 error=error)
