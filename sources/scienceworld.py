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
        observation, reward, completed, info = env.step(action)
        error = True if observation in error_messages else False
        updated_state = parse_observation(observation=observation,
                                          task=goal,
                                          info=info,
                                          completed=completed,
                                          error=error)
        return updated_state

    return step_function


def parse_observation(observation: str,
                      task: str,
                      info: dict,
                      completed: bool = False,
                      error: bool = False) -> State:
    """
    ScienceWorld environment specific function to convert environment information into a State object.

    :param info: dict containing scienceworld metadata resulted from performed action
    :param observation: agent observation
    :param inventory:  itens in inventory
    :param look_around: agent observations in the current environment state
    :param task: agent's goal
    :param score: score received by the environment
    :param completed: informs whether the agent could finish the main goal
    :param error: indicates if the agent executes an action that resulted in an error
    :return:
    """
    look_around = info['look']
    x = re.search(r"([\S\s]*?)(?:In it, you see:)([\S\s]*?)(?:You also see:)([\S\s]*)", look_around)
    if x is None:
        x = re.search(r"([\S\s]*?)(?:Here you see:)([\S\s]*?)(?:You also see:)([\S\s]*)", look_around)
    groups = x.groups()

    location = groups[0]
    objects = groups[1]
    doors = groups[2]

    loc_split = [location.strip()]
    obs_split = [obs.strip() for obs in objects.split('\n') if len(obs.strip()) > 0]
    obs_split = [f"You see {obs}" for obs in obs_split]
    doors_split = [door.strip() for door in doors.split('\n') if len(door.strip()) > 0]
    inventory = info['inv'].replace('\n', ' ').replace('\t', '')
    env_state_sentences = loc_split + obs_split + doors_split + [inventory, observation]

    return State(goal=task, beliefs=env_state_sentences, score=info['score'], completed=completed, error=error)
