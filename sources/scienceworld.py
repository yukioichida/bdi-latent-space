import re

from sources.bdi_components.belief import State


def parse_observation(observation: str, inventory: str, look_around: str, task: str, valid_actions: list[str]) -> State:
    """
    ScienceWorld environment specific function to convert environment information into a State object.

    :param observation: agent observation
    :param inventory:  itens in inventory
    :param look_around: agent observations in the current environment state
    :param task: agent's goal
    :param valid_actions: actions allowed to perform given the current environment state
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
                 valid_actions=valid_actions)
