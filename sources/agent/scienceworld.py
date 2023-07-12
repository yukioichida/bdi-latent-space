import re

def parse_observation(observation: str, inventory: str) -> list[str]:
    x = re.search(r"([\S\s]*?)(?:In it, you see:)([\S\s]*?)(?:You also see:)([\S\s]*)", observation)
    groups = x.groups()

    location = groups[0]
    objects = groups[1]
    doors = groups[2]

    loc_split = [location.strip()]
    obs_split = [obs.strip() for obs in objects.split('\n') if len(obs.strip()) > 0]
    doors_split = [door.strip() for door in doors.split('\n') if len(door.strip()) > 0]
    inventory_items = inventory.replace('\n', ' ').replace('\t', '')
    return loc_split + obs_split + doors_split + [inventory_items]