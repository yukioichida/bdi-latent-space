from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification, \
    RobertaTokenizer
import torch
import re

if __name__ == '__main__':
    max_length = 512

#if plan.task entails belief.task:
	#if belief.look+inv entails plan.plan_context:
    goal = "Your task is to boil gallium"
    observation = """
     This room is called the art studio. In it, you see: 
        the agent
        a substance called air
        a large cupboard. The large cupboard door is closed. 
        a table. On the table is: a glass cup (containing nothing).
        a wood cup (containing blue paint)
        a wood cup (containing yellow paint)
        a wood cup (containing red paint)
    You also see:
        A door to the hallway (that is closed)
    """

    x = re.search(r"([\S\s]*?)(?:In it, you see:)([\S\s]*?)(?:You also see:)([\S\s]*)", observation)
    groups = x.groups()


    for i, g in enumerate(groups):
        print(f"part {i}")
        print(g)

    location = groups[0]
    objects = groups[1]
    doors = groups[2]

    # spatial map
    # varrer as ações válidas começando com go to
    # escolher uma aleatória

    for obj in objects.split("\n"):
        obj = obj.strip()
        loc = location.strip()
        if obj:
            print(f"{loc} In it, you see {obj}")

    beliefs = location + objects + doors

    loc = [location.strip()]
    obs = [f"you see {obs.strip()}" for obs in objects.split('\n')if len(obs.strip()) > 0]
    d = [door.strip() for door in doors.split('\n') if len(door.strip()) > 0]

    print(loc)
    print(obs)
    print(d)

