import json
from collections import namedtuple

Plan = namedtuple("Plan", "context actions")

goldpath_file = "../data/goldsequences-0-1-2-3-4-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29.json"
#goldpath_file = "../data/goldsequences-0.json"

with open(goldpath_file) as file:
    data = json.load(file)


variations = len(data['0']['goldActionSequences'])

all_tasks = []

for var in range(variations):
    task = data['0']['goldActionSequences'][var]['taskDescription'].split('.')[0]
    all_tasks.append(task)

for idx, task in enumerate(all_tasks):
    print(f"variation {idx} - {task}")

import difflib

def matches(list1, list2):
    while True:
        mbs = difflib.SequenceMatcher(None, list1, list2).get_matching_blocks()
        if len(mbs) == 1: break
        for i, j, n in mbs[::-1]:
            if n > 0: yield list1[i: i + n]
            del list1[i: i + n]
            del list2[j: j + n]

variations = [26, 27] # water

var_actions = []

for var in variations:
    actions = []
    for path in data['0']['goldActionSequences'][var]['path']:
        actions.append(path['action'])

    var_actions.append(actions)
print(var_actions[0])
print(var_actions[1])
list(matches(var_actions[0], var_actions[1]))