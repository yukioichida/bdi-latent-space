# exampeLoadGoldPaths.py
#
# This example illustrates how to traverse the gold (oracle) paths for ScienceWorld.
#

import json

#filenameIn = "goldsequences-0.json"
filenameIn = "goldsequences-0-1-2-3-4-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29.json"
f = open(filenameIn)
data = json.load(f)

print(type(data))
print("Tasks stored in this gold data: " + str(data.keys()))
numMoves = 0
numSequences = 0

all_observation = []

for taskIdx in data.keys():
    print("Task Index: " + str(taskIdx))

    taskData = data[taskIdx]
    taskName = taskData['taskName']
    
    goldSequences = taskData['goldActionSequences']

    for goldSequence in goldSequences:
        variationIdx = goldSequence['variationIdx']
        taskDescription = goldSequence['taskDescription']
        fold = goldSequence['fold']
        path = goldSequence['path']

        #print(f"variation {variationIdx} - Task Description {taskDescription} - taskName {taskName} - fold {fold}")
        for step in path:
            action = step['action']
            obs = step['observation']
            freelook = step['freelook']

            score = step['score']
            isCompleted = step['isCompleted']
            #print("> " + str(action))
            #print(obs.replace("\t", "").split("\n"))
            #print("")
            distinct_obs = obs.replace("\t", "").split("\n")
            #all_observation.append(obs)
            #print(action)
            #print(distinct_obs)
            #print(freelook)
            #print(step['inventory'])
            [all_observation.append(o) for o in distinct_obs]
 
            numMoves +=1
        numSequences += 1


print("----------------------")
print("Summary Statistics:")
print("numTasks: " + str(len(data.keys())))
print("numSequences: " + str(numSequences))
print("numMoves: " + str(numMoves))
print("numDistinctObservations: " + str(len(set(all_observation))))

# montar dataset, por frase primeiro, reconstruir a frase,
all_obs = list(set(all_observation))

all_len = [len(s.split()) for s in all_obs]

print(max(all_len))
