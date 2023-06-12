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

    x = re.search(r"([\S\s]*?)(?<=In it, you see:)([\S\s]*?)(?:CONSIDERING)([\S\s]*)(?:You also see:)([\S\s]*)", observation)
    groups = x.groups()
    print(groups)

