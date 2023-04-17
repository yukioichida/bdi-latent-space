"""

- Como eu escolhi o plano para dar certo?
	- pegar trajeto de um agente RL (bootstrap)
	- ver onde deu errado
	- colocar um plano na nossa linguagem
	- e mostrar como podemos melhorar




"""


if __name__ == '__main__':

    plan = """
        IF your task is to boil an water 
        THEN
        go to the kitchen, pick the metal pot, fill with water and boil it
    """

    # plan go to the kitchen
    subplan_a = """
        IF you want to go to the kitchen CONSIDERING you see a door to the kitchen THEN
        open door to the kitchen
        go to the kitchen
    """

    subplan_b = """
        IF you want to pick the metal pot CONSIDERING you are in the kitchen THEN
        open cupboard
        pick up metal pot
    """

    subplan_c = """
        IF you want to fill metal pot with water CONSIDERING you already have a metal pot in your inventory and you see a sink
        THEN
        pour metal pot into sink
        move metal pot to sink
        activate sink
        deactivate sink
        pick up metal pot
    """

    # regex = (?<=IF)(.*?)(?:CONSIDERING)(.*)(?:THEN)(.*)
    subplan_d = """
        IF you want to boil water CONSIDERING you have a metal pot in inventory with water and a thermometer and you see a stove
        THEN
        deactivate stove
        move metal pot to stove
        activate stove
        examine substance in metal pot
        use thermometer in inventory on substance in metal pot
        examine substance in metal pot
        use thermometer in inventory on substance in metal pot
        examine substance in metal pot
        use thermometer in inventory on substance in metal pot
        examine substance in metal pot
        use thermometer in inventory on substance in metal pot
        examine substance in metal pot
        use thermometer in inventory on substance in metal pot
    """

    import re

    #txt = "The rain in Spain"
    x = re.search(r"(?<=IF)([\S\s]*?)(?:CONSIDERING)([\S\s]*)(?:THEN)([\S\s]*)", subplan_d)
    for g in x.groups():
        part = g.replace('\n', ' ').replace('\t', ' ').strip()
        print(f"Part:\n {part}")
