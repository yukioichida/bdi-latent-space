import re

from sources.bdi.plans import PlanLibrary, PlanParser

if __name__ == '__main__':
    plan = """
            IF your goal is to boil an water CONSIDERING your are not in the kitchen 
            THEN
            go to the kitchen first, pick the metal pot, fill with water, heat the water
        """

    # plan go to the kitchen
    subplan_a = """
            IF your goal is to go to the kitchen first CONSIDERING you see a door to the kitchen THEN
            open door to the kitchen,
            go to the kitchen
        """

    subplan_b = """
            IF your goal is to pick the metal pot CONSIDERING you are in the kitchen THEN
            open cupboard,
            pick up metal pot
        """

    subplan_c = """
            IF your goal is to fill metal pot with water CONSIDERING you already have a metal pot in your inventory and you see a sink
            THEN
            pour metal pot into sink,
            move metal pot to sink,
            activate sink,
            deactivate sink,
            pick up metal pot
        """

    # regex = (?<=IF)(.*?)(?:CONSIDERING)(.*)(?:THEN)(.*)
    subplan_d = """
            IF your goal is to heat the water CONSIDERING you have a metal pot in inventory with water and a thermometer and you see a stove
            THEN
            deactivate stove,
            move metal pot to stove,
            activate stove,
            examine substance in metal pot,
            use thermometer in inventory on substance in metal pot,
            examine substance in metal pot,
            use thermometer in inventory on substance in metal pot,
            examine substance in metal pot,
            use thermometer in inventory on substance in metal pot,
            examine substance in metal pot,
            use thermometer in inventory on substance in metal pot,
            examine substance in metal pot,
            use thermometer in inventory on substance in metal pot
        """

    parser = PlanParser()

    plan = parser.parse(plan)
    subplan_a = parser.parse(subplan_a)
    subplan_b = parser.parse(subplan_b)
    subplan_c = parser.parse(subplan_c)
    subplan_d = parser.parse(subplan_d)

    plan_library = PlanLibrary(None)
    plan_library.load_plans([plan, subplan_a, subplan_b, subplan_c, subplan_d])
    print(plan_library.subtasks)
    actions = plan_library.get_actions(plan)
    print(actions)


