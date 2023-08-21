all_plans = {'root': ['subtask_a', 'action_b', 'subtask_b'],
             'subtask_a': ['a_action_a', 'a_action_b'],
             'subtask_b': ['c_action_a', 'c_action_b'],
             'a_action_a': ['0', '1', '2']}


def dfs(goal, visited):
    plan = all_plans[goal]
    print(f"plan for {goal} - {plan}")
    for subplan in plan:
        if subplan not in visited:
            # validates plan context
            # entailment = nli_model.check_context(current_belief_base, plan.context)
            # if entailment:
            #   for event in plan.body:
            #       if event is subplan:
            #           visited.append(subplan)
            #           dfs(subplan, visited) validar se deu erro
            #       else:
            #           action = event
            #           obs, info = env.step(action)
            # else:
            #   throw error
            if subplan not in all_plans.keys():
                # executing action
                print(f"leaf -> {subplan}")
            else:
                # find another plan
                visited.append(subplan)
                dfs(subplan, visited)


dfs("root", [])
