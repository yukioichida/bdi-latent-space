init_count(1).
!start.

+!start <-
    .call_my_plan("hello").

+!subgoal_a: init_count(X) & X > 0 <-
    .print("subgoal a called");
    !subgoal_a2.

+!subgoal_b: init_count(X) & X > 0 <-
    .print("subgoal b called").

+!subgoal_a2: init_count(X) & X > 0 <-
    .print("subgoal a2 called").

+!my_plan(X) <-
    !subgoal_a;
    !subgoal_b;
    .print("Called with:", X).