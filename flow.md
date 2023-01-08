# memorize observartions
- obs = embedding(nl_obs)
- discretize = f(obs)
- memory.append(discretize)
- for m in memory:
  - for p in plan_library:
    - inference = nli(m, p)
    - if inference is ENTAILMENT:
      - candidate_plans.append(p)
    - else if inference is CONTRADICTION:
      - break
    - else
      - continue
# Diário
- Talvez não usar representações discretas no primeiro momento
  - usar embeddings dos textos como snapshot de memória
  - esses snapshots devem ser usados no modelo de NLI e no RL