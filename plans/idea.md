# Thesis Idea: Fast & Slow KG Agent with LLM-Guided Reasoning

## One-Line Summary

Adapt the Fast & Slow RL architecture (Tan & Motani, 2023) to Knowledge Graph
traversal agents by replacing the hash table with a KG, keeping the MLP fast
mechanism, and adding an LLM in front of the KG to semantically prune candidate
relations before the parallel lookahead runs.

---

## Background: The Original Fast & Slow Architecture

Paper: "Learning, Fast and Slow" -- Tan & Motani, ICDL 2023
arXiv: 2301.13758
Code:  https://github.com/tanchongmin/Learning-Fast-and-Slow

The original system has two mechanisms operating in parallel each step:

**Fast mechanism** -- goal-conditioned MLP
- Input: (cur_state, goal_state)
- Output: probability distribution over actions
- Trained online every step via hippocampal replay (self-supervised)
- Provides directional bias when slow mechanism has no plan yet

**Slow mechanism** -- hash table + parallel lookahead
- Stores transitions: state -> (action, next_state)
- B=100 branches each randomly walk up to D=20 steps through stored transitions
- If any branch reaches the goal, take the first action of the shortest path
- Overrides the fast mechanism when a plan is found

**UCT action selection (Eq. 1)**
```
a* = argmax_a ( p(a) - alpha * sqrt(num_visits(a)) )
```
Penalizes over-visited actions within the current episode.

**Hippocampal replay**
- Past trajectory: (state, goal=next_state, action) for every visited state
- Future trajectory: (state, goal=episode_goal, action) for imagined plan states
- Both batched into one model.fit call per step

**Dynamic adaptation**
- On conflict (same state+action -> different next_state): overwrite memory
- Handles environment changes at episode 50 (obstacle layout switches)

**Key results on dynamic 10x10 grid:**
- Fast & Slow: 92% solve rate
- PPO: 54%, A2C: 24%, DQN: 4.9%
- 4x fewer steps above minimum than all baselines

---

## The Thesis Idea

### Core Claim

The Fast & Slow architecture generalizes to Knowledge Graph reasoning agents.
The biological analogy holds: the KG is the hippocampus (slow, reliable, exact),
the MLP is the neocortex (fast, approximate, learnable), and the LLM is a
semantic reasoning layer that makes the slow mechanism smarter by pruning
irrelevant KG neighbors before the parallel lookahead runs.

### Architecture Mapping

| Original            | Thesis Version                                 |
|---------------------|------------------------------------------------|
| Hash table          | Knowledge Graph (NetworkX / Neo4j)             |
| Grid position       | KG entity (e.g. "Paris", "Einstein")           |
| {Up, Down, L, R}    | Named relation (e.g. "bornIn", "locatedIn")    |
| MLP fast mechanism  | Same MLP, kept as-is, trained via replay       |
| Random minion walk  | LLM-pruned minion walk (new)                   |
| Obstacle change ep50| KG triple corruption at step 50 (new experiment)|

### What Changes

**Only one thing changes in the slow mechanism:**

```python
# ORIGINAL: minion randomly samples from all stored transitions
options = [(ns, a) for a in range(num_actions) for ns in memory[s][a]]
ns, a = random.choice(options)

# THESIS: LLM prunes neighbors before minion samples
neighbors = kg.neighbors(s)
prompt = f"""
Current entity: {s}
Goal entity:    {goal_entity}
Available relations: {neighbors}

Which relations are semantically relevant to reaching the goal?
Return a ranked shortlist.
"""
pruned_neighbors = llm.call(prompt)
ns, a = random.choice(pruned_neighbors)  # minion samples from pruned set
```

**Everything else is identical to the original:**
- MLP fast mechanism with online hippocampal replay
- UCT explore-exploit formula
- Episode memory (visit counts)
- Conflict deletion on KG edge update
- B=100 parallel branches, D=20 depth

### Why Keep the MLP?

The MLP is load-bearing in the original system. It contributes most in two cases:

1. Early in an episode before the slow mechanism has built enough memory to plan
2. When the slow mechanism fails to find a path (queue is empty)

In the KG setting, the KG already exists and is populated from the start, so
case 1 is less critical. But case 2 still matters -- the KG may have gaps, dead
ends, or corrupted regions. The MLP provides a fallback directional signal in
those cases. Removing it would mean pure random UCT when the plan fails, which
the original paper shows degrades solve rate from 92% to 51%.

### Why Add the LLM?

The original minion randomly samples transitions. In a grid world with 4 actions
this is fine -- the branching factor is small. In a KG, a single entity can have
dozens or hundreds of outgoing relations. Random sampling across all of them
wastes most of the B=100 branches on irrelevant paths.

The LLM's job is narrow and well-suited to its strengths: given a current entity,
a goal entity, and a list of candidate relations, rank which relations are
semantically relevant. This is a similarity/relevance judgment, not a planning
task -- exactly what LLMs are good at.

This also directly addresses the main weakness of ToG (Sun et al., 2024): ToG
uses LLM beam search on a static KG with no online learning, no episodic memory,
and no dynamic adaptation. This system has all three.

---

## Novel Contribution: Dynamic Evaluation

This is the centerpiece experiment and the clearest gap over all prior work.

**Protocol:**
1. Take a KGQA benchmark subset (e.g. WebQSP, 200 questions)
2. At question 50, corrupt 20% of answer-supporting triples in the KG
   (delete correct edges, insert plausible-but-wrong ones)
3. Measure re-adaptation speed: how many questions until accuracy recovers?

**Why this matters:**
- Real-world KGs change constantly (Wikidata has ~100k edits/day)
- Every static system (ToG, RoG, SubgraphRAG, GNN-RAG) will degrade and not
  recover -- they have no conflict deletion or online update mechanism
- This system recovers because: (a) conflict deletion removes wrong edges when
  traversal produces contradictions, (b) the MLP updates its directional bias
  via replay, (c) the LLM re-prunes based on updated neighbors

This experiment is a direct analog to the obstacle-change-at-episode-50
experiment in the original paper, which is where Fast & Slow's 92% vs 54% gap
over PPO is most visible.

---

## Comparison to Closest Prior Work

| System       | LLM | KG  | Online Learning | Dynamic Adaptation | Episodic Memory |
|--------------|-----|-----|-----------------|--------------------|-----------------|
| ToG          | yes | yes | no              | no                 | no              |
| RoG          | yes | yes | no              | no                 | no              |
| SubgraphRAG  | yes | yes | no              | no                 | no              |
| GNN-RAG      | no  | yes | no              | no                 | no              |
| **This work**| yes | yes | **yes (MLP)**   | **yes**            | **yes (UCT)**   |

---

## Benchmarks

**Primary (static, for comparison to baselines):**
- WebQSP (1-2 hop, Freebase) -- required for credibility
- CWQ (compositional, Freebase) -- shows multi-hop reasoning
- MetaQA 1/2/3-hop (movie KG) -- clean ablation environment

**Primary (dynamic, novel contribution):**
- WebQSP subset with triple corruption at question 50
- Metric: questions to recovery (lower = better)
- Baselines cannot run this experiment at all

**Metrics:**
- Hits@1, F1 (standard KGQA)
- Questions to recovery after corruption (novel)
- LLM call count per question (efficiency)

---

## Ablations

Directly mirror the original paper's ablation table:

| Variant                  | Expected effect                                    |
|--------------------------|----------------------------------------------------|
| No LLM pruning           | More wasted branches, lower solve rate on dense KGs|
| No MLP (slow only)       | Degrades when KG has gaps or corruption            |
| No slow (MLP only)       | Can't exploit stored KG paths                      |
| No conflict deletion     | Can't recover from KG corruption                   |
| Vary B (branches)        | More branches = better recall, higher cost         |
| Vary D (depth)           | More depth = longer paths found, diminishing return|
| Vary LLM pruning cutoff  | How many neighbors to keep after pruning           |

---

## Risk Register

| Risk                                      | Mitigation                                         |
|-------------------------------------------|----------------------------------------------------|
| LLM pruning removes correct relation      | Keep top-k not top-1; ablate k                     |
| MLP doesn't converge on KG entity space   | Use entity embeddings as input, not raw strings    |
| KG too sparse for slow mechanism to plan  | Fall back to LLM-only mode; report separately      |
| Dynamic experiment too easy / too hard    | Vary corruption rate (10%, 20%, 50%)               |
| LLM API cost too high                     | Cache LLM calls; use local model (Llama 3)         |

---

## References

- Tan & Motani (2023). Learning, Fast and Slow. ICDL. arXiv:2301.13758
- Sun et al. (2024). Think-on-Graph. arXiv:2307.07697
- Luo et al. (2024). Reasoning on Graphs (RoG). arXiv:2310.01061
- Mavromatis & Karypis (2024). GNN-RAG. arXiv:2405.20139
- He et al. (2024). SubgraphRAG. ICLR 2025. arXiv:2406.03051
- McClelland et al. (1995). Why there are complementary learning systems. Psych Review.
- Botvinick et al. (2019). Reinforcement Learning, Fast and Slow. Trends in Cog Sci.
