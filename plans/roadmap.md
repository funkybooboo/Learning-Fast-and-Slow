# Fast & Slow with LLM Pruning: Master's Thesis Roadmap

## High-Level Overview

This thesis adapts the "Learning, Fast and Slow" reinforcement learning architecture to Knowledge Graph reasoning by adding LLM-guided semantic pruning. The original Fast & Slow system combines a fast neural network (MLP) with a slow memory-based planning mechanism, achieving 92% solve rate on dynamic grid navigation tasks—far exceeding standard RL baselines like PPO (54%). We propose to: (1) improve the original grid results by adding LLM pruning to handle high branching factors, and (2) extend the architecture to Knowledge Graph Question Answering (KGQA), where it can leverage both learned patterns and explicit graph structure while adapting to KG changes over time.

**Why this matters:** Current KGQA systems (ToG, RoG, GNN-RAG) are static—they cannot adapt when knowledge graphs change. Real KGs like Wikidata receive 100k+ edits daily. Our system's conflict deletion mechanism enables online adaptation, a capability no existing system has. By validating on standard benchmarks (original grid environment + MetaQA KGQA), we can make direct performance comparisons while demonstrating this novel adaptation capability.

**Thesis scope:** This is designed as a safe, solid master's thesis with clear benchmarks, modular phases that can be paused/resumed, and explicit decision gates. Each phase produces standalone results. The plan assumes sporadic work over an extended timeline (months to years), with detailed resumption guides for picking up after breaks.

---

## Core Contributions

1. **LLM-enhanced Fast & Slow on grids:** Show that LLM semantic pruning improves the original Fast & Slow results on the standard grid benchmark (target: improve 92% → 95%+ solve rate)

2. **Fast & Slow for Knowledge Graphs:** Adapt the dual-system architecture to KGQA, achieving competitive performance on MetaQA standard benchmark (target: >70% Hits@1 on 1-hop, >60% on 2-hop)

3. **Dynamic adaptation capability:** Demonstrate recovery from KG corruption—a novel capability that no existing KGQA system possesses (target: recover to 90% of pre-corruption performance within 50 questions)

---

## Benchmark Strategy

**Why benchmarks matter:** To prove our system is better, we need direct comparisons to published baselines. This plan uses two benchmark tracks:

### Track 1: Grid World (Direct Comparison)
- **Benchmark:** 10×10 dynamic grid from original Fast & Slow paper
- **Baselines:** PPO (54%), A2C (24%), DQN (4.9%), Fast & Slow (92%)
- **Our claim:** "LLM pruning improves Fast & Slow from 92% → X%"
- **Why this works:** Exact same setup, direct comparison, indisputable if we improve

### Track 2: Knowledge Graphs (Separate Contribution)
- **Benchmark:** MetaQA 1-hop and 2-hop (standard KGQA benchmark)
- **Baselines:** ToG (~97% on 1-hop), RoG (~98% on 1-hop)
- **Our claim:** "We match/approach their performance on static benchmarks WHILE enabling dynamic adaptation (which they cannot do)"
- **Caveat:** ToG/RoG are one-shot systems, we use online learning—different settings but same benchmark allows rough comparison

**Success definition:** Strong results on Track 1 (grid improvement) + competitive results on Track 2 (MetaQA) + unique dynamic adaptation capability = solid master's thesis.

---

## Phase Structure Overview

The plan is organized into modular phases, each with clear goals, benchmarks, and decision gates:

- **Phase 0:** Setup & Orientation (1-2 weeks) - Environment setup, literature review, understand baselines
- **Phase 0.5:** Grid Baseline Reproduction (1-2 weeks) - Reproduce original 92% result exactly
- **Phase 1:** Grid + LLM Improvement (2-3 weeks) - Add LLM pruning to grids, show improvement
- **Phase 2:** KG Feasibility (2-3 weeks) - Toy synthetic KG experiment, validate architecture
- **Phase 3:** MetaQA 1-hop (3-4 weeks) - Real KGQA benchmark, competitive performance
- **Phase 4:** Dynamic Adaptation (2-3 weeks) - Corruption experiments, novel capability
- **Phase 5:** MetaQA 2-hop (2-3 weeks) - Multi-hop scaling validation
- **Phase 6:** Growth & Refinement (flexible) - Additional experiments as needed
- **Phase 7:** Write-up (3-5 weeks) - Thesis document
- **Phase 8:** Extensions (optional) - WebQSP, MetaQA 3-hop, publication work

Each phase is self-contained and can be paused for months. Phases 0-5 are the core thesis. Phases 6-8 are flexible/optional.

---

## Phase 0: Setup & Orientation

**Duration:** 1-2 weeks  
**Goal:** Understand the landscape, set up environment, establish baseline knowledge

### Why This Phase Matters

Before writing any code, you need to deeply understand: (1) how the original Fast & Slow algorithm works, (2) what performance numbers you're trying to beat, and (3) what datasets and tools are available. This phase is pure research and setup—no implementation yet. Think of it as "loading context" so you can make informed decisions later.

### Research Questions to Answer

#### Understanding the Original Algorithm
- [ ] **How does the Fast & Slow algorithm work in detail?**
  - [ ] What is the exact UCT formula? `action = argmax(MLP_prob - alpha * sqrt(visit_count))`
  - [ ] How does hippocampal replay work? (backward replay: past trajectory, forward replay: imagined future)
  - [ ] When does conflict deletion trigger? (same state+action → different next state)
  - [ ] What are B (branches) and D (depth) hyperparameters? (B=100 parallel minions, D=20 lookahead depth)
  - [ ] Read original paper (arXiv:2301.13758) and code in this repo thoroughly

#### Understanding KG Reasoning Baselines
- [ ] **What are the current SOTA results on MetaQA?**
  - [ ] ToG (Think-on-Graph): What are their exact Hits@1 numbers on 1/2/3-hop?
  - [ ] RoG (Reasoning on Graphs): Same metrics, what's their approach?
  - [ ] GNN-RAG: How do they combine GNNs with retrieval?
  - [ ] Key insight: What makes these systems work well? (LLM reasoning, graph structure, both?)
  - [ ] **Action:** Create a table of baseline results to target

#### Understanding Entity Embeddings
- [ ] **What KG embeddings exist and which are suitable?**
  - [ ] TransE, ComplEx, RotatE: What are the differences?
  - [ ] Which is best for MetaQA (movie KG) vs Freebase (general KG)?
  - [ ] Are pre-trained embeddings available for MetaQA?
  - [ ] Can we use LLM-based embeddings (BERT entity names) as alternative?
  - [ ] **Decision point:** Which embedding approach to start with in Phase 2?

#### Understanding MetaQA Dataset
- [ ] **How is MetaQA structured?**
  - [ ] How many entities, relations, triples in the KG?
  - [ ] What's the format of 1-hop questions? (e.g., "Who directed [movie]?")
  - [ ] What's the train/val/test split? (need exact numbers for reproducibility)
  - [ ] How many questions per hop level? (1-hop, 2-hop, 3-hop)
  - [ ] **Action:** Download and inspect the data files

### Implementation Tasks

#### Environment Setup
- [ ] **Set up Python environment**
  - [ ] Create new virtual environment: `python -m venv venv` or `conda create -n fast-slow-kg python=3.10`
  - [ ] Install core dependencies:
    - [ ] NetworkX (for KG graph structure)
    - [ ] PyTorch (for MLP neural network)
    - [ ] transformers (for LLM access)
    - [ ] openai (for GPT API)
    - [ ] matplotlib, pandas, tqdm (utilities)
  - [ ] Update requirements.txt with all dependencies
  - [ ] Test that original Fast&Slow.ipynb runs (verify environment works)

#### Dataset Acquisition
- [ ] **Download MetaQA dataset**
  - [ ] Find official source (likely GitHub repo or paper website)
  - [ ] Download 1-hop, 2-hop, 3-hop question files
  - [ ] Download MetaQA KG triples file
  - [ ] Verify data integrity (check file sizes match expected)
  - [ ] **Action:** Document download source and file structure in `datasets.md`

- [ ] **Write MetaQA data loader**
  - [ ] Parse KG triples: (head_entity, relation, tail_entity)
  - [ ] Parse questions: (question_text, answer_entity)
  - [ ] Test loader: print sample questions and verify format
  - [ ] **Deliverable:** `data_loader.py` with functions to load KG and questions

#### Code Analysis
- [ ] **Analyze original Fast & Slow code**
  - [ ] Read through `Fast&Slow.ipynb` cell by cell
  - [ ] Extract key functions:
    - [ ] `minion()` - parallel lookahead search
    - [ ] `GetModel()` - MLP architecture
    - [ ] UCT action selection formula
    - [ ] Hippocampal replay training
    - [ ] Conflict deletion logic
  - [ ] Document what can be reused vs what needs rewriting for KG
  - [ ] **Decision:** Refactor existing code or start fresh? (Recommendation: start fresh for cleaner architecture, but reference original heavily)
  - [ ] **Deliverable:** `original_code_analysis.md` with findings

#### LLM Access Setup
- [ ] **Set up LLM API access**
  - [ ] Get OpenAI API key (for GPT-3.5/GPT-4)
  - [ ] Test basic API call: `openai.ChatCompletion.create(...)`
  - [ ] Test relation pruning prompt:
    ```
    "Given current entity: Paris, goal entity: France, available relations: [locatedIn, capitalOf, population], 
    which relations help navigate from Paris to France? Return top 3."
    ```
  - [ ] Verify response parsing works
  - [ ] Research local model options: Llama 3.1, Qwen, Mistral (for later cost reduction)
  - [ ] **Estimate costs:** ~2000 MetaQA questions × ~10 hops × 1 API call = 20k calls × $0.002 = ~$40 for full experiments

### Deliverables

- [ ] **`literature_review.md`** - Summary of key papers (Fast & Slow, ToG, RoG) with baseline numbers
- [ ] **`datasets.md`** - MetaQA statistics, data format, download instructions
- [ ] **`original_code_analysis.md`** - What to reuse/rewrite from original repo
- [ ] **`data_loader.py`** - Working MetaQA data loader
- [ ] **LLM API test script** - Verify API access works

### Success Criteria

✅ Can load and parse MetaQA data (print 10 sample questions)  
✅ Understand original algorithm well enough to explain to someone else  
✅ Have table of baseline numbers to target (ToG/RoG results on MetaQA)  
✅ LLM API working with test prompt  
✅ Environment set up and original code runs

### Resumption Guide (If Returning After Break)

**Where you left off:** Phase 0 is pure setup and research—no implementation yet.

**To resume:**
1. Re-read `literature_review.md` to refresh on Fast & Slow algorithm
2. Check `datasets.md` to remember MetaQA structure
3. Test that environment still works: `python data_loader.py`
4. Review baseline numbers table—these are your targets
5. Proceed to Phase 0.5 (grid reproduction)

---

## Phase 0.5: Grid Baseline Reproduction

**Duration:** 1-2 weeks  
**Goal:** Exactly reproduce the original Fast & Slow 92% solve rate on the grid benchmark

### Why This Phase Matters

Before claiming "we improved Fast & Slow," you must first prove you can reproduce the original results exactly. This validates that: (1) you understand the algorithm correctly, (2) your implementation is faithful to the original, and (3) you have a reliable baseline to compare against. This is a critical checkpoint—if you can't reproduce 92%, something is wrong and you need to debug before proceeding.

**Benchmark:** 10×10 grid world with dynamic obstacles (obstacle layout switches at episode 50)  
**Target:** 92% solve rate over 100 episodes (matching original paper Table 1)  
**Baselines to compare:** PPO (54%), A2C (24%), DQN (4.9%), Fast & Slow (92%)

### Implementation Tasks

#### Subtask 0.5.1: Verify Original Code Works
- [ ] **Run original Fast&Slow.ipynb**
  - [ ] Execute all cells in the notebook
  - [ ] Verify it produces ~92% solve rate on 10×10 grid
  - [ ] Note: May need to adjust for library version differences (TensorFlow, gym)
  - [ ] If it doesn't work, debug environment issues first

#### Subtask 0.5.2: Extract and Refactor Grid Environment
- [ ] **Extract grid environment from notebook**
  - [ ] Copy `GridEnv` class to standalone file: `grid_environment.py`
  - [ ] Test that it works: create env, run random actions, verify state transitions
  - [ ] Document environment:
    - [ ] State space: (cur_x, cur_y, goal_x, goal_y)
    - [ ] Action space: {left, up, right, down}
    - [ ] Reward: +1 if goal reached, 0 otherwise
    - [ ] Forbidden squares: obstacles that block movement
    - [ ] Dynamic change: obstacles switch at episode 50

#### Subtask 0.5.3: Extract and Refactor Fast & Slow Components
- [ ] **Extract MLP model**
  - [ ] Copy `GetModel()` function to `fast_mechanism.py`
  - [ ] Architecture: Input(4) → Dense(128, ReLU) → Dense(128, ReLU) → Dense(4, softmax)
  - [ ] Test: create model, run forward pass, verify output shape

- [ ] **Extract minion search**
  - [ ] Copy `minion()` function to `slow_mechanism.py`
  - [ ] Implement memory structure: `defaultdict(lambda: defaultdict(lambda: []))`
  - [ ] Test: add some transitions to memory, run minion search, verify it finds paths

- [ ] **Extract UCT action selection**
  - [ ] Implement: `action = argmax(MLP_prob - alpha * sqrt(visit_count))`
  - [ ] Track visit counts per episode in `episode_memory`

- [ ] **Extract hippocampal replay**
  - [ ] Backward replay: `(state, next_state_as_goal, action_taken)` for all visited states
  - [ ] Forward replay: `(imagined_state, episode_goal, imagined_action)` for minion paths
  - [ ] Batch and train MLP every step

- [ ] **Extract conflict deletion**
  - [ ] If `memory[state][action]` has entry != new_next_state, clear and replace
  - [ ] This enables adaptation when environment changes at episode 50

#### Subtask 0.5.4: Run Reproduction Experiment
- [ ] **Set up experiment**
  - [ ] 10×10 grid
  - [ ] Random start and goal each episode
  - [ ] Obstacles: horizontal line at episode 0-49, vertical line at episode 50-99
  - [ ] 100 episodes total
  - [ ] Hyperparameters: B=100 branches, D=20 depth, alpha=1.0

- [ ] **Run Fast & Slow**
  - [ ] Track solve rate (% episodes reaching goal)
  - [ ] Track steps per episode
  - [ ] Track steps above minimum (efficiency metric)
  - [ ] **Target:** 92% solve rate (match original paper)

- [ ] **Run baselines (optional but recommended)**
  - [ ] PPO: Use stable-baselines3 (already in requirements.txt)
  - [ ] A2C: Use stable-baselines3
  - [ ] Random: Random action selection
  - [ ] **Why:** Confirms your environment is correct if baselines match paper

#### Subtask 0.5.5: Analyze Results
- [ ] **Create plots**
  - [ ] Solve rate over 100 episodes (should stay high, dip slightly at episode 50, recover)
  - [ ] Steps per episode (should be close to minimum)
  - [ ] Compare to baselines (Fast & Slow should dominate)

- [ ] **Verify reproduction**
  - [ ] Did you achieve ~92% solve rate? (within ±3% is acceptable)
  - [ ] Did performance dip at episode 50 then recover? (shows adaptation works)
  - [ ] Are steps above minimum low? (shows efficiency)

### Deliverables

- [ ] **`grid_environment.py`** - Standalone grid environment
- [ ] **`fast_mechanism.py`** - MLP model with hippocampal replay
- [ ] **`slow_mechanism.py`** - Memory + minion search
- [ ] **`grid_reproduction.py`** - Script to run reproduction experiment
- [ ] **`phase0.5_results.ipynb`** - Plots and analysis
- [ ] **`phase0.5_report.md`** - Did we reproduce 92%? Any issues?

### Success Criteria

✅ Achieve 90-95% solve rate on grid (within range of original 92%)  
✅ Performance dips at episode 50 then recovers (adaptation works)  
✅ Steps above minimum is low (<10% above optimal)  
✅ Code is clean and modular (ready to extend with LLM)

### 🚦 DECISION GATE 0.5

**Go:** Successfully reproduced ~92% → Proceed to Phase 1 (add LLM to grids)  
**Debug:** Achieved 70-90% → Close but not quite, debug and iterate  
**No-Go:** <70% solve rate → Major implementation error, must fix before proceeding

**If No-Go:** Compare your implementation line-by-line with original notebook. Check:
- Is UCT formula correct?
- Is hippocampal replay training the MLP properly?
- Are minions finding paths in memory?
- Is conflict deletion working at episode 50?

### Resumption Guide (If Returning After Break)

**Where you left off:** You've reproduced the original grid results (or are debugging).

**To resume:**
1. Check `phase0.5_report.md` - did you achieve 92%?
2. If yes: Review code in `fast_mechanism.py` and `slow_mechanism.py` - you'll extend these in Phase 1
3. If no: Review debugging notes, continue fixing implementation
4. Re-run `grid_reproduction.py` to verify results are reproducible
5. Proceed to Phase 1 (add LLM pruning)

---

## Phase 1: Grid + LLM Improvement

**Duration:** 2-3 weeks  
**Goal:** Add LLM semantic pruning to grid world, show improvement over original 92%

### Why This Phase Matters

This is your first contribution: "LLM pruning improves Fast & Slow." The grid world is a controlled environment where you can isolate the effect of LLM pruning without the complexity of KGs. If LLM pruning helps on grids (where branching factor is only 4), it provides strong evidence it will help on KGs (where branching factor is 50-100). This phase also lets you develop and test your LLM integration before tackling KGs.

**Benchmark:** Same 10×10 grid as Phase 0.5  
**Baseline:** Fast & Slow (92% from Phase 0.5)  
**Target:** Improve to 95%+ solve rate OR reduce steps by 20%+

### Research Questions to Answer

- [ ] **Can LLM provide useful directional guidance on grids?**
  - [ ] Given current position (3, 5) and goal (7, 2), can LLM suggest "move right and down"?
  - [ ] Is this better than MLP's learned directional bias?

- [ ] **Where should LLM pruning happen?**
  - [ ] Option 1: Only at root (1 LLM call per agent action)
  - [ ] Option 2: Every minion step (many LLM calls, expensive)
  - [ ] Option 3: Adaptive (only when MLP is uncertain)

- [ ] **What's the computational cost?**
  - [ ] LLM calls per episode
  - [ ] Wall-clock time per episode
  - [ ] API cost for 100 episodes

### Implementation Tasks

#### Subtask 1.1: Design LLM Pruning for Grids
- [ ] **Create LLM pruning function**
  - [ ] Input: current position (x, y), goal position (gx, gy), available actions [left, up, right, down]
  - [ ] Prompt design:
    ```
    "You are navigating a 10x10 grid. Current position: (3, 5). Goal position: (7, 2).
    Available actions: left, up, right, down.
    Which actions move you closer to the goal? Rank them from best to worst."
    ```
  - [ ] Parse LLM response to get ranked action list
  - [ ] Return top-k actions (start with k=2, meaning keep 2 out of 4 actions)

- [ ] **Test LLM pruning**
  - [ ] Test on 10 sample (current, goal) pairs
  - [ ] Verify LLM suggests reasonable directions (e.g., "right" when goal is to the right)
  - [ ] Check for hallucinations (LLM inventing actions not in the list)

#### Subtask 1.2: Integrate LLM Pruning with Minions
- [ ] **Modify minion search**
  - [ ] Before sampling action: call LLM pruning to get top-k actions
  - [ ] Sample only from pruned action set
  - [ ] If pruned set is empty (shouldn't happen), fall back to all actions

- [ ] **Implement pruning strategies**
  - [ ] **Strategy 1: Root-only pruning**
    - [ ] Call LLM once at start of episode to prune root action
    - [ ] Minions use unpruned actions for subsequent steps
    - [ ] Cost: 1 LLM call per episode
  
  - [ ] **Strategy 2: Every-step pruning**
    - [ ] Call LLM at every minion step
    - [ ] Cost: B × D LLM calls per episode (100 × 20 = 2000 calls!)
    - [ ] Likely too expensive, but test for comparison
  
  - [ ] **Strategy 3: Adaptive pruning**
    - [ ] Call LLM only when MLP probability is low (e.g., max_prob < 0.4)
    - [ ] Indicates MLP is uncertain, LLM can help
    - [ ] Cost: Variable, depends on MLP confidence

- [ ] **Add LLM caching**
  - [ ] Cache LLM responses: `cache[(cur_pos, goal_pos)] = ranked_actions`
  - [ ] Avoid redundant API calls for same (current, goal) pair
  - [ ] Especially important for every-step pruning

#### Subtask 1.3: Run Grid + LLM Experiments
- [ ] **Baseline (from Phase 0.5)**
  - [ ] Fast & Slow without LLM: 92% solve rate

- [ ] **Ablation 1: Root-only LLM pruning**
  - [ ] Fast & Slow + LLM (root only)
  - [ ] Measure: solve rate, steps, LLM calls

- [ ] **Ablation 2: Adaptive LLM pruning**
  - [ ] Fast & Slow + LLM (adaptive)
  - [ ] Measure: solve rate, steps, LLM calls

- [ ] **Ablation 3: Every-step LLM pruning (optional)**
  - [ ] Fast & Slow + LLM (every step)
  - [ ] Likely too expensive, but useful for comparison
  - [ ] Measure: solve rate, steps, LLM calls

- [ ] **Metrics to collect**
  - [ ] Solve rate (% episodes reaching goal)
  - [ ] Steps per episode (efficiency)
  - [ ] Steps above minimum
  - [ ] LLM calls per episode
  - [ ] Wall-clock time per episode
  - [ ] API cost estimate

#### Subtask 1.4: Analyze Results
- [ ] **Performance comparison**
  - [ ] Table: Solve rate for each configuration
  - [ ] Did LLM pruning improve over 92% baseline?
  - [ ] Which pruning strategy is best? (likely root-only or adaptive)

- [ ] **Cost-performance tradeoff**
  - [ ] Plot: X-axis = LLM calls per episode, Y-axis = solve rate
  - [ ] Identify optimal configuration (best performance per LLM call)

- [ ] **Qualitative analysis**
  - [ ] Sample 5 episodes: visualize agent path with and without LLM
  - [ ] Did LLM pruning help avoid dead ends?
  - [ ] Any cases where LLM pruning hurt performance?

### Deliverables

- [ ] **`llm_pruning.py`** - LLM pruning function with caching
- [ ] **`grid_llm_experiment.py`** - Script to run grid + LLM experiments
- [ ] **`phase1_results.ipynb`** - Plots and analysis
- [ ] **`phase1_report.md`** - Did LLM improve grid performance? By how much?

### Success Criteria

✅ LLM pruning improves solve rate (92% → 95%+) OR reduces steps by 20%+  
✅ Root-only or adaptive pruning is computationally tractable (<10 LLM calls per episode)  
✅ Understand which pruning strategy works best and why  
✅ Have working LLM integration ready to extend to KGs

### 🚦 DECISION GATE 1

**Go:** LLM pruning clearly helps (≥3% improvement) → Proceed to Phase 2 (KG feasibility)  
**Pivot:** LLM pruning helps slightly (1-2% improvement) → Still proceed, but lower expectations for KG  
**No-Go:** LLM pruning doesn't help or hurts performance → Reconsider thesis, may need to pivot to "Fast & Slow for KG without LLM"

**If No-Go:** Possible reasons LLM didn't help:
- Grid branching factor is too low (only 4 actions) - LLM overhead not worth it
- LLM prompt is poorly designed - try different prompts
- LLM is giving bad advice - check sample responses
- **Decision:** Either fix LLM integration or pivot to "Fast & Slow for KG" without LLM pruning

### Resumption Guide (If Returning After Break)

**Where you left off:** You've tested LLM pruning on grids.

**To resume:**
1. Check `phase1_report.md` - did LLM improve performance?
2. Review best configuration (likely root-only pruning with k=2)
3. Re-read `llm_pruning.py` to understand implementation
4. If successful: Proceed to Phase 2 (KG feasibility)
5. If unsuccessful: Review decision gate options

---

## Phase 2: KG Feasibility (Toy Experiment)

**Duration:** 2-3 weeks  
**Goal:** Validate that Fast & Slow + LLM works on a synthetic Knowledge Graph

### Why This Phase Matters

Before tackling real KGQA benchmarks (MetaQA), you need to validate the core architecture on a controlled synthetic KG. This lets you test: (1) Can MLP learn with entity embeddings instead of grid coordinates? (2) Does LLM pruning help with higher branching factor (10-20 relations vs 4 actions)? (3) Does conflict deletion work for KG edge changes? This is a critical go/no-go checkpoint—if the architecture doesn't work on a toy KG, it won't work on MetaQA.

**Benchmark:** Synthetic KG with 100 entities, 5 relation types, known structure  
**Target:** >90% solve rate on navigation tasks (entity A → entity B)

### Research Questions to Answer

- [ ] **Does MLP learn directional bias with entity embeddings?**
  - [ ] Grid: MLP learns "move toward goal coordinates" (easy, continuous space)
  - [ ] KG: MLP learns "move toward goal entity embedding" (harder, discrete space)
  - [ ] Will MLP converge or just output random probabilities?

- [ ] **Does LLM pruning help with higher branching factor?**
  - [ ] Grid: 4 actions, LLM pruning marginal benefit
  - [ ] Toy KG: 10-20 relations per entity, LLM pruning should help more

- [ ] **Can conflict deletion detect and recover from KG changes?**
  - [ ] Corrupt KG after 50 episodes (delete edges, add wrong edges)
  - [ ] Does system recover via conflict deletion?

### Implementation Tasks

#### Subtask 2.1: Create Synthetic KG Environment
- [ ] **Design synthetic KG**
  - [ ] 100 entities: `Entity_001` to `Entity_100`
  - [ ] 5 relation types: `North`, `South`, `East`, `West`, `Shortcut`
  - [ ] Structure: Grid-like layout (entities arranged in 10×10 grid) + random shortcuts
  - [ ] Example: `Entity_023` --North--> `Entity_013` (move up one row)
  - [ ] Random shortcuts: `Entity_005` --Shortcut--> `Entity_087` (teleport)
  - [ ] **Why this structure:** Grid-like makes it easy to verify correctness, shortcuts add complexity

- [ ] **Implement KG as NetworkX graph**
  - [ ] Nodes: entities
  - [ ] Edges: (entity, relation, next_entity)
  - [ ] Function: `get_neighbors(entity) -> [(relation, next_entity), ...]`
  - [ ] Test: print neighbors of 5 random entities, verify structure

- [ ] **Create gym-like environment**
  - [ ] State: (current_entity, goal_entity)
  - [ ] Action: choose relation (not entity—minion will sample entity from memory)
  - [ ] Step: transition to next entity via chosen relation
  - [ ] Reward: +1 if goal reached, 0 otherwise
  - [ ] Done: goal reached or max steps (20) exceeded
  - [ ] **Deliverable:** `toy_kg_environment.py`

#### Subtask 2.2: Implement Entity Embeddings
- [ ] **Choose embedding strategy**
  - [ ] **Option A: Position-based** (recommended for toy KG)
    - [ ] Since entities are in 10×10 grid, use (row, col) as 2D embedding
    - [ ] Simple, interpretable, MLP should learn easily
  
  - [ ] **Option B: Random init, fixed**
    - [ ] Random 64-dim vectors for each entity
    - [ ] Tests if MLP can learn without positional info
  
  - [ ] **Option C: Learned embeddings**
    - [ ] Initialize random, update via gradient descent
    - [ ] More complex, defer to later if needed

- [ ] **Implement embedding lookup**
  - [ ] `get_embedding(entity) -> vector`
  - [ ] MLP input: `concat(cur_entity_emb, goal_entity_emb)`
  - [ ] Test: verify embedding shapes are correct

#### Subtask 2.3: Adapt Fast & Slow Components for KG
- [ ] **Update MLP architecture**
  - [ ] Input: concat(cur_entity_emb, goal_entity_emb) - shape depends on embedding dim
  - [ ] Hidden: Dense(128, ReLU) → Dense(128, ReLU)
  - [ ] Output: softmax over relation types (5 relations in toy KG)
  - [ ] **Key difference from grid:** Output is over relations, not entities

- [ ] **Update minion search**
  - [ ] Sample relation from MLP output or LLM-pruned set
  - [ ] Sample entity from `memory[current_entity][relation]`
  - [ ] Random walk up to D steps (start with D=10 for toy)
  - [ ] Return first action if path to goal found

- [ ] **Update UCT action selection**
  - [ ] Same formula: `action = argmax(MLP_prob - alpha * sqrt(visit_count))`
  - [ ] But "action" is now a relation, not a direction

- [ ] **Update hippocampal replay**
  - [ ] Backward replay: `(visited_entity, next_entity_as_goal, relation_taken)`
  - [ ] Forward replay: `(imagined_entity, episode_goal, imagined_relation)` from minion paths
  - [ ] Train MLP online every step

- [ ] **Conflict deletion (same as grid)**
  - [ ] If `memory[entity][relation]` has entry != new_next_entity, clear and replace

#### Subtask 2.4: Add LLM Pruning for KG
- [ ] **Design LLM prompt for toy KG**
  - [ ] Input: current_entity, goal_entity, available relations
  - [ ] Prompt:
    ```
    "You are navigating a knowledge graph. Current entity: Entity_023. Goal entity: Entity_087.
    Available relations: [North, South, East, West, Shortcut].
    Which relations help navigate from Entity_023 to Entity_087? Rank them."
    ```
  - [ ] **Challenge:** LLM doesn't know the graph structure, so it can only reason about relation names
  - [ ] For toy KG, relation names are directional (North, South, etc.), so LLM can reason spatially

- [ ] **Implement pruning strategies**
  - [ ] Root-only: 1 LLM call per episode
  - [ ] Adaptive: Call LLM when branching factor >10 or MLP uncertain
  - [ ] Cache LLM responses

#### Subtask 2.5: Run Toy KG Experiments
- [ ] **Baseline: Random action selection**
  - [ ] Measure: solve rate, steps per episode

- [ ] **Ablation 1: Slow only (no MLP)**
  - [ ] Minions search memory, no MLP guidance
  - [ ] Measure: solve rate, steps

- [ ] **Ablation 2: Fast only (no minions)**
  - [ ] MLP chooses relation, no memory-based planning
  - [ ] Measure: solve rate, steps

- [ ] **Ablation 3: Fast + Slow (no LLM)**
  - [ ] Full system without LLM pruning
  - [ ] Measure: solve rate, steps

- [ ] **Ablation 4: Fast + Slow + LLM**
  - [ ] Full system with LLM pruning (root-only)
  - [ ] Measure: solve rate, steps, LLM calls

- [ ] **Dynamic adaptation test**
  - [ ] Run 100 episodes
  - [ ] After episode 50: corrupt 10% of edges (delete some, add wrong ones)
  - [ ] Measure: does solve rate dip then recover?

#### Subtask 2.6: Analyze Results
- [ ] **Performance comparison**
  - [ ] Table: Solve rate for each ablation
  - [ ] Does MLP learn? (Fast+Slow > Slow-only?)
  - [ ] Does LLM help? (Fast+Slow+LLM > Fast+Slow?)

- [ ] **MLP convergence analysis**
  - [ ] Plot: MLP action prediction accuracy over episodes
  - [ ] Is MLP learning directional bias or just random?
  - [ ] If not learning: try different embeddings or increase training

- [ ] **Dynamic adaptation analysis**
  - [ ] Plot: Solve rate over 100 episodes (should dip at 50, then recover)
  - [ ] How many episodes to recover to 90% of pre-corruption performance?

### Deliverables

- [ ] **`toy_kg_environment.py`** - Synthetic KG environment
- [ ] **`entity_embeddings.py`** - Embedding utilities
- [ ] **`kg_fast_mechanism.py`** - MLP adapted for KG
- [ ] **`kg_slow_mechanism.py`** - Minion search adapted for KG
- [ ] **`toy_kg_experiment.py`** - Script to run experiments
- [ ] **`phase2_results.ipynb`** - Plots and analysis
- [ ] **`phase2_report.md`** - Findings and recommendations

### Success Criteria

✅ MLP learns better than random (action prediction accuracy >60%)  
✅ System solves >90% of toy KG navigation tasks  
✅ LLM pruning improves performance (at least +10% solve rate)  
✅ Conflict deletion enables recovery after corruption (solve rate returns to >80% within 20 episodes)  
✅ Computational cost is tractable (<1 min per episode)

### 🚦 DECISION GATE 2

**Go:** All success criteria met → Proceed to Phase 3 (MetaQA 1-hop)  
**Pivot:** Some criteria failed → Adjust architecture before MetaQA:
  - If MLP not learning: Try different embeddings or remove MLP
  - If LLM not helping: Adjust prompt or pruning strategy
  - If adaptation not working: Debug conflict deletion logic  
**No-Go:** Multiple failures → Architecture may not be viable for KG, major rethink needed

**Key Questions to Resolve Before Phase 3:**
- Which entity embedding approach works best?
- Which LLM pruning strategy is optimal?
- Is the MLP necessary or redundant with LLM?
- What hyperparameters (B, D, alpha, k) should we use?

### Resumption Guide (If Returning After Break)

**Where you left off:** You've tested Fast & Slow on a synthetic KG.

**To resume:**
1. Check `phase2_report.md` - did all success criteria pass?
2. Review best configuration (embeddings, pruning strategy, hyperparameters)
3. Re-read code in `kg_fast_mechanism.py` and `kg_slow_mechanism.py`
4. If successful: Proceed to Phase 3 (MetaQA 1-hop)
5. If issues: Review decision gate and address failures

---

## Phase 3: MetaQA 1-hop Benchmark

**Duration:** 3-4 weeks  
**Goal:** Achieve competitive performance on MetaQA 1-hop standard benchmark

### Why This Phase Matters

This is where you prove your system works on a real KGQA benchmark. MetaQA 1-hop is the simplest multi-hop reasoning task (answer is 1 hop from question entity), making it the perfect starting point. Your goal is to achieve >70% Hits@1, which is competitive with baselines (ToG ~97%, RoG ~98%). Note that ToG/RoG are one-shot systems while you use online learning, so direct comparison isn't apples-to-apples, but being in the same ballpark validates your approach.

**Benchmark:** MetaQA 1-hop (standard train/val/test split)  
**Baselines:** ToG (~97% Hits@1), RoG (~98% Hits@1)  
**Target:** >70% Hits@1 (competitive, given different setup)  
**Metrics:** Hits@1, Hits@3, F1, average steps, LLM calls per question

### Research Questions to Answer

- [ ] **How to convert KGQA to episodic RL format?**
  - [ ] Episode = one question
  - [ ] Start state = question entity (e.g., "Inception" for "Who directed Inception?")
  - [ ] Goal state = answer entity (e.g., "Christopher Nolan")
  - [ ] How to extract question entity from question text?

- [ ] **What entity embeddings work for MetaQA?**
  - [ ] MetaQA is a movie KG (entities: movies, actors, directors)
  - [ ] Options: TransE, ComplEx, BERT-based, or train from scratch?

- [ ] **Should we pre-train on random KG exploration?**
  - [ ] Option A: Pure online learning from questions
  - [ ] Option B: Pre-train on random (start, goal) pairs, then finetune on questions
  - [ ] Which is better?

### Implementation Tasks

#### Subtask 3.1: MetaQA Data Pipeline
- [ ] **Parse MetaQA KG**
  - [ ] Load triples file: (head_entity, relation, tail_entity)
  - [ ] Build NetworkX graph (or dict-based structure for speed)
  - [ ] Verify graph connectivity: no isolated components
  - [ ] Statistics: How many entities? Relations? Triples?
  - [ ] **Deliverable:** `metaqa_kg.py` with KG loading functions

- [ ] **Parse MetaQA 1-hop questions**
  - [ ] Format: "Who directed [movie]?" → answer: director entity
  - [ ] Extract question entity (e.g., "Inception")
  - [ ] Map entity name to KG entity ID
  - [ ] Handle multi-answer questions (some questions have >1 correct answer)
  - [ ] Load train/val/test splits
  - [ ] **Deliverable:** `metaqa_questions.py` with question parsing

- [ ] **Create KGQA environment**
  - [ ] State: (current_entity, goal_entity)
  - [ ] Action: choose relation
  - [ ] Step: transition to next entity
  - [ ] Reward: +1 if goal reached (or any correct answer for multi-answer questions)
  - [ ] Done: goal reached or max steps (10 for 1-hop)
  - [ ] **Deliverable:** `metaqa_environment.py`

#### Subtask 3.2: Entity Embeddings for MetaQA
- [ ] **Choose embedding approach**
  - [ ] **Option 1: Train TransE on MetaQA KG**
    - [ ] Use PyKEEN library: `pip install pykeen`
    - [ ] Train TransE: 100-200 epochs, embedding dim 128
    - [ ] Save embeddings to file
    - [ ] Pro: Captures KG structure
    - [ ] Con: Takes time to train (hours)
  
  - [ ] **Option 2: Use entity names + BERT**
    - [ ] Entity name → BERT embedding (768-dim)
    - [ ] Pro: No training needed, captures semantic meaning
    - [ ] Con: Doesn't capture KG structure
  
  - [ ] **Option 3: Random embeddings**
    - [ ] Control baseline
    - [ ] Should perform worse than TransE/BERT

- [ ] **Implement embedding lookup**
  - [ ] `get_embedding(entity_id) -> vector`
  - [ ] Handle out-of-vocabulary entities (shouldn't happen if KG is complete)
  - [ ] **Deliverable:** `metaqa_embeddings.py`

#### Subtask 3.3: Adapt System for MetaQA
- [ ] **Update MLP**
  - [ ] Input: concat(cur_entity_emb, goal_entity_emb)
  - [ ] Output: softmax over MetaQA relation types (~10 relations)
  - [ ] Train via hippocampal replay (same as toy KG)

- [ ] **Update LLM pruning prompt**
  - [ ] Include relation names: "directed_by", "starred_actors", "written_by", etc.
  - [ ] Include question context (optional): "To answer 'Who directed Inception?', which relation is most relevant?"
  - [ ] Test prompt on 10 sample questions, verify LLM suggests correct relations

- [ ] **Implement training loop**
  - [ ] Iterate through train questions (in order or shuffled?)
  - [ ] For each question: run episode, update MLP, store in memory
  - [ ] Track cumulative memory size over time
  - [ ] Evaluate on val set every N questions (e.g., every 100)

#### Subtask 3.4: Run MetaQA 1-hop Experiments

**Start with minimal ablations (3-4 configs), grow as needed:**

- [ ] **Baseline: Random**
  - [ ] Random relation selection
  - [ ] Measure: Hits@1, steps

- [ ] **Ablation 1: Slow only (no MLP)**
  - [ ] Memory + minion search, no MLP guidance
  - [ ] Measure: Hits@1, steps

- [ ] **Ablation 2: Fast + Slow (no LLM)**
  - [ ] Full system without LLM pruning
  - [ ] Measure: Hits@1, steps

- [ ] **Ablation 3: Fast + Slow + LLM**
  - [ ] Full system with LLM pruning (root-only)
  - [ ] Measure: Hits@1, steps, LLM calls

**If time permits and results are promising, add:**

- [ ] **Ablation 4: Fast only (no minions)**
  - [ ] MLP only, no memory-based planning
  - [ ] Measure: Hits@1, steps

- [ ] **Ablation 5: Different LLM pruning strategies**
  - [ ] Root-only vs adaptive
  - [ ] Measure: Hits@1, LLM calls

#### Subtask 3.5: Analyze Results
- [ ] **Performance comparison**
  - [ ] Table: Hits@1 for each ablation
  - [ ] Compare to baselines: ToG (~97%), RoG (~98%)
  - [ ] Are we competitive (>70%)? If not, why?

- [ ] **Component contribution**
  - [ ] Does MLP help? (Fast+Slow > Slow-only?)
  - [ ] Does LLM help? (Fast+Slow+LLM > Fast+Slow?)
  - [ ] Quantify improvement from each component

- [ ] **Learning curve**
  - [ ] Plot: Hits@1 over training questions
  - [ ] Does performance improve over time (online learning works)?
  - [ ] When does it plateau?

- [ ] **Error analysis**
  - [ ] Sample 20 failed questions
  - [ ] Why did they fail?
    - [ ] Minions never found path (memory gaps)
    - [ ] LLM pruned correct relation
    - [ ] MLP suggested wrong direction
    - [ ] Max steps exceeded

### Deliverables

- [ ] **`metaqa_kg.py`** - MetaQA KG loading
- [ ] **`metaqa_questions.py`** - Question parsing
- [ ] **`metaqa_environment.py`** - KGQA environment
- [ ] **`metaqa_embeddings.py`** - Entity embeddings
- [ ] **`metaqa_train.py`** - Training loop
- [ ] **`metaqa_evaluate.py`** - Evaluation script
- [ ] **`phase3_results.ipynb`** - Plots and analysis
- [ ] **`phase3_report.md`** - Findings and recommendations

### Success Criteria

✅ Achieve >70% Hits@1 on MetaQA 1-hop (competitive with baselines)  
✅ Fast+Slow outperforms Slow-only (MLP helps)  
✅ Fast+Slow+LLM outperforms Fast+Slow (LLM helps)  
✅ System is computationally tractable (<5 min per question)  
✅ Understand failure modes and limitations

### 🚦 DECISION GATE 3

**Go:** Achieved >70% Hits@1, clear benefit from all components → Proceed to Phase 4 (dynamic adaptation)  
**Pivot:** Achieved 50-70% Hits@1, some components not helping → Simplify architecture, run more ablations  
**No-Go:** <50% Hits@1 → Major issues, need to debug or reconsider approach

**If Pivot or No-Go:**
- Check MLP convergence: Is it learning or outputting random probabilities?
- Check LLM pruning: Is it helping or hurting? Try different prompts
- Check entity embeddings: Try TransE if using BERT, or vice versa
- Check training protocol: Try pre-training on random KG exploration
- Check memory: Is it accumulating useful transitions?

### Resumption Guide (If Returning After Break)

**Where you left off:** You've run experiments on MetaQA 1-hop.

**To resume:**
1. Check `phase3_report.md` - what was your best Hits@1?
2. Review best configuration (embeddings, pruning strategy, hyperparameters)
3. Re-read error analysis - what are the main failure modes?
4. If successful (>70%): Proceed to Phase 4 (dynamic adaptation)
5. If moderate (50-70%): Review pivot options, run more ablations
6. If poor (<50%): Debug thoroughly before proceeding

---

## Phase 4: Dynamic Adaptation Experiments

**Duration:** 2-3 weeks  
**Goal:** Demonstrate recovery from KG corruption—a novel capability no baseline system has

### Why This Phase Matters

This is your unique contribution that no existing KGQA system can do. ToG, RoG, GNN-RAG are all static—if the KG changes, they degrade and cannot recover without full retraining. Your system's conflict deletion mechanism enables online adaptation. This phase proves that capability experimentally. Even if your static benchmark performance (Phase 3) is only moderate, strong dynamic adaptation results make for a compelling thesis.

**Benchmark:** MetaQA 1-hop with corruption at question 100  
**Baseline:** Frozen system (no online learning) degrades and doesn't recover  
**Target:** Recover to 90% of pre-corruption performance within 50 questions

### Research Questions to Answer

- [ ] **How much corruption can the system tolerate?**
  - [ ] 10% edge corruption? 20%? 50%?
  - [ ] At what point does recovery fail?

- [ ] **What type of corruption is hardest to recover from?**
  - [ ] Delete edges (remove true triples)
  - [ ] Add edges (insert false triples)
  - [ ] Mixed (delete + add)

- [ ] **How does conflict deletion enable recovery?**
  - [ ] When does conflict deletion trigger?
  - [ ] How many conflicts are detected after corruption?
  - [ ] Does memory get cleaned up over time?

### Implementation Tasks

#### Subtask 4.1: Design Corruption Protocol
- [ ] **Choose corruption strategy**
  - [ ] **Start simple:** 10% edge deletion only
  - [ ] Randomly select 10% of KG edges and delete them
  - [ ] Document exact procedure for reproducibility
  - [ ] Later: Try 20%, add-only, mixed (if time permits)

- [ ] **Implement corruption function**
  - [ ] `corrupt_kg(kg, corruption_rate=0.1, corruption_type='delete')`
  - [ ] For delete: randomly remove edges
  - [ ] For add: randomly add plausible-but-wrong edges (same relation type, different tail)
  - [ ] Save corrupted KG to file for reproducibility
  - [ ] **Deliverable:** `kg_corruption.py`

#### Subtask 4.2: Run Dynamic Adaptation Experiment
- [ ] **Experimental setup**
  - [ ] Train on MetaQA 1-hop questions 1-100 (clean KG)
  - [ ] At question 100: corrupt KG (10% deletion)
  - [ ] Continue training on questions 101-200 (corrupted KG)
  - [ ] Evaluate on val set every 10 questions

- [ ] **Full system (with adaptation)**
  - [ ] Fast + Slow + LLM with conflict deletion enabled
  - [ ] Track: Hits@1 over all 200 questions
  - [ ] Track: Memory conflicts detected per question
  - [ ] Track: Memory size over time

- [ ] **Baseline (no adaptation)**
  - [ ] Train on questions 1-100 (clean KG)
  - [ ] At question 100: corrupt KG
  - [ ] Freeze system (no MLP updates, no memory updates)
  - [ ] Evaluate on questions 101-200
  - [ ] Should degrade and not recover

#### Subtask 4.3: Measure Recovery Metrics
- [ ] **Define recovery metrics**
  - [ ] **Pre-corruption performance:** Hits@1 on questions 90-100 (before corruption)
  - [ ] **Peak degradation:** Lowest Hits@1 after corruption (likely questions 101-110)
  - [ ] **Recovery point:** First question where Hits@1 returns to 90% of pre-corruption level
  - [ ] **Questions to recovery:** Recovery point - 100 (target: <50 questions)
  - [ ] **Final performance:** Hits@1 on questions 190-200 (after full recovery)

- [ ] **Collect metrics**
  - [ ] Run full system 3 times with different random seeds
  - [ ] Compute mean ± std for all metrics
  - [ ] Compare to frozen baseline (should not recover)

#### Subtask 4.4: Visualize Adaptation
- [ ] **Create plots**
  - [ ] **Plot 1: Hits@1 over 200 questions**
    - [ ] X-axis: question number, Y-axis: Hits@1
    - [ ] Vertical line at question 100 (corruption point)
    - [ ] Two lines: full system (should dip then recover), frozen baseline (should dip and stay low)
    - [ ] This is your key result plot!
  
  - [ ] **Plot 2: Memory conflicts over time**
    - [ ] X-axis: question number, Y-axis: conflicts detected
    - [ ] Should spike at question 100, then decrease as memory is cleaned up
  
  - [ ] **Plot 3: Memory size over time**
    - [ ] X-axis: question number, Y-axis: number of transitions in memory
    - [ ] Shows memory growth and cleanup

#### Subtask 4.5: Ablation - Vary Corruption Severity (Optional)
**Only do this if Phase 4.1-4.4 work well and you have time:**

- [ ] **5% corruption (mild)**
  - [ ] Should recover very quickly (<20 questions)

- [ ] **20% corruption (severe)**
  - [ ] Harder to recover, may take 50-100 questions

- [ ] **50% corruption (extreme)**
  - [ ] May not fully recover, but should show partial recovery

- [ ] **Compare recovery speed**
  - [ ] Plot: X-axis = corruption rate, Y-axis = questions to recovery
  - [ ] Shows robustness to different corruption levels

### Deliverables

- [ ] **`kg_corruption.py`** - KG corruption utilities
- [ ] **`dynamic_adaptation_experiment.py`** - Script to run experiments
- [ ] **`phase4_results.ipynb`** - Plots and analysis
- [ ] **`phase4_report.md`** - Findings and analysis

### Success Criteria

✅ System recovers to 90% of pre-corruption performance within 50 questions  
✅ Frozen baseline degrades and does not recover (proves adaptation is necessary)  
✅ Conflict deletion triggers after corruption (memory is being cleaned up)  
✅ Recovery is consistent across multiple runs (not just luck)

### 🚦 DECISION GATE 4

**Go:** Clear recovery demonstrated → Proceed to Phase 5 (MetaQA 2-hop)  
**Pivot:** Partial recovery (70-80%) → Still proceed, but lower expectations  
**No-Go:** No recovery (<70%) → Debug conflict deletion, may need to adjust mechanism

**If No-Go:**
- Check conflict deletion logic: Is it triggering?
- Check memory: Are corrupted transitions being removed?
- Check MLP: Is it adapting to new patterns?
- Try higher corruption (20%) to make effect more obvious
- Try different corruption type (add instead of delete)

### Resumption Guide (If Returning After Break)

**Where you left off:** You've tested dynamic adaptation on MetaQA 1-hop.

**To resume:**
1. Check `phase4_report.md` - did system recover?
2. Review key plot: Hits@1 over 200 questions (should show dip and recovery)
3. Review recovery metrics: questions to recovery, final performance
4. If successful: Proceed to Phase 5 (MetaQA 2-hop)
5. If issues: Review decision gate and debug

---

## Phase 5: MetaQA 2-hop Scaling

**Duration:** 2-3 weeks  
**Goal:** Show system scales to multi-hop reasoning (2-hop questions)

### Why This Phase Matters

MetaQA 1-hop is relatively easy (answer is 1 hop from question entity). MetaQA 2-hop requires reasoning over 2 hops, which is significantly harder. Showing that your system scales to 2-hop validates that it's not just memorizing 1-hop patterns but can handle deeper reasoning. Target performance is lower than 1-hop (>60% Hits@1 vs >70% for 1-hop), but demonstrating scaling is important for thesis completeness.

**Benchmark:** MetaQA 2-hop (standard train/val/test split)  
**Baselines:** ToG (~90% Hits@1), RoG (~95% Hits@1)  
**Target:** >60% Hits@1 (competitive, given different setup)

### Implementation Tasks

#### Subtask 5.1: Adapt System for 2-hop
- [ ] **Load MetaQA 2-hop questions**
  - [ ] Same format as 1-hop, but answer is 2 hops away
  - [ ] Example: "Who directed the movies starring [actor]?" → actor --starred_in--> movie --directed_by--> director

- [ ] **Increase max steps**
  - [ ] 1-hop: max 10 steps
  - [ ] 2-hop: max 20 steps (need more steps to find 2-hop paths)

- [ ] **Adjust hyperparameters (if needed)**
  - [ ] May need larger B (branches) or D (depth) for deeper search
  - [ ] Start with same hyperparameters as 1-hop, adjust if performance is poor

#### Subtask 5.2: Run MetaQA 2-hop Experiments
- [ ] **Use best configuration from Phase 3**
  - [ ] Best embeddings (likely TransE)
  - [ ] Best pruning strategy (likely root-only)
  - [ ] Best hyperparameters (B, D, alpha, k)

- [ ] **Run minimal ablations**
  - [ ] Random baseline
  - [ ] Slow only (no MLP)
  - [ ] Fast + Slow (no LLM)
  - [ ] Fast + Slow + LLM (full system)

- [ ] **Measure performance**
  - [ ] Hits@1, Hits@3, F1
  - [ ] Average steps per question
  - [ ] LLM calls per question

#### Subtask 5.3: Compare 1-hop vs 2-hop
- [ ] **Performance degradation**
  - [ ] Table: Hits@1 for 1-hop vs 2-hop across all ablations
  - [ ] Expected: 2-hop is 10-20% lower than 1-hop
  - [ ] Which component matters most for 2-hop? (likely slow mechanism)

- [ ] **Efficiency analysis**
  - [ ] Steps per question: 2-hop should take ~2x more steps
  - [ ] LLM calls: Should be similar (1 call per question if root-only)

#### Subtask 5.4: Dynamic Adaptation on 2-hop (Optional)
**Only if time permits:**

- [ ] **Run same corruption experiment as Phase 4**
  - [ ] Train on questions 1-100, corrupt at 100, continue to 200
  - [ ] Measure recovery

- [ ] **Compare 1-hop vs 2-hop recovery**
  - [ ] Is recovery harder on 2-hop? (likely yes, longer paths)

### Deliverables

- [ ] **`metaqa_2hop_experiment.py`** - Script to run 2-hop experiments
- [ ] **`phase5_results.ipynb`** - Plots and analysis
- [ ] **`phase5_report.md`** - Findings and comparison to 1-hop

### Success Criteria

✅ Achieve >60% Hits@1 on MetaQA 2-hop  
✅ System scales from 1-hop to 2-hop (performance degrades but not catastrophically)  
✅ Fast+Slow+LLM outperforms ablations on 2-hop  
✅ Understand which components are most important for multi-hop reasoning

### 🚦 DECISION GATE 5

**Go:** Achieved >60% Hits@1 on 2-hop → Proceed to Phase 6 (growth & refinement)  
**Pivot:** Achieved 40-60% Hits@1 → Moderate results, but sufficient for thesis  
**No-Go:** <40% Hits@1 → System doesn't scale well, may need architectural changes

**If Pivot or No-Go:**
- Try increasing B/D (more branches, deeper search)
- Try different LLM pruning strategy (adaptive instead of root-only)
- Try pre-training on random 2-hop paths
- Consider focusing thesis on 1-hop + dynamic adaptation (still a solid contribution)

### Resumption Guide (If Returning After Break)

**Where you left off:** You've tested system on MetaQA 2-hop.

**To resume:**
1. Check `phase5_report.md` - what was your Hits@1 on 2-hop?
2. Compare to 1-hop results - how much did performance degrade?
3. Review ablations - which components matter most for 2-hop?
4. If successful (>60%): Proceed to Phase 6 (optional refinements)
5. If moderate (40-60%): Decide if additional experiments are needed
6. If poor (<40%): Review decision gate options

---

## Phase 6: Growth & Refinement (Flexible)

**Duration:** Flexible (2-4 weeks)  
**Goal:** Add experiments as needed based on Phase 3-5 results

### Why This Phase Matters

This is a flexible phase where you add experiments based on what you've learned. If Phases 3-5 went smoothly and you have time, you can add more ablations, try different configurations, or explore extensions. If Phases 3-5 had issues, you can use this time to debug and improve. This phase grows organically based on your results and timeline.

### Possible Tasks (Pick Based on Needs)

#### Option A: More Ablations (If Results Are Strong)
- [ ] **Hyperparameter sweep**
  - [ ] Vary B (branches): [10, 50, 100, 200]
  - [ ] Vary D (depth): [5, 10, 20]
  - [ ] Find optimal configuration

- [ ] **Entity embedding comparison**
  - [ ] TransE vs ComplEx vs BERT
  - [ ] Which works best for MetaQA?

- [ ] **LLM model comparison**
  - [ ] GPT-4 vs GPT-3.5 vs local Llama
  - [ ] Cost-performance tradeoff

#### Option B: More Dynamic Experiments (If Adaptation Works Well)
- [ ] **Vary corruption severity**
  - [ ] 5%, 10%, 20%, 50%
  - [ ] Plot: recovery speed vs corruption rate

- [ ] **Vary corruption type**
  - [ ] Delete-only, add-only, mixed
  - [ ] Which is hardest to recover from?

- [ ] **Targeted corruption**
  - [ ] Corrupt only answer-supporting edges (harder)
  - [ ] Corrupt random edges (easier)

#### Option C: MetaQA 3-hop (If 2-hop Is Strong)
- [ ] **Extend to 3-hop**
  - [ ] Load MetaQA 3-hop questions
  - [ ] Increase max steps to 30
  - [ ] Run best configuration
  - [ ] Target: >40% Hits@1 (3-hop is very hard)

#### Option D: Debugging & Improvement (If Results Are Weak)
- [ ] **Error analysis**
  - [ ] Sample 50 failed questions
  - [ ] Categorize failure modes
  - [ ] Identify patterns

- [ ] **Targeted fixes**
  - [ ] If MLP not learning: Try different embeddings or architecture
  - [ ] If LLM not helping: Try different prompts
  - [ ] If memory gaps: Try pre-training on random exploration

- [ ] **Iterative improvement**
  - [ ] Make changes, re-run experiments
  - [ ] Track improvement over iterations

### Deliverables

- [ ] **`phase6_experiments.ipynb`** - Additional experiments
- [ ] **`phase6_report.md`** - Findings and final recommendations

### Success Criteria

✅ Address any gaps identified in Phases 3-5  
✅ Have comprehensive experimental results for thesis  
✅ Understand system strengths and limitations  
✅ Ready to write thesis

### Resumption Guide (If Returning After Break)

**Where you left off:** You're adding optional experiments.

**To resume:**
1. Review Phases 3-5 reports - what gaps need filling?
2. Check timeline - how much time do you have?
3. Prioritize: What experiments are most important for thesis?
4. Run selected experiments
5. Proceed to Phase 7 (write-up) when satisfied

---

## Phase 7: Write-up & Analysis

**Duration:** 3-5 weeks  
**Goal:** Write complete master's thesis document

### Why This Phase Matters

This is where you synthesize all your experimental results into a coherent thesis narrative. The thesis should clearly explain: (1) the problem (static KGQA systems can't adapt), (2) your solution (Fast & Slow + LLM for KG), (3) your results (grid improvement + MetaQA performance + dynamic adaptation), and (4) your contributions (LLM-enhanced F&S + KG adaptation + novel capability). Good writing is critical—even strong results can be undermined by poor presentation.

### Writing Tasks

#### Subtask 7.1: Thesis Structure
- [ ] **Introduction (5-7 pages)**
  - [ ] Motivation: Why KG reasoning matters (applications, challenges)
  - [ ] Problem: Current KGQA systems are static, can't adapt to KG changes
  - [ ] Proposed solution: Fast & Slow + LLM pruning for KG
  - [ ] Contributions:
    1. LLM-enhanced Fast & Slow improves grid results (92% → X%)
    2. Fast & Slow adapted to KG reasoning (competitive on MetaQA)
    3. Dynamic adaptation capability (novel, no baselines can do this)
  - [ ] Thesis outline

- [ ] **Related Work (8-10 pages)**
  - [ ] **KG reasoning systems:** ToG, RoG, GNN-RAG, SubgraphRAG
    - [ ] How they work, their strengths/weaknesses
    - [ ] Key limitation: static, no adaptation
  - [ ] **RL for KG:** RL-based path finding (MINERVA, MultiHopKG)
    - [ ] How they differ from your approach
  - [ ] **Fast & Slow learning:** Original paper, neuroscience motivation
    - [ ] Complementary learning systems theory
    - [ ] Dual-system architecture
  - [ ] **LLMs for KG:** How LLMs are used in KG tasks
    - [ ] LLM as reasoner, LLM as retriever, LLM + KG hybrid
  - [ ] **Positioning:** Your work combines RL + KG + LLM + online adaptation (unique combination)

- [ ] **Background (5-7 pages)**
  - [ ] **Knowledge Graphs:** Definitions, structure, examples (Freebase, Wikidata)
  - [ ] **KGQA task:** Formal definition, metrics (Hits@1, F1)
  - [ ] **Fast & Slow algorithm:** Detailed explanation
    - [ ] Fast mechanism: MLP with hippocampal replay
    - [ ] Slow mechanism: Memory + minion search
    - [ ] UCT action selection
    - [ ] Conflict deletion
  - [ ] **Benchmarks:** Grid world, MetaQA (statistics, examples)

- [ ] **Method (10-12 pages)**
  - [ ] **Architecture overview:** Diagram showing Fast + Slow + LLM components
  - [ ] **Fast mechanism for KG:**
    - [ ] Entity embeddings (TransE, BERT, etc.)
    - [ ] MLP architecture
    - [ ] Hippocampal replay adapted for KG
  - [ ] **Slow mechanism for KG:**
    - [ ] KG memory structure
    - [ ] Minion search on KG
    - [ ] UCT action selection for relations
  - [ ] **LLM semantic pruning:**
    - [ ] Prompt design (examples for grid and KG)
    - [ ] Pruning strategies (root-only, adaptive)
    - [ ] Caching for efficiency
  - [ ] **Conflict deletion for KG:**
    - [ ] How conflicts are detected
    - [ ] How memory is updated
    - [ ] Enables dynamic adaptation
  - [ ] **Training protocol:**
    - [ ] Episodic learning on KGQA questions
    - [ ] Online updates (MLP + memory)
    - [ ] Hyperparameters (B, D, alpha, k)

- [ ] **Experiments (8-10 pages)**
  - [ ] **Datasets:**
    - [ ] Grid world: 10×10 dynamic grid, 100 episodes
    - [ ] MetaQA: Movie KG, 1-hop and 2-hop questions, train/val/test splits
  - [ ] **Baselines:**
    - [ ] Grid: PPO (54%), A2C (24%), DQN (4.9%), Fast & Slow (92%)
    - [ ] MetaQA: ToG (~97% 1-hop), RoG (~98% 1-hop)
    - [ ] Note: Different settings (online vs one-shot), rough comparison only
  - [ ] **Evaluation metrics:**
    - [ ] Solve rate, steps per episode (grid)
    - [ ] Hits@1, Hits@3, F1 (MetaQA)
    - [ ] LLM calls per question (efficiency)
    - [ ] Recovery metrics (dynamic adaptation)
  - [ ] **Implementation details:**
    - [ ] PyTorch for MLP, NetworkX for KG, OpenAI API for LLM
    - [ ] Hyperparameters: B=100, D=20, alpha=1.0, k=3
    - [ ] Entity embeddings: TransE (128-dim)
    - [ ] Hardware: [your machine specs]

- [ ] **Results (12-15 pages)**
  - [ ] **Grid experiments:**
    - [ ] Table: Solve rate for all ablations
    - [ ] Did LLM improve over 92%? By how much?
    - [ ] Plot: Solve rate over episodes
  - [ ] **MetaQA 1-hop:**
    - [ ] Table: Hits@1 for all ablations
    - [ ] Compare to baselines (ToG, RoG)
    - [ ] Plot: Hits@1 over training questions (learning curve)
    - [ ] Component contribution: MLP vs LLM vs both
  - [ ] **MetaQA 2-hop:**
    - [ ] Table: Hits@1 for all ablations
    - [ ] Compare to 1-hop (performance degradation)
    - [ ] Which components matter most for 2-hop?
  - [ ] **Dynamic adaptation:**
    - [ ] Plot: Hits@1 over 200 questions (key result!)
    - [ ] Show dip at corruption, then recovery
    - [ ] Compare to frozen baseline (no recovery)
    - [ ] Table: Recovery metrics (questions to recovery, final performance)
    - [ ] Vary corruption severity (if done in Phase 6)
  - [ ] **Ablation studies:**
    - [ ] Quantify contribution of each component
    - [ ] Hyperparameter sensitivity (if done in Phase 6)
    - [ ] LLM model comparison (if done in Phase 6)
  - [ ] **Computational cost:**
    - [ ] LLM calls per question
    - [ ] Wall-clock time per question
    - [ ] Cost-performance tradeoff

- [ ] **Discussion (5-7 pages)**
  - [ ] **Key findings:**
    - [ ] LLM pruning improves Fast & Slow on grids
    - [ ] Fast & Slow works on KG reasoning (competitive on MetaQA)
    - [ ] Dynamic adaptation is possible (novel capability)
  - [ ] **Comparison to baselines:**
    - [ ] When does our system excel? (dynamic scenarios, online learning)
    - [ ] When do baselines excel? (one-shot, static KG)
    - [ ] Why the performance gap? (different settings, online vs one-shot)
  - [ ] **Component analysis:**
    - [ ] Is MLP necessary? (depends on results)
    - [ ] Is LLM necessary? (likely yes for high branching factor)
    - [ ] Is slow mechanism necessary? (yes for planning)
  - [ ] **Limitations:**
    - [ ] Performance gap vs ToG/RoG on static benchmarks
    - [ ] Computational cost (LLM calls)
    - [ ] Requires online learning (can't do one-shot)
    - [ ] Tested only on MetaQA (movie domain)
  - [ ] **Broader implications:**
    - [ ] Real-world KGs change constantly (Wikidata 100k edits/day)
    - [ ] Static systems will degrade over time
    - [ ] Online adaptation is necessary for production systems
    - [ ] Dual-system architecture is general (works on grids and KGs)

- [ ] **Conclusion (2-3 pages)**
  - [ ] Summary of contributions
  - [ ] Main takeaways:
    - [ ] LLM pruning enhances Fast & Slow
    - [ ] Dual-system architecture generalizes to KG
    - [ ] Dynamic adaptation is achievable and necessary
  - [ ] Future work:
    - [ ] Scale to larger KGs (WebQSP, Freebase)
    - [ ] Test on other domains (not just movies)
    - [ ] Improve efficiency (reduce LLM calls)
    - [ ] Combine with other KG reasoning techniques (GNNs, etc.)
    - [ ] Deploy in production system with real KG updates

#### Subtask 7.2: Create Figures & Tables
- [ ] **Architecture diagram**
  - [ ] Visual representation of Fast + Slow + LLM components
  - [ ] Data flow: entity embeddings → MLP → UCT → action → minion search → LLM pruning
  - [ ] Use draw.io, PowerPoint, or TikZ (LaTeX)

- [ ] **Performance plots**
  - [ ] Grid: Solve rate over episodes (with/without LLM)
  - [ ] MetaQA 1-hop: Hits@1 over training questions
  - [ ] MetaQA 2-hop: Hits@1 over training questions
  - [ ] Dynamic adaptation: Hits@1 over 200 questions (KEY PLOT!)
  - [ ] All plots: Clean, publication-quality, with error bars (if multiple runs)

- [ ] **Comparison tables**
  - [ ] Grid: Solve rate for all methods (PPO, A2C, DQN, F&S, F&S+LLM)
  - [ ] MetaQA 1-hop: Hits@1 for all ablations + baselines (ToG, RoG)
  - [ ] MetaQA 2-hop: Hits@1 for all ablations + baselines
  - [ ] Dynamic adaptation: Recovery metrics (questions to recovery, final performance)

- [ ] **Ablation tables**
  - [ ] Component contribution: quantify improvement from MLP, LLM, slow mechanism
  - [ ] Hyperparameter sensitivity (if done)
  - [ ] LLM model comparison (if done)

- [ ] **Qualitative examples**
  - [ ] Sample KG traversal paths (show agent navigating from question entity to answer)
  - [ ] Example LLM pruning decisions (show prompt and response)
  - [ ] Error cases (why did these questions fail?)

#### Subtask 7.3: Final Experiments & Polishing
- [ ] **Run final experiments with best configurations**
  - [ ] Use optimal hyperparameters from Phase 6
  - [ ] Run 3-5 times with different random seeds
  - [ ] Compute mean ± std for all metrics
  - [ ] Ensures results are reproducible and statistically significant

- [ ] **Fill any experimental gaps**
  - [ ] Missing ablations identified during writing
  - [ ] Additional baselines if needed (e.g., random baseline)
  - [ ] Statistical significance tests (t-tests comparing ablations)

- [ ] **Code cleanup**
  - [ ] Refactor for readability
  - [ ] Add docstrings and comments
  - [ ] Create README with instructions to reproduce experiments
  - [ ] Organize code into modules: `fast_mechanism.py`, `slow_mechanism.py`, `llm_pruning.py`, etc.
  - [ ] Create scripts to run all experiments: `run_grid_experiments.sh`, `run_metaqa_experiments.sh`

#### Subtask 7.4: Review & Revision
- [ ] **Internal review**
  - [ ] Read through full draft (print it out, read on paper)
  - [ ] Check for logical flow (does each section lead to the next?)
  - [ ] Verify all claims have evidence (every claim should cite a figure/table)
  - [ ] Ensure figures/tables are referenced in text (don't include orphaned figures)
  - [ ] Check for consistency (same terminology throughout)

- [ ] **Advisor feedback**
  - [ ] Submit draft to advisor (give them 1-2 weeks to review)
  - [ ] Schedule meeting to discuss feedback
  - [ ] Incorporate feedback (may require additional experiments)
  - [ ] Iterate as needed (2-3 rounds of feedback is normal)

- [ ] **Peer review (optional but recommended)**
  - [ ] Share with lab mates or peers
  - [ ] Get fresh perspective on clarity
  - [ ] Identify confusing sections

- [ ] **Final polish**
  - [ ] Proofread for grammar/typos (use Grammarly or similar)
  - [ ] Format according to thesis guidelines (margins, fonts, spacing)
  - [ ] Check citations (use BibTeX, ensure all references are complete)
  - [ ] Generate table of contents, list of figures, list of tables
  - [ ] Add acknowledgments, abstract

### Deliverables

- [ ] **`thesis.pdf`** - Complete master's thesis document
- [ ] **`thesis_latex/`** - LaTeX source files (if using LaTeX)
- [ ] **`figures/`** - All publication-quality figures (PDF or PNG)
- [ ] **`tables/`** - All tables (CSV or LaTeX)
- [ ] **`final_results/`** - All experimental results (pickled or CSV)
- [ ] **`README.md`** - Instructions to reproduce all experiments
- [ ] **`requirements.txt`** - Final dependencies

### Success Criteria

✅ Thesis clearly explains problem, method, and contributions  
✅ All experimental claims are backed by data (figures/tables)  
✅ Figures and tables are publication-quality  
✅ Code is clean and reproducible  
✅ Advisor approves for submission

### Resumption Guide (If Returning After Break)

**Where you left off:** You're writing the thesis.

**To resume:**
1. Check which sections are complete (Introduction? Results?)
2. Review advisor feedback (if any)
3. Continue writing from where you left off
4. Use outline above to guide structure
5. Submit to advisor when draft is complete

---

## Phase 8: Extensions (Optional)

**Duration:** Flexible (anytime after thesis submission)  
**Goal:** Extend work for publication or further research

### Why This Phase Matters

This phase is entirely optional and can be done after thesis submission. If you want to publish your work at a conference (AKBC, ICLR workshop, etc.) or continue research, these extensions strengthen your contribution. You can also use this phase to explore ideas that didn't fit in the thesis timeline.

### Possible Extensions

#### Option A: Scale to WebQSP
- [ ] **Adapt system for Freebase**
  - [ ] WebQSP uses Freebase (much larger than MetaQA)
  - [ ] Implement lazy loading (can't load full Freebase in memory)
  - [ ] Train entity embeddings on Freebase subset

- [ ] **Run WebQSP experiments**
  - [ ] Best configuration from MetaQA
  - [ ] Minimal ablations (time-consuming on large KG)
  - [ ] Compare to ToG (~67% Hits@1), RoG (~71% Hits@1)

- [ ] **Dynamic adaptation on WebQSP**
  - [ ] Same corruption protocol as MetaQA
  - [ ] Does adaptation still work on larger KG?

#### Option B: MetaQA 3-hop
- [ ] **Extend to 3-hop**
  - [ ] Load MetaQA 3-hop questions
  - [ ] Increase max steps to 30
  - [ ] Run best configuration
  - [ ] Target: >40% Hits@1 (3-hop is very hard)

- [ ] **Compare 1/2/3-hop**
  - [ ] How does performance degrade with hop count?
  - [ ] Which components are most important for deep reasoning?

#### Option C: Additional Ablations
- [ ] **Comprehensive hyperparameter sweep**
  - [ ] Grid search over B, D, alpha, k
  - [ ] Find globally optimal configuration

- [ ] **Entity embedding comparison**
  - [ ] TransE vs ComplEx vs RotatE vs BERT
  - [ ] Which is best for different KG types?

- [ ] **LLM model comparison**
  - [ ] GPT-4 vs GPT-3.5 vs Claude vs local Llama
  - [ ] Cost-performance tradeoff

- [ ] **Training protocol comparison**
  - [ ] Pure online vs pre-training + finetuning
  - [ ] Which is better?

#### Option D: Real-World Deployment
- [ ] **Deploy on live KG**
  - [ ] Connect to Wikidata API
  - [ ] Run system on real-time KG updates
  - [ ] Measure adaptation in production setting

- [ ] **User study**
  - [ ] Compare system to baselines with human evaluation
  - [ ] Which system produces better answers?

#### Option E: Publication Preparation
- [ ] **Write conference paper**
  - [ ] 8-page paper for AKBC, ICLR workshop, etc.
  - [ ] Focus on key contributions (dynamic adaptation)
  - [ ] Condense thesis into paper format

- [ ] **Create poster/presentation**
  - [ ] For conference or lab presentation
  - [ ] Highlight key results visually

### Deliverables

- [ ] **`webqsp_results.ipynb`** - WebQSP experiments (if done)
- [ ] **`metaqa_3hop_results.ipynb`** - 3-hop experiments (if done)
- [ ] **`conference_paper.pdf`** - Publication draft (if done)
- [ ] **`poster.pdf`** - Conference poster (if done)

### Success Criteria

✅ Extensions strengthen thesis contributions  
✅ Results are publication-quality (if aiming for publication)  
✅ Code is clean and reproducible

---

## Risk Management

### High-Priority Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Can't reproduce original 92% on grids** | Low | High | Debug thoroughly in Phase 0.5 before proceeding; compare line-by-line with original code |
| **LLM pruning doesn't help on grids** | Medium | High | Try different prompts; if still fails, pivot to "Fast & Slow for KG without LLM" |
| **MLP doesn't converge on KG entities** | Medium | High | Test early in Phase 2; try different embeddings (TransE, BERT); if fails, remove MLP and rely on LLM + slow mechanism |
| **Can't achieve >70% on MetaQA 1-hop** | Medium | Medium | Lower target to >60%; emphasize dynamic adaptation as main contribution |
| **Dynamic adaptation doesn't work** | Low | High | Test early in Phase 2 (toy KG); if fails, debug conflict deletion; if still fails, pivot to emphasize LLM pruning contribution |
| **Timeline slips due to sporadic work** | High | Low | Plan is modular with resumption guides; each phase is self-contained; can pause/resume anytime |

### Medium-Priority Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **LLM API costs too high** | Medium | Medium | Use caching aggressively; switch to local model (Llama); reduce LLM calls (root-only pruning) |
| **Entity embeddings don't capture semantics** | Medium | Medium | Test multiple embedding types (TransE, BERT, etc.); ablate in Phase 6 |
| **Hyperparameters don't transfer across datasets** | Medium | Low | Do separate sweep for each dataset; document sensitivity |
| **Advisor wants more experiments** | High | Medium | Phase 6 is flexible for additional experiments; budget extra time for revisions |
| **Writing takes longer than expected** | High | Medium | Start writing early (draft intro/related work during Phase 3-5); budget 5 weeks instead of 3 |

---

## Success Metrics Summary

### Must-Have (Core Contributions)
1. **Reproduce original grid results:** 90-95% solve rate (validates implementation)
2. **LLM improves grids:** 92% → 95%+ OR 20% fewer steps (Contribution 1)
3. **Competitive on MetaQA 1-hop:** >70% Hits@1 (Contribution 2)
4. **Dynamic adaptation works:** Recover to 90% within 50 questions (Contribution 3 - UNIQUE)

### Nice-to-Have (Strengthens Thesis)
5. **Scales to MetaQA 2-hop:** >60% Hits@1 (shows multi-hop reasoning)
6. **Component ablations:** Quantify contribution of MLP, LLM, slow mechanism
7. **Efficient:** Tractable computational cost (<5 min/question)

### Stretch Goals (Publication-Level)
8. **Strong 2-hop results:** >70% Hits@1 on 2-hop
9. **WebQSP validation:** Competitive on standard benchmark
10. **MetaQA 3-hop:** >40% Hits@1 (very challenging)

---

## Timeline Estimates

**With sporadic work (10 hrs/week):**
- Phase 0: 2-3 weeks
- Phase 0.5: 2-3 weeks
- Phase 1: 3-4 weeks
- Phase 2: 3-4 weeks
- Phase 3: 4-5 weeks
- Phase 4: 3-4 weeks
- Phase 5: 3-4 weeks
- Phase 6: 2-4 weeks (flexible)
- Phase 7: 5-6 weeks
- **Total: 27-40 weeks (7-10 months of active work)**

**With focused work (20 hrs/week):**
- Phase 0: 1-2 weeks
- Phase 0.5: 1-2 weeks
- Phase 1: 2-3 weeks
- Phase 2: 2-3 weeks
- Phase 3: 3-4 weeks
- Phase 4: 2-3 weeks
- Phase 5: 2-3 weeks
- Phase 6: 1-2 weeks (flexible)
- Phase 7: 3-4 weeks
- **Total: 17-26 weeks (4-6 months of active work)**

**Note:** These are estimates for active work time. With sporadic work over years, calendar time will be much longer. The modular structure allows pausing between phases.

---

## Next Steps

1. **Review this roadmap** - Does the scope make sense? Any concerns?
2. **Discuss with advisor** - Get buy-in on scope and timeline
3. **Set up environment** - Start Phase 0 (install dependencies, get API keys)
4. **Create progress tracker** - Use GitHub issues or Trello to track tasks
5. **Schedule regular check-ins** - Weekly or bi-weekly progress reviews (even if solo)

---

## Open Questions for Discussion

1. **Advisor expectations:** What does your advisor consider sufficient for a master's thesis? (This plan is designed to be safe and solid, but confirm with them)

2. **Benchmark targets:** Are >70% on MetaQA 1-hop and >60% on 2-hop realistic targets? (Depends on how much the online learning setup hurts performance)

3. **LLM costs:** Do you have budget for OpenAI API calls (~$50-100 for full experiments)? Or should we plan to use local models from the start?

4. **Publication goals:** Do you want to publish at a conference, or is thesis-only sufficient? (Affects scope of Phase 8)

5. **Code reuse:** Should we refactor the original Fast & Slow code or start fresh? (Recommendation: start fresh for cleaner architecture, but reference original heavily)

6. **Contribution framing:** Should we emphasize dynamic adaptation (most novel) or present all three contributions equally? (Current plan: equal emphasis, but can adjust)

7. **Comparison fairness:** How to frame comparison to ToG/RoG given different settings (online vs one-shot)? (Current plan: report their numbers with caveat, emphasize dynamic adaptation as separate contribution)

**Ready to proceed?** Start with Phase 0 (Setup & Orientation) when you're ready!
