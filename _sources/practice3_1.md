# Practice 3.1: Pacman with AlphaBeta + Neural Networks

## Game Theory and Adversarial Search applied to Pacman

### Introduction

In this practical session we will combine what you learned in **Practice 3** (how to play Pacman with a neural network in *greedy* mode) with **adversarial search** techniques: the **Minimax** algorithm and its optimization **Alpha-Beta Pruning**.

The final goal is to build a hybrid agent that integrates the experience learned by the neural network with a forward-looking reasoning of several moves ahead, evaluating future positions through a score function that mixes classical heuristics with the network's prediction.

---

## Part 1: Minimax Algorithm

### What is Minimax?

The Minimax algorithm is a decision-making algorithm used in **two-player zero-sum games**. The key principle is that one player (the *maximizer*) tries to maximize their score, while the opponent (the *minimizer*) tries to minimize the maximizer's score.

#### Core Concepts

1. **Game Tree**: a tree structure representing all possible game states.
2. **Max Player**: the player trying to maximize the evaluation score.
3. **Min Player**: the player trying to minimize the evaluation score.
4. **Evaluation Function**: a function that assigns a numeric value to game states.
5. **Depth**: how many moves ahead to look in the game tree.

#### The Minimax Principle

```
Max Level:  Choose the action that leads to the HIGHEST value
Min Level:  Choose the action that leads to the LOWEST value
```

The algorithm works by recursively evaluating all possible future states, assuming both players play optimally.

### Simple Example: Tic-Tac-Toe

Consider a simplified tic-tac-toe position where it's X's turn:

```
Current state:
X | O |  
---------
  | X |  
---------
  | O |
```

X (maximizer) has three possible moves. The minimax algorithm will:
1. Try each possible move.
2. Assume O (minimizer) will respond optimally.
3. Choose the move that guarantees the best outcome.

### Mathematical Formulation

For a game state `s`, the minimax value is defined as:

```
minimax(s) = {
    evaluation(s)                           if s is terminal
    max(minimax(child) for child in s)      if s is Max node
    min(minimax(child) for child in s)      if s is Min node
}
```

### Application to Pacman

Now we will apply the Minimax algorithm to the Pacman game, where:
- **Pacman (Agent 0)**: maximizing player - wants to maximize score.
- **Ghosts (Agents 1+)**: minimizing players - want to minimize Pacman's score.

### Game Structure in Pacman

In Pacman Minimax with **2 ghosts**:
- One **ply** = Pacman moves + Ghost 1 moves + Ghost 2 moves.
- **Depth 1** = one complete ply (Pacman + both ghosts).
- **Depth 2** = two complete plies.

### Agent Order

The agents take turns in this specific order:
1. **Pacman (Agent 0)** - MAX player.
2. **Ghost 1 (Agent 1)** - MIN player.
3. **Ghost 2 (Agent 2)** - MIN player.
4. Back to **Pacman (Agent 0)** - depth increments.

This cycle repeats until the maximum depth is reached.

### Detailed Example: Depth 2 with 2 Ghosts

Let's walk through a complete example with **depth = 2** and **2 ghosts**.

#### Initial Setup

```
Game State:
- Pacman at position (1,1)
- Ghost 1 at position (3,1)
- Ghost 2 at position (1,3)
- Food dot at (2,1)
- Evaluation scores for different scenarios:
  * Pacman eats food: +10
  * Ghost catches Pacman: -500
  * Normal move: -1 (time penalty)
```

#### The Game Tree (Depth 2)

```
                    ROOT: Pacman Turn (MAX)
                   /          |          \
               NORTH        EAST        SOUTH
              Score: ?    Score: ?    Score: ?
                |            |            |
        
    DEPTH 0: Ghost1 Turn (MIN)   DEPTH 0: Ghost1 Turn (MIN)   DEPTH 0: Ghost1 Turn (MIN)
         /    |    \                  /    |    \                  /    |    \
      N     E     S                N     E     S                N     E     S  
      |     |     |                |     |     |                |     |     |
      
    DEPTH 0: Ghost2 Turn (MIN)   DEPTH 0: Ghost2 Turn (MIN)   DEPTH 0: Ghost2 Turn (MIN)
         /    |    \                  /    |    \                  /    |    \
      N     E     S                N     E     S                N     E     S  
      |     |     |                |     |     |                |     |     |
      
    DEPTH 1: Pacman Turn (MAX)   DEPTH 1: Pacman Turn (MAX)   DEPTH 1: Pacman Turn (MAX)
    /    |    \                  /    |    \                  /    |    \
   N    E    S                  N    E    S                  N    E    S
   |    |    |                  |    |    |                  |    |    |
   
   [... continues for depth 1 with both ghosts ...]
   
 DEPTH 2: EVALUATE            DEPTH 2: EVALUATE            DEPTH 2: EVALUATE
   +9  -501  +8                +8   +9  -501               -501  +8   +7
```

#### Step-by-Step Execution

**Step 1: Root Node (Pacman's initial turn)**
- Pacman has 3 possible actions: NORTH, EAST, SOUTH.
- We need to evaluate each branch.

**Step 2: Evaluate EAST branch**

2.1. **Pacman moves EAST** → Pacman at (2,1), eats food (+10 points).

2.2. **Ghost 1's turn (MIN)**
- Ghost 1 can move: NORTH, EAST, SOUTH.
- Let's say Ghost 1 moves NORTH → Ghost 1 at (3,0).

2.3. **Ghost 2's turn (MIN)**
- Ghost 2 can move: NORTH, EAST, SOUTH.
- Let's say Ghost 2 moves EAST → Ghost 2 at (2,3).

2.4. **Pacman's turn again (depth 1)**
- Pacman can move: NORTH, EAST, SOUTH.
- Evaluate each possibility with the new ghost positions:
  - NORTH: Pacman at (2,2) → Continue with both ghosts...
  - EAST: Pacman at (3,1) → Continue with both ghosts...
  - SOUTH: Pacman at (2,0) → Continue with both ghosts...

2.5. **Complete evaluation for this path**
- After all ghost responses and final Pacman moves, this path evaluates to **+5**.

2.6. **Back to Ghost 2's choices (step 2.3)**
- Ghost 2 evaluates all its moves:
  - NORTH: leads to final value +3
  - EAST: leads to final value +5  ← we calculated this
  - SOUTH: leads to final value +4
- Ghost 2 chooses MIN: min(+3, +5, +4) = **+3**

2.7. **Back to Ghost 1's choices (step 2.2)**
- Ghost 1 evaluates all its moves (each leading to Ghost 2's turn):
  - NORTH: leads to Ghost 2 min = +3  ← we calculated this
  - EAST: leads to Ghost 2 min = +2
  - SOUTH: leads to Ghost 2 min = +1
- Ghost 1 chooses MIN: min(+3, +2, +1) = **+1**

**Therefore: Pacman EAST → Final value = +1**

**Step 3: Evaluate NORTH branch** 

3.1. **Pacman moves NORTH** → Pacman at (1,2)

3.2. **Ghost 1's turn** → Tries all moves → Best for Ghost 1: +2
3.3. **Ghost 2's turn** → Tries all moves → Best for Ghost 2: +1
3.4. **Continue recursion...** → Final evaluation: **+1**

**Step 4: Evaluate SOUTH branch**

4.1. **Pacman moves SOUTH** → Pacman at (1,0)
4.2. **Ghost 1's turn** → Tries all moves → Best for Ghost 1: -1
4.3. **Ghost 2's turn** → Tries all moves → Best for Ghost 2: -2  
4.4. **Continue recursion...** → Final evaluation: **-2**

**Step 5: Pacman's final decision**
- NORTH = +1
- EAST = +1  ← TIE! (same as NORTH)
- SOUTH = -2

**Pacman chooses NORTH or EAST** (depends on which is evaluated first) because both guarantee +1, which is better than -2.

### Implementation for Pacman

```python
class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for Pacman with multiple ghosts
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction.
        """
        
        def minimax(agentIndex, depth, gameState):
            """
            Recursive minimax function
            
            Args:
            - agentIndex: Current agent (0=Pacman, 1+=Ghosts)  
            - depth: Current depth in the game tree
            - gameState: Current state of the game
            
            Returns:
            - Best evaluation score for this state
            """
            # Base case: terminal state or maximum depth reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman's turn (Maximizer)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)
            # Ghost's turn (Minimizer)  
            else:
                return minValue(agentIndex, depth, gameState)
        
        def maxValue(agentIndex, depth, gameState):
            """
            Handles Pacman's moves (maximizing player)
            """
            v = float('-inf')  # Start with worst possible value
            legalActions = gameState.getLegalActions(agentIndex)
            
            # No legal actions available
            if not legalActions:
                return self.evaluationFunction(gameState)

            # Try each possible action and choose the best
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                # After Pacman moves, first ghost plays (agent 1)
                v = max(v, minimax(1, depth, successor))
            return v

        def minValue(agentIndex, depth, gameState):
            """
            Handles Ghost moves (minimizing players)
            """
            v = float('inf')  # Start with best possible value for Pacman
            legalActions = gameState.getLegalActions(agentIndex)
            
            # No legal actions available
            if not legalActions:
                return self.evaluationFunction(gameState)

            # Determine next agent and depth
            nextAgent = agentIndex + 1
            nextDepth = depth
            
            # If all ghosts have moved, return to Pacman and increment depth
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0      # Back to Pacman
                nextDepth = depth + 1  # New ply begins

            # Try each possible action and choose the worst for Pacman
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, minimax(nextAgent, nextDepth, successor))
            return v

        # Main decision logic for Pacman
        bestAction = None
        bestScore = float('-inf')

        # Try each legal action for Pacman
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # Start minimax with first ghost (agent 1) at current depth
            score = minimax(1, 0, successor)
            
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction
```

### Complete Execution Trace

Let's trace through our depth-2 example step by step:

**Initial Call**: `getAction(gameState)`

**1. Try Pacman action: EAST**
   - `generateSuccessor(0, EAST)` → New state with Pacman at (2,1)
   - Call `minimax(1, 0, successor_state)`

**2. Ghost's turn (agentIndex=1, depth=0)**
   - `minValue(1, 0, state)` called
   - Ghost legal actions: [NORTH, EAST, SOUTH]
   
   **2.1 Try Ghost 1 action: NORTH**
   - `generateSuccessor(1, NORTH)` → Ghost 1 at (3,0)
   - nextAgent = 2 (Ghost 2), nextDepth = 0
   - Call `minimax(2, 0, successor_state)`
   
   **2.1.1 Ghost 2's turn (agentIndex=2, depth=0)**
   - `minValue(2, 0, state)` called
   - Ghost 2 legal actions: [NORTH, EAST, SOUTH]
   
   **2.1.1.1 Try Ghost 2 action: NORTH**
   - `generateSuccessor(2, NORTH)` → Ghost 2 at (1,4)
   - nextAgent = 3, but numAgents = 3, so nextAgent = 0, nextDepth = 1
   - Call `minimax(0, 1, successor_state)`
   
   **2.1.1.1.1 Pacman's turn (agentIndex=0, depth=1)**
   - `maxValue(0, 1, state)` called
   - Pacman legal actions: [NORTH, EAST, SOUTH]
   - Try each action, call minimax for each with deeper depth...
   - Eventually returns max value: +3
   
   **2.1.1.1 Back to Ghost 2 choice NORTH** → value +3
   
   **2.1.1.2 Try Ghost 2 action: EAST** → Eventually value +5
   
   **2.1.1.3 Try Ghost 2 action: SOUTH** → Eventually value +4
   
   **2.1.1 Ghost 2 chooses minimum**: min(+3, +5, +4) = +3
   
   **2.1 Back to Ghost 1 choice NORTH** → value +3

   **2.2 Try Ghost 1 action: EAST** → Eventually value +2
   
   **2.3 Try Ghost 1 action: SOUTH** → Eventually value +1

**3. Ghost 1 chooses minimum**: min(+3, +2, +1) = +1

**4. Pacman EAST final score**: +1

**Repeat for Pacman actions NORTH and SOUTH**

**5. Final decision**: max(NORTH:+1, EAST:+1, SOUTH:-2) = NORTH or EAST (tie)

### Key Implementation Details

#### 1. Agent Turn Management
```python
# After each agent acts, move to next agent
nextAgent = agentIndex + 1

# When all agents have acted, return to Pacman and increment depth  
if nextAgent == gameState.getNumAgents():
    nextAgent = 0      # Back to Pacman
    nextDepth = depth + 1  # Start new ply
```

#### 2. Depth Progression
- **Depth increments** only when control returns to Pacman.
- One **complete ply** = Pacman moves + Ghost 1 moves + Ghost 2 moves.
- **Depth 0**: first round - Pacman, Ghost 1, Ghost 2.
- **Depth 1**: second round - Pacman, Ghost 1, Ghost 2.

#### 3. Agent Turn Cycle
The complete cycle for one ply with 2 ghosts:
```
Agent 0 (Pacman) → Agent 1 (Ghost 1) → Agent 2 (Ghost 2) → Agent 0 (Pacman, depth++)
```

#### 4. Base Cases
```python
# Stop recursion when:
# 1. Game ends (win/lose)
# 2. Maximum depth reached  
# 3. No legal actions available
if gameState.isWin() or gameState.isLose() or depth == self.depth:
    return self.evaluationFunction(gameState)
```

The Minimax algorithm guarantees that Pacman will make the best possible decision, assuming all ghosts also play optimally to minimize Pacman's score. With multiple ghosts, the algorithm creates a deeper, more complex tree, but the same principle applies: Pacman maximizes while each ghost minimizes in sequence.

---

## Part 2: Alpha-Beta Pruning

### Optimizing Minimax with Alpha-Beta Pruning

While the Minimax algorithm guarantees optimal decisions, it has a significant drawback: it explores **every possible branch** of the game tree, which can be computationally expensive. **Alpha-Beta Pruning** is an optimization technique that can significantly reduce the number of nodes evaluated without affecting the final result.

### What is Alpha-Beta Pruning?

Alpha-Beta Pruning works by **eliminating branches** that cannot possibly influence the final decision. It maintains two values:

- **Alpha (α)**: the best value that the **maximizer** (Pacman) has found so far.
- **Beta (β)**: the best value that the **minimizer** (Ghost) has found so far.

### The Pruning Principle

The key insight is:
> If at any point we discover that the current path will not be chosen, we can stop exploring it.

**Specifically:**
- **In MAX nodes**: if the value becomes ≥ β, prune (the minimizer won't allow this path).
- **In MIN nodes**: if the value becomes ≤ α, prune (the maximizer won't choose this path).

### Alpha-Beta Algorithm (pseudocode)

```python
def alphabeta(node, depth, alpha, beta, is_maximizing_player):
    if depth == 0 or node.is_terminal():
        return node.evaluate()
    
    if is_maximizing_player:
        max_eval = float('-inf')
        for child in node.get_children():
            eval_score = alphabeta(child, depth-1, alpha, beta, False)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:  # Prune!
                break
        return max_eval
    else:
        min_eval = float('inf')
        for child in node.get_children():
            eval_score = alphabeta(child, depth-1, alpha, beta, True)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:  # Prune!
                break
        return min_eval
```

### Application to Pacman

The game structure (agents, plies, order) is identical to Minimax. The only change is that we now propagate α and β through the tree and prune when the condition is met.

### Detailed Example: Depth 2 with 2 Ghosts and Pruning

#### Initial Setup (same as Minimax)

```
Game State:
- Pacman at position (1,1)
- Ghost 1 at position (3,1)
- Ghost 2 at position (1,3)
- Food dot at (2,1)
```

#### The Alpha-Beta Game Tree (Depth 2)

```
                    ROOT: Pacman Turn (MAX)
                   α=-∞, β=+∞
                   /          |          \
               NORTH        EAST        SOUTH
              Score: ?    Score: ?    Score: ?
                |            |            |
        
    DEPTH 0: Ghost1 Turn (MIN)   DEPTH 0: Ghost1 Turn (MIN)   DEPTH 0: Ghost1 Turn (MIN)
           α=-∞, β=+∞               α=+1, β=+∞               α=+1, β=+∞
         /    |    \                  /    |    \                  /    ✗    ✗
      N     E     S                N     E     S                N    PRUNED
      |     |     |                |     |     |                |         
      
    DEPTH 0: Ghost2 Turn (MIN)   DEPTH 0: Ghost2 Turn (MIN)   DEPTH 0: Ghost2 Turn (MIN)
           α=-∞, β=+∞               α=+1, β=+∞               α=+1, β=+∞
         /    |    \                  /    |    ✗                |
      N     E     S                N     E   PRUNED            N → returns -1
      |     |     |                |     |                    ↓
      |     |     |                |     |                  PRUNE!
      
    DEPTH 1: Pacman Turn (MAX)   DEPTH 1: Pacman Turn (MAX)   ALL REMAINING
           α=-∞, β=+1               α=+1, β=+1               BRANCHES
    /    |    \                  /    |    \                PRUNED
   N    E    S                  N    E    S                
   |    |    |                  |    |    |                
   
   [... Ghost1 → Ghost2 → Evaluate ...]     
   
 DEPTH 2: EVALUATE            DEPTH 2: EVALUATE      
   +1  +2  +3                   +1   +1  +1          
       ↓                           ↓
   Returns +1                   Returns +1
```

Legend:
- ✗ = pruned branches (not evaluated)
- α, β = alpha-beta values at each node
- Numbers = final evaluation scores

### Step-by-Step Alpha-Beta Execution

#### Initial State
- **Root**: α = -∞, β = +∞
- Pacman tries actions in order: NORTH, EAST, SOUTH.

#### Branch 1: Pacman NORTH

**1.1. Pacman moves NORTH**
- Call `alphabeta(1, 0, -∞, +∞, False)` (Ghost 1's turn)

**1.2. Ghost 1's turn (MIN node)**
- α = -∞, β = +∞
- Ghost 1 tries NORTH first → through complete evaluation → returns +3
- β = min(+∞, +3) = +3
- Ghost 1 tries EAST → returns +2
- β = min(+3, +2) = +2
- Ghost 1 tries SOUTH → returns +1
- β = min(+2, +1) = +1

**1.3. Ghost 1 returns +1**
- Back to root: α = max(-∞, +1) = **+1**

#### Branch 2: Pacman EAST  

**2.1. Pacman moves EAST**
- Call `alphabeta(1, 0, +1, +∞, False)` (Ghost 1's turn)
- **Note**: α is now +1 from the previous branch!

**2.2. Ghost 1's turn (MIN node)**
- α = +1, β = +∞
- Ghost 1 tries NORTH → through Ghost 2 → returns +2
- β = min(+∞, +2) = +2

**2.3. Ghost 1 tries EAST**
- Call `alphabeta(2, 0, +1, +2, False)` (Ghost 2's turn)

**2.4. Ghost 2's turn (MIN node)**
- α = +1, β = +2
- Ghost 2 tries NORTH → through complete depth evaluation → returns +1
- β = min(+2, +1) = +1
- Ghost 2 tries EAST → returns +1
- β = min(+1, +1) = +1
- **Check: β(+1) ≤ α(+1)? YES!**
- **PRUNE! Skip Ghost 2 SOUTH**

**2.5. Ghost 2 returns +1**
- Back to Ghost 1: β = min(+2, +1) = +1
- **Check: β(+1) ≤ α(+1)? YES!**
- **PRUNE! Skip Ghost 1 SOUTH**

**2.6. Ghost 1 returns +1**
- Back to root: α = max(+1, +1) = +1

#### Branch 3: Pacman SOUTH

**3.1. Pacman moves SOUTH**
- Call `alphabeta(1, 0, +1, +∞, False)` (Ghost 1's turn)
- **Note**: α is still +1!

**3.2. Ghost 1's turn (MIN node)**
- α = +1, β = +∞
- Ghost 1 tries NORTH → Ghost 2 turn

**3.3. Ghost 2's turn (MIN node)**
- α = +1, β = +∞
- Ghost 2 tries NORTH → Pacman (depth=1) → eventually returns **-1**
- v = min(+∞, -1) = -1
- β = min(+∞, -1) = -1
- **Check: v(-1) < α(+1)? YES!**
- **PRUNE! Return -1 immediately**

**3.4. Ghost 2 returns -1**
- Back to Ghost 1: β = min(+∞, -1) = -1
- **Check: β(-1) ≤ α(+1)? YES!**
- **PRUNE! Skip Ghost 1 EAST and SOUTH**

**3.5. Ghost 1 returns -1**
- Back to root: α remains +1

#### Final Decision

**Root Node Final State:**
- NORTH: +1
- EAST: +1 ← TIE
- SOUTH: -1 (pruned early)

**Pacman chooses NORTH or EAST** (both give +1)

### Key Alpha-Beta Insights

#### 1. Multiple Minimizer Levels
With 2 ghosts we have two consecutive MIN levels:
- Ghost 1 minimizes over Ghost 2's choices.
- Ghost 2 minimizes over Pacman's future choices.
- Both can trigger pruning independently.

#### 2. Cascading Pruning
When Ghost 2 triggers pruning (β ≤ α), it not only skips Ghost 2's remaining moves but can also cause Ghost 1 to skip its remaining moves.

#### 3. Agent-Specific Pruning Conditions
```python
# In MAX nodes: prune when v ≥ β
if v > beta:
    return v

# In MIN nodes: prune when v ≤ α  
if v < alpha:
    return v
```

#### 4. Order Matters
The order of exploring children affects pruning efficiency. The earlier we find a "promising" move, the more we will prune.

### Comparison: Minimax vs Alpha-Beta

**Minimax**: evaluates **81 nodes** in the tree (3^4 with 2 ghosts).
**Alpha-Beta**: evaluates **43 nodes** in the tree.
**Savings**: 47% fewer evaluations!

### Important Note

**Alpha-Beta pruning returns the exact same result** as regular Minimax - it's just faster! The pruned branches are guaranteed to not affect the optimal decision. With multiple ghosts, the algorithm creates more opportunities for pruning, especially when the first few evaluations establish good alpha and beta bounds.

---

## Part 3: Assignment — Neural Network + Alpha-Beta Integration

In the previous practical session you implemented the `NeuralAgent` that used a neural network trained on game data to play Pacman in a **greedy** way: on every turn, the network predicts a probability distribution over actions and the best one is chosen (with a small percentage of exploration).

Now we will go one step further: we will combine the **experience learned by the network** with a **multi-step adversarial reasoning** through Alpha-Beta.

### Reminder: the evaluation function from Practice 3

In `NeuralAgent.evaluationFunction()` from Practice 3 you only had two basic heuristic factors:

```python
# Factor 1: distance to closest food
if food:
    min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
    score += 1.0 / (min_food_distance + 1)

# Factor 2: ghost proximity
for ghost_state in ghost_states:
    ghost_pos = ghost_state.getPosition()
    ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
    
    if ghost_state.scaredTimer > 0:
        # If ghost is scared, get closer
        score += 50 / (ghost_distance + 1)
    else:
        # If ghost is not scared, avoid it
        if ghost_distance <= 2:
            score -= 200  # Big penalty for being too close
```

### Goal of this practical session

Build a hybrid `AlphaBetaNeuralAgent` whose evaluation function combines:

1. The **traditional score** (with the heuristic factors you'll improve).
2. The **neural network's prediction** (the "experience" learned across many games).

Both components will be weighted with configurable weights, allowing experimentation with different proportions of heuristic vs learned reasoning.

---

## Part 4: Tasks

### Task 1 — Improve the evaluation function with new heuristics

You must add **at least 2 new heuristics** to the `NeuralAgent` evaluation function, in addition to the two factors already provided (distance to food + ghost proximity).

**In the report you must justify the choice of each heuristic** and explain the weight you gave it.

### Task 2 — Train the model with good games

Collect a dataset of games by playing with the improved `NeuralAgent` (or by controlling it manually if you prefer) and train the network. The dataset quality is **critical**: the more winning games you have, the better the network will learn.

> **Optional (recommended for top grade):** you can balance the dataset (e.g., equalize the frequency of each action), or tweak the neural network's hyperparameters (layers, neurons, learning rate, batch size, optimizer, dropout, etc.).

### Task 3 — Implement `AlphaBetaNeuralAgent`

Create a new agent class that inherits from the Alpha-Beta structure but whose evaluation function is the **weighted combination**:

```python
final_score = w_heuristic * traditional_score + w_neural * neural_score
```

Weights `w_heuristic` and `w_neural` must be **configurable parameters** that can be tuned when launching the agent.

For example:

```python
def evaluation_combined(self, state):
    # 1) Traditional score (with the new heuristics from Task 1)
    trad_score = self.traditional_evaluation(state)
    
    # 2) Neural network score
    neural_score = self.neural_evaluation(state)
    
    # 3) Weighted combination
    return self.w_heuristic * trad_score + self.w_neural * neural_score
```

> **Optional (for top grade):** you can make the weights **dynamic**:
> - Different weights depending on Pacman's quadrant.
> - Different weights when there are scared ghosts (the network may not have seen many capsule-active situations, so weight heuristics more).
> - Weights that decay over game time.
>
> Any strategy you reasonably justify is valid.

### Task 4 — Report results in a comparative table

You must run **3 configurations** of the agent on **2 layouts**: the classical (`mediumClassic`) and a new one defined below. The neural network is only trained on the classical, but tested on both.

| Configuration                                       | Classical layout | New layout |
|-----------------------------------------------------|:----------------:|:----------:|
| Greedy neural agent (no modifications)              |   score / win    | score / win |
| Greedy neural agent + 2 new heuristics              |   score / win    | score / win |
| Alpha-Beta + network + heuristics (final)           |   score / win    | score / win |

For each cell, report the average score over at least 10 games and the win rate.

### Task 5 — New layout to test

In addition to the classical `mediumClassic` layout, you must test this modified layout:

```
%%%%%%%%%%%%%%%%%%%%
%o... ........ ....%
%.%%. .%%..%%. .%%.%
%.%..............%.%
%.%.%%.%%  %%.%%.%.%
%......%G  G%......%
%.%.%%.%%%%%%.%%.%.%
%.%..............%.%
%.%%.%.%%..%%.%.%%.%
%....%...P....%...o%
%%%%%%%%%%%%%%%%%%%%

```

Note that it has openings in areas where the classical has walls (more open paths), which changes distances and game dynamics. Create it as a new file inside the `layouts/` folder (e.g., `customMaze.lay`) and launch it with the option `-l customMaze`.

### Task 6 — Record the video of the best game

You must record **one video**: the **final configuration** (Alpha-Beta + network + heuristics) playing its best game. You have to record two clips:

1. The best game on the `mediumClassic` layout.
2. The best game on the new layout.

**Video requirements:**
- Must be **reproducible**: include the exact command used to launch the game (with a fixed seed) and the corresponding trained model. If we cannot reproduce the game by executing the command you report, it will be **penalized**.
- No cuts in the middle of the game.

---

## What you must submit

You must submit a single `.zip` containing the following files, in this order:

1. **Report** in Jupyter Notebook format (preferably with rich markdown) **or** PDF. It doesn't matter which one you choose, but it must be **well structured** following this order:
   
   1. Introduction and goal of the practical session.
   2. **Part 1 — New heuristics:** detailed description of the two (or more) heuristics you added to the evaluation function. Justify the choice and weight of each.
   3. **Part 2 — Network training:** description of the dataset used (how many games, win-rate of training games), final network architecture and hyperparameters. If you adjusted dataset balancing or architecture, describe it here (optional but valued).
   4. **Part 3 — `AlphaBetaNeuralAgent`:** explanation of how the Alpha-Beta + neural network integration works. How `final_score` is computed, what weights you used and why. If you used dynamic weights (optional), explain the strategy.
   5. **Part 4 — Results:** comparative table of the 3 configurations on the 2 layouts (see Task 4). Commentary and analysis of the results: did adding the new heuristics improve performance? Did Alpha-Beta? Does it generalize well to the new layout?
   6. **Conclusions.**
   7. **Exact commands** to reproduce the video (with seeds, layout and model).

2. **Source code** (the files that have changed compared to the previous submission):
   - `multiAgents.py` with your improved `NeuralAgent` and `AlphaBetaNeuralAgent` implementations.
   - `net.py` if you modified the network architecture.
   - `models/pacman_model.pth` with your trained model.
   - Any other relevant auxiliary file (training scripts, new layout, etc.).

3. **Video** of the best game (a single file with both clips, or two separate files if you submit them apart).

> **Critical reminder:** the video must be **reproducible** with the code and model you submit. If we run your reported command and don't get an equivalent result, it will be penalized.

---

## Source Web and GitHub
- [GitHub Repository](https://github.com/AhmedBegggaUA/pacman)
- [Web Page (Berkeley AI)](https://inst.eecs.berkeley.edu/~cs188/fa24/projects/proj2/)