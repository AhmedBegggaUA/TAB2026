# Search Algorithms

## Practical Session: UCS and A\*

### Introduction

In the previous session you implemented DFS and BFS by translating the NumPy versions into the Pacman framework. Both algorithms treated all steps as equal — a move North costs the same as a move West. In this session we introduce two algorithms that take **cost into account**: **Uniform Cost Search (UCS)** and **A\* Search**.

You will follow the same translation process as before: understand the algorithm, see how each piece maps to the Pacman `problem` interface, and implement it inside `search.py`.

---

### Uniform Cost Search

BFS finds the path with the fewest actions. UCS finds the path with the **lowest total cost**, which is more useful when different actions have different costs. Instead of expanding nodes in the order they were added (FIFO), UCS always expands the node with the **lowest accumulated cost so far**.

The Pacman framework provides a data structure for this:

**`util.PriorityQueue`** — pops the element with the lowest priority value. Its interface is:

```{code-block} python
pq = util.PriorityQueue()
pq.push(item, priority)   # adds item with the given numeric priority
pq.pop()                  # removes and returns the item with lowest priority
pq.isEmpty()              # returns True if the queue is empty
```

The only change with respect to BFS is what you store in the frontier and how you decide the priority when pushing:

- Each frontier element is now a triple `(state, actions_so_far, cost_so_far)`.
- The priority when you push is `cost_so_far` — the total cost of the path taken to reach that state.

When you expand a node and find a successor with step cost `cost`, you push:

```{code-block} python
pq.push((next_state, actions + [action], cost_so_far + cost), cost_so_far + cost)
```

The visited logic stays the same as in BFS: use a `set`, and only expand a state once.

One important subtlety: **mark a state as visited when you pop it, not when you push it.** The same state can be pushed multiple times via different paths with different costs. You only want to expand it the first time it is popped — which, because of the priority queue, is guaranteed to be via the cheapest path.

```{code-block} python
while not frontier.isEmpty():
    state, actions, cost = frontier.pop()

    if state in visited:
        continue
    visited.add(state)

    if problem.isGoalState(state):
        return actions

    for next_state, action, step_cost in problem.getSuccessors(state):
        if next_state not in visited:
            frontier.push(
                (next_state, actions + [action], cost + step_cost),
                cost + step_cost
            )
```

---

### A\* Search

UCS expands nodes in order of $g(n)$ — the cost already spent. A\* improves on this by also considering $h(n)$ — an **estimate of the remaining cost to the goal**:

$$
f(n) = g(n) + h(n)
$$

Nodes with a lower $f$ are expanded first. If $h(n)$ is **admissible** — it never overestimates the true remaining cost — then A\* is guaranteed to find the optimal solution while typically expanding far fewer nodes than UCS.

The implementation of A\* is **identical to UCS** with one change: the priority when pushing is `cost_so_far + heuristic(next_state, problem)` instead of just `cost_so_far`.

A\* in `search.py` receives the heuristic as a second argument:

```{code-block} python
def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    ...
```

`nullHeuristic` always returns 0, making A\* degrade to UCS. The Manhattan distance heuristic is already implemented in `searchAgents.py` as `manhattanHeuristic`:

$$
h(n) = |x_n - x_{goal}| + |y_n - y_{goal}|
$$

It is admissible because in a grid without walls the Manhattan distance is the exact cost, and walls can only make the true cost larger — never smaller.

To use the heuristic inside your implementation, call it as:

```{code-block} python
heuristic(next_state, problem)
```

So when pushing a successor, the priority becomes:

```{code-block} python
f = cost + step_cost + heuristic(next_state, problem)
frontier.push(
    (next_state, actions + [action], cost + step_cost),
    f
)
```

The rest of the loop — the visited logic, the goal check, the return — is identical to UCS.

---

### Putting it all together: the translation table

The table below summarises how each piece of the algorithm changes across the four methods you have now implemented:

| | DFS | BFS | UCS | A\* |
|---|---|---|---|---|
| Frontier | `util.Stack` | `util.Queue` | `util.PriorityQueue` | `util.PriorityQueue` |
| Element stored | `(state, actions)` | `(state, actions)` | `(state, actions, cost)` | `(state, actions, cost)` |
| Priority | — | — | `cost_so_far` | `cost_so_far + h(state)` |
| Mark visited on | pop | push | pop | pop |
| Optimal? | No | Yes (uniform cost) | Yes | Yes (admissible h) |

---

## Exercise

**Part 1 — UCS.** Open `search.py` and implement `uniformCostSearch` inside its stub. Follow the structure described above. Test it on the following mazes:

```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
python pacman.py -l mediumDottedMaze -p StayEastSearchAgent
python pacman.py -l mediumScaryMaze  -p StayWestSearchAgent
```

`StayEastSearchAgent` and `StayWestSearchAgent` use exponential cost functions that penalise moving in certain directions — UCS should find very different paths from BFS on these layouts. Run the autograder:

```bash
python autograder.py -q q3
```

**Part 2 — A\*.** Implement `aStarSearch` in `search.py`. Remember it receives `heuristic` as a second parameter and the only change from UCS is the priority formula. Test it using the Manhattan distance heuristic:

```bash
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

Run the autograder:

```bash
python autograder.py -q q4
```

**Part 3 — report.** Fill in the table below and answer the questions.

| Algorithm | Maze | Nodes expanded | Path length | Path cost |
|---|---|---|---|---|
| UCS | mediumMaze | | | |
| UCS | mediumDottedMaze | | | |
| UCS | mediumScaryMaze | | | |
| A\* (`nullHeuristic`) | bigMaze | | | |
| A\* (`manhattanHeuristic`) | bigMaze | | | |

- On `mediumMaze`, does UCS find the same path as BFS? Why or why not?
- On `bigMaze`, how many nodes does A\* expand with `manhattanHeuristic` compared to `nullHeuristic`? What does this tell you about the value of a good heuristic?
- Why must `visited` be marked on **pop** in UCS and A\*, but marking on **push** is safe in BFS?