# Search Algorithms

## Practical Session: BFS and DFS

### Introduction

In this session you will work with two search algorithms — **BFS** and **DFS** — that you already know from the previous session. Both were implemented using NumPy adjacency matrices and plain Python lists. Here we will revisit them, understand the key structural difference between the two, and then adapt them to solve maze problems using the data structures provided by the Pacman framework.

---

### The algorithms you already have

Below are the two implementations from the previous session. Read them carefully before continuing — pay special attention to how `to_explore` is managed in each one and how that single difference drives completely different exploration behaviour.

```{code-block} python
import numpy as np
import networkx as nx

def dfs_numpy(graph, source, target):
    nodes = list(graph.nodes())
    adjacency_matrix = nx.to_numpy_array(graph, nodelist=nodes)

    visited = np.zeros(len(nodes), dtype=bool)
    source_index = nodes.index(source)
    target_index = nodes.index(target)
    visited[source_index] = True

    to_explore = [source_index]
    current_position = len(to_explore) - 1  # DFS: always the last element

    parent = np.ones(len(nodes), dtype=int) * -1
    steps = 0

    while len(to_explore) > 0:
        steps += 1
        current_index = to_explore[current_position]
        to_explore.pop(current_position)

        if current_index == target_index:
            break

        neighbors = np.where(adjacency_matrix[current_index] == 1)[0]
        for neighbor_index in neighbors:
            if not visited[neighbor_index]:
                visited[neighbor_index] = True
                to_explore.append(neighbor_index)
                current_position = len(to_explore) - 1
                parent[neighbor_index] = current_index

        if len(to_explore) > 0:
            current_position = len(to_explore) - 1

    path = []
    if parent[target_index] != -1 or source_index == target_index:
        current = target_index
        while current != -1:
            path.insert(0, nodes[current])
            current = parent[current]

    return steps, path


def bfs_numpy(graph, source, target):
    nodes = list(graph.nodes())
    adjacency_matrix = nx.to_numpy_array(graph, nodelist=nodes)

    visited = np.zeros(len(nodes), dtype=bool)
    source_index = nodes.index(source)
    target_index = nodes.index(target)
    visited[source_index] = True

    to_explore = [source_index]
    current_position = 0  # BFS: always the first pending element

    parent = np.ones(len(nodes), dtype=int) * -1
    steps = 0

    while current_position < len(to_explore):
        steps += 1
        current_index = to_explore[current_position]
        current_position += 1

        if current_index == target_index:
            break

        neighbors = np.where(adjacency_matrix[current_index] == 1)[0]
        for neighbor_index in neighbors:
            if not visited[neighbor_index]:
                visited[neighbor_index] = True
                to_explore.append(neighbor_index)
                parent[neighbor_index] = current_index

    shortest_path = []
    if parent[target_index] != -1 or source_index == target_index:
        current = target_index
        while current != -1:
            shortest_path.insert(0, nodes[current])
            current = parent[current]

    return steps, shortest_path
```

The only structural difference between the two is how the next node is chosen from `to_explore`:

- **DFS** always reads and removes the **last** element — this is LIFO behaviour (Last In, First Out), equivalent to a stack. The search goes as deep as possible before backtracking.
- **BFS** reads the element at `current_position` and advances the pointer — this is FIFO behaviour (First In, First Out), equivalent to a queue. The search expands level by level and is guaranteed to find the shortest path when all edges have equal cost.

---

### Stack and Queue in the Pacman framework

The Pacman framework (`util.py`) provides two data structures that formalise exactly this behaviour:

**`util.Stack`** — implements LIFO order. Its interface is:

```{code-block} python
s = util.Stack()
s.push(item)      # adds item on top
s.pop()           # removes and returns the top item
s.isEmpty()       # returns True if the stack is empty
```

**`util.Queue`** — implements FIFO order. Its interface is:

```{code-block} python
q = util.Queue()
q.push(item)      # adds item at the back
q.pop()           # removes and returns the front item
q.isEmpty()       # returns True if the queue is empty
```

Instead of managing `to_explore` as a plain list and manually tracking `current_position`, you can now delegate that logic entirely to these data structures. The rest of the algorithm — the visited set, the parent tracking, the path reconstruction — stays exactly the same.

---

### The Pacman environment

Download and unzip the project:

```bash
wget https://inst.eecs.berkeley.edu/~cs188/fa24/assets/projects/search.zip
unzip search.zip && cd search
```

Before touching any code, run the game to see what you are working with:

```bash
python pacman.py
```

The folder contains several files. The only two you will edit in this session are `search.py` and, optionally, `searchAgents.py`. The rest are part of the framework and you should not modify them. The most relevant ones to be aware of are:

- `search.py` — where you will write your implementations. It already contains empty stubs for `depthFirstSearch` and `breadthFirstSearch`.
- `searchAgents.py` — defines the `SearchAgent` class, which calls your functions and executes the resulting action sequence in the maze.
- `util.py` — provides `Stack`, `Queue`, and `PriorityQueue`. You do not need to read it in detail, just import and use them.
- `pacman.py` — the main entry point. You never edit this.

Open `search.py` and find the stubs. They look like this:

```{code-block} python
def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.
    ...
    """
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """
    Search the shallowest nodes in the search tree first.
    ...
    """
    util.raiseNotDefined()
```

`util.raiseNotDefined()` is a placeholder that throws an error if you run the code without implementing the function. Replace everything inside each stub with your implementation.

---

### From adjacency matrix to the Pacman problem interface

In the Pacman framework the graph is not given to you explicitly. Instead, the `problem` object exposes three methods that replace what you previously did with the adjacency matrix and index lookups:

```{code-block} python
problem.getStartState()      # returns the starting state, e.g. (5, 5)
problem.isGoalState(state)   # returns True if state is a goal
problem.getSuccessors(state) # returns [(next_state, action, cost), ...]
```

Before writing anything, add these print statements at the top of `depthFirstSearch` and run it to see what each method actually returns:

```{code-block} python
print("Start:", problem.getStartState())
print("Is start the goal?", problem.isGoalState(problem.getStartState()))
print("Successors of start:", problem.getSuccessors(problem.getStartState()))
```

Running this on `tinyMaze` will print something like:

```
Start: (5, 5)
Is start the goal? False
Successors of start: [((5, 4), 'South', 1), ((4, 5), 'West', 1)]
```

The state is a position `(x, y)`. Each successor is a tuple of three elements: the next position, the action string to get there (`'North'`, `'South'`, `'East'`, `'West'`), and the step cost (always 1 in these mazes). Remove these prints once you understand the structure.

Now let us go through every piece of the NumPy algorithms and see what replaces it.

#### Neighbours: adjacency matrix → `getSuccessors`

In the NumPy version you found neighbours by querying the adjacency matrix:

```{code-block} python
neighbors = np.where(adjacency_matrix[current_index] == 1)[0]
for neighbor_index in neighbors:
    ...
```

In Pacman you call `getSuccessors` on the current state directly. It returns the neighbours together with the action needed to reach each one:

```{code-block} python
for next_state, action, cost in problem.getSuccessors(current_state):
    ...
```

You no longer need an adjacency matrix or integer indices — `getSuccessors` handles all of that internally.

#### Visited: NumPy boolean array → Python `set`

In the NumPy version the visited array was indexed by integer node indices:

```{code-block} python
visited = np.zeros(len(nodes), dtype=bool)
visited[source_index] = True
if not visited[neighbor_index]:
    visited[neighbor_index] = True
```

In Pacman states are `(x, y)` tuples, not integers, so you cannot use a NumPy array as a lookup table. Use a plain Python `set` instead:

```{code-block} python
visited = set()
visited.add(problem.getStartState())

# to check whether a state has been visited:
if next_state not in visited:
    visited.add(next_state)
```

#### Path reconstruction: `parent` array → actions carried in the frontier

In the NumPy version you stored a `parent` array and reconstructed the path backwards at the end:

```{code-block} python
parent = np.ones(len(nodes), dtype=int) * -1
parent[neighbor_index] = current_index
# ... at the end, walk backwards from target to source
```

In Pacman you cannot index an array with `(x, y)` tuples. The standard solution is to carry the path **forward** inside each frontier element. Each item you push is a tuple `(state, actions_so_far)`, where `actions_so_far` is the list of action strings taken from the start to reach that state:

```{code-block} python
# Initialise the frontier with the start state and an empty action list
frontier.push((problem.getStartState(), []))
```

When you pop a node and expand it, you push each successor with the path extended by one action:

```{code-block} python
state, actions = frontier.pop()

for next_state, action, cost in problem.getSuccessors(state):
    if next_state not in visited:
        frontier.push((next_state, actions + [action]))
```

There is no reconstruction step at the end. When you pop a node and `problem.isGoalState(state)` returns `True`, the `actions` list already contains the complete path — return it directly:

```{code-block} python
if problem.isGoalState(state):
    return actions
```

That list of action strings is exactly what `SearchAgent` needs to move Pacman through the maze step by step.

#### The loop condition

In the NumPy version the loop ran while the list was non-empty or while the pointer had not reached the end. With `util.Stack` and `util.Queue` this simplifies to:

```{code-block} python
while not frontier.isEmpty():
    state, actions = frontier.pop()
    ...
```

If the frontier empties without finding the goal, return an empty list:

```{code-block} python
return []
```

---

## Exercise

**Part 1 — validate on a small graph.** Before working on the maze, build a small graph by hand in a notebook and run `dfs_numpy` and `bfs_numpy` on it. This will serve as your ground truth:

```{code-block} python
import networkx as nx

G = nx.Graph()
G.add_edges_from([(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,4)])

steps_dfs, path_dfs = dfs_numpy(G, source=0, target=4)
steps_bfs, path_bfs = bfs_numpy(G, source=0, target=4)

print("DFS —", steps_dfs, "steps, path:", path_dfs)
print("BFS —", steps_bfs, "steps, path:", path_bfs)
```

Visualise the graph and highlight the two paths. Verify by hand that BFS returns the shortest one.

**Part 2 — implement in Pacman.** Open `search.py` and implement `depthFirstSearch` and `breadthFirstSearch` inside their respective stubs, following the translation described above. Use `util.Stack` for DFS and `util.Queue` for BFS. The loop condition changes from `while len(to_explore) > 0` to `while not frontier.isEmpty()`, and the pop becomes `frontier.pop()` — everything else is a direct translation of the NumPy version.

Test each implementation on the mazes below. The `-z .5` flag just zooms out so the big maze fits on screen.

```bash
# DFS
python pacman.py -l tinyMaze   -p SearchAgent
python pacman.py -l mediumMaze -p SearchAgent
python pacman.py -l bigMaze    -z .5 -p SearchAgent

# BFS
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
python pacman.py -l bigMaze    -p SearchAgent -a fn=bfs -z .5
```

Run the autograder to confirm correctness:

```bash
python autograder.py -q q1   # DFS
python autograder.py -q q2   # BFS
```

**Part 3 — report.** For each algorithm and each maze, report the path length (number of actions printed by the game) and the number of nodes expanded (printed as "Search nodes expanded"). Fill in the table and answer the questions below.

| Algorithm | Maze | Nodes expanded | Path length |
|---|---|---|---|
| DFS | tinyMaze | | |
| DFS | mediumMaze | | |
| DFS | bigMaze | | |
| BFS | tinyMaze | | |
| BFS | mediumMaze | | |
| BFS | bigMaze | | |

- Does DFS always find the shortest path? Why or why not?
- Does BFS always find the shortest path? Under what condition is that guaranteed?
