# Search Algorithms

## Practical Session: Custom Mazes and Algorithm Comparison

### Introduction

In the previous two sessions you implemented DFS, BFS, UCS and A\* in the Pacman framework. All tests were run on mazes provided by the course. In this session you will design and test your own mazes, observe how the structure of the maze affects each algorithm, and consolidate everything with a final comparison.

---

### How mazes are defined in Pacman

Mazes live in the `layouts/` folder inside the project directory. Each layout is a plain text file with extension `.lay`. Open any existing one to see the format:

```bash
cat ./layouts/smallClassic.lay
```

You will see something like:

```
%%%%%%%%%%%%%%%%%%%%
%......%G  G%......%
%.%%...%%  %%...%%.%
%.%o.%........%.o%.%
%.%%.%.%%%%%%.%.%%.%
%........P.........%
%%%%%%%%%%%%%%%%%%%%
ahmedbegga@MacBook-Air
```

Each character has a fixed meaning:

| Character | Meaning |
|---|---|
| `%` | Wall |
| ` ` (space) | Empty corridor — Pacman can walk here |
| `P` | Pacman's starting position (exactly one per layout) |
| `.` | Food dot — the goal in single-food problems |
| `o` | Power capsule (not relevant for search) |

The grid is read top-to-bottom, left-to-right. The coordinate system places `(0, 0)` at the bottom-left, so the top-left character in the file corresponds to the highest $y$ coordinate. Walls must form a closed border around the maze — there should be no gaps on the edges.

A minimal valid layout looks like this:

```
%%%%%
%P  %
% %%% 
%   %
% . %
%%%%%
```

Save your layout file as `layouts/myMaze.lay` and run it with:

```bash
python pacman.py -l myMaze -p SearchAgent -a fn=bfs
```

You can swap `fn=bfs` for `fn=dfs`, `fn=ucs`, or `fn=astar,heuristic=manhattanHeuristic` to test all four algorithms on the same maze.

---

### Designing mazes that reveal algorithm behaviour

Not all mazes are equally informative. The goal of this session is to design layouts that expose a specific property of each algorithm. Below are three templates to guide you, each targeting a different behaviour.

#### Maze A — DFS finds a very long path

DFS follows one branch as deep as possible before backtracking. A long corridor that loops away from the goal will trap DFS into exploring a huge detour while BFS finds the direct route immediately.

```
%%%%%%%%%%%%
%P         %
%%%%%%%%%% %
%         .%
%%%%%%%%%%%%
```

In this layout the only path goes all the way right, then wraps around — DFS will walk the long way because it dives in and does not backtrack until it is forced to. Try making the corridor longer or adding dead ends to exaggerate the effect.

#### Maze B — BFS expands many nodes but finds the optimal path

BFS expands level by level in all directions, so in an open maze with few walls it will visit almost every cell before reaching a distant goal. This makes the node count large even though the path is short.

```
%%%%%%%%%%%%%
%P          %
%           %
%           %
%          .%
%%%%%%%%%%%%%
```

Place the goal far from the start in an open space. BFS will flood the entire grid before finding it.

#### Maze C — A\* shines over UCS with a good heuristic

Design a maze where there are two routes of similar length but one is more "straight" toward the goal. A\* with the Manhattan distance heuristic will prefer the straighter route and expand fewer nodes than UCS, which is blind to direction.

```
%%%%%%%%%%%%%%%
%P    %       %
%     %   .   %
%     %       %
%             %
%%%%%%%%%%%%%%%
```

The wall in the middle creates a choice: go around it to the left or to the right. The Manhattan heuristic will guide A\* toward the side closer to the goal, while UCS treats both sides equally.

---

## Exercise

**Part 1 — run the provided templates.** Save each of the three maze templates above as a `.lay` file in the `layouts/` folder (`mazeA.lay`, `mazeB.lay`, `mazeC.lay`). Run all four algorithms on each maze and fill in the table:

```bash
# Example for mazeA — repeat for mazeB, mazeC and for fn=dfs, fn=ucs, fn=astar,heuristic=manhattanHeuristic
python pacman.py -l mazeA -p SearchAgent -a fn=bfs
```

| Maze | Algorithm | Nodes expanded | Path length |
|---|---|---|---|
| mazeA | DFS | | |
| mazeA | BFS | | |
| mazeA | UCS | | |
| mazeA | A\* | | |
| mazeB | DFS | | |
| mazeB | BFS | | |
| mazeB | UCS | | |
| mazeB | A\* | | |
| mazeC | DFS | | |
| mazeC | BFS | | |
| mazeC | UCS | | |
| mazeC | A\* | | |

**Part 2 — design your own maze.** Create a layout of at least 10×10 cells that is specifically designed to make one algorithm clearly outperform the others in terms of nodes expanded. Save it as `layouts/myMaze.lay`, run all four algorithms on it, and report the results in the same table format as above.

In your notebook, explain in a short paragraph why you expect the maze to favour that algorithm and whether the results confirmed your expectation.

**Part 3 — final comparison.** Using all the results collected across the three sessions, answer the following questions:

- Across all mazes tested, which algorithm consistently expands the fewest nodes? Does the answer change depending on the maze structure?
- BFS and UCS both guarantee the optimal path. Did they find paths of the same length on `mediumMaze`? Explain why or why not.
- Describe a scenario — a specific maze shape or cost structure — where you would choose each of the four algorithms. Justify your choice.

---

## Summary of all three sessions

### Session 1 — BFS and DFS

**What you learned:** the structural difference between DFS (LIFO, stack) and BFS (FIFO, queue), how to translate a NumPy graph implementation into the Pacman `problem` interface, and why BFS is optimal for uniform-cost graphs while DFS is not.

**What you implemented:** `depthFirstSearch` and `breadthFirstSearch` in `search.py`.

**Exercises completed:**
- Validate `dfs_numpy` and `bfs_numpy` on a small handmade NetworkX graph.
- Implement both functions in `search.py` using `util.Stack` and `util.Queue`.
- Test on `tinyMaze`, `mediumMaze`, and `bigMaze`.
- Pass autograder questions `q1` and `q2`.
- Report nodes expanded and path length for each algorithm and maze.

---

### Session 2 — UCS and A\*

**What you learned:** how to incorporate edge costs using a priority queue, what makes A\* different from UCS (the heuristic term $h(n)$), why admissibility guarantees optimality, and why visited nodes must be marked on pop rather than on push in cost-sensitive search.

**What you implemented:** `uniformCostSearch` and `aStarSearch` in `search.py`.

**Exercises completed:**
- Implement `uniformCostSearch` using `util.PriorityQueue` with `cost_so_far` as priority.
- Test UCS on `mediumMaze`, `mediumDottedMaze`, and `mediumScaryMaze`.
- Implement `aStarSearch` with the Manhattan distance heuristic.
- Test A\* on `bigMaze`.
- Pass autograder questions `q3` and `q4`.
- Report nodes expanded, path length, and path cost for each algorithm and maze.

---

### Session 3 — Custom mazes and comparison

**What you learned:** how Pacman layout files are structured, how maze topology influences the behaviour of each algorithm, and how to design a maze that exposes a specific algorithmic property.

**What you implemented:** three layout files (`mazeA.lay`, `mazeB.lay`, `mazeC.lay`) and one original layout (`myMaze.lay`).

**Exercises completed:**
- Run all four algorithms on the three provided maze templates and fill in the comparison table.
- Design an original maze of at least 10×10 cells that favours one specific algorithm.
- Write a short justification of your design choice and compare it against the observed results.
- Answer the three final reflection questions synthesising the results from all three sessions.