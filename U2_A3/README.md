# AI Lab Assignment — A* Search & Minimax with Alpha-Beta Pruning

**Student:** Darwin Jacob
**Institution:** Saveetha University (SIMATS), Chennai
**Subject:** Artificial Intelligence Laboratory

## Overview

This project contains the complete solution for a two-part AI lab assignment:

1. **Question 1 (10 Marks):** Dry run of the **A\* Search Algorithm** on a
   weighted graph from node `A` to goal node `G`, using given heuristic
   values.
2. **Question 2 (10 Marks):** Dry run of **Minimax with Alpha-Beta Pruning**
   on a two-ply adversarial game tree, evaluated left to right.

## File Structure

| File | Description |
|---|---|
| `Problem_Statement.pdf` | Both questions exactly as given, formatted with the graph, game tree, heuristic table, and task requirements. |
| `Solution.pdf` | Full worked dry run for both questions — iteration-by-iteration A* table and node-by-node Alpha-Beta trace, with explanations and final answers. |
| `search_algorithms.py` | Python implementation that programmatically performs both dry runs (A* and Minimax/Alpha-Beta) and generates `output.png`. |
| `output.png` | Visualization: (left) the search graph with the A* optimal path highlighted, (right) the game tree with the pruned leaf marked. |
| `Report.pdf` | Formal lab report — objective, theory, methodology, results, discussion, and conclusion. |
| `README.md` | This file. |

## How to Run the Code

Requirements: Python 3, `matplotlib`

```bash
pip install matplotlib
python3 search_algorithms.py
```

Running the script will:
- Print the full A* Search dry run (Current Node, Open List, Closed List, g, h, f) for every iteration.
- Print the full Minimax with Alpha-Beta Pruning dry run (alpha, beta, selected value, pruned nodes).
- Print a final summary of both results.
- Save `output.png` — a two-panel visualization of the search graph (optimal path highlighted) and the game tree (pruned branch marked).

## Results Summary

### Question 1 — A* Search
- **Optimal Path:** A → B → E → G
- **Total Path Cost:** 6

### Question 2 — Minimax with Alpha-Beta Pruning
- **Best Move for MAX:** Left branch (MIN node with value 3)
- **Final Minimax Value:** 3
- **Pruned Nodes:** Right MIN node's third leaf (value = 2), pruned because α (3) ≥ β (1)

## Verification

The manual dry run (documented in `Solution.pdf`) and the Python program
(`search_algorithms.py`) produce **identical results**, confirming the
correctness of both the hand-worked solution and the code.
