"""
=====================================================================
 AI LAB ASSIGNMENT : A* SEARCH  &  MINIMAX WITH ALPHA-BETA PRUNING
=====================================================================
 Author  : DARWIN 
 Purpose : Dry-run implementation + verification for:
              Question 1 - A* Search Algorithm (Graph: A -> G)
              Question 2 - Minimax with Alpha-Beta Pruning (Game Tree)

 This script:
   1. Builds the weighted graph and heuristic table from the
      question and runs A* Search, printing every iteration
      (Current Node, Open List, Closed List, g, h, f).
   2. Builds the given 2-ply game tree and runs Minimax with
      Alpha-Beta Pruning, printing alpha/beta updates and pruned
      branches at every node.
   3. Renders a single PNG (output.png) with two panels:
        - Left  : the search graph with the optimal path highlighted
        - Right : the game tree with pruned branch marked in red
=====================================================================
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import heapq

# =====================================================================
# QUESTION 1 : A* SEARCH ALGORITHM
# =====================================================================

# ---- Graph definition (undirected, weighted) -----------------------
graph = {
    "A": {"B": 2, "C": 4},
    "B": {"A": 2, "C": 3, "D": 7, "E": 2},
    "C": {"A": 4, "B": 3, "E": 3},
    "D": {"B": 7, "E": 2},
    "E": {"B": 2, "C": 3, "D": 2, "G": 2},
    "G": {"E": 2},
}

# ---- Heuristic values h(n) : straight-line estimate to goal G ------
heuristic = {"A": 7, "B": 6, "C": 4, "D": 3, "E": 2, "G": 0}

START, GOAL = "A", "G"


def a_star_search(graph, heuristic, start, goal):
    """
    Performs A* Search and prints a full dry run table for every
    iteration: Current Node, Open List, Closed List, g(n), h(n), f(n).
    Returns (optimal_path, total_cost).
    """

    # Each open-list entry: (f, g, node, path)
    open_list = [(heuristic[start], 0, start, [start])]
    closed_list = []

    # best-known g(n) for every node reached so far
    best_g = {start: 0}

    iteration = 0
    print("=" * 100)
    print("QUESTION 1 : A* SEARCH DRY RUN  (Start = A, Goal = G)")
    print("=" * 100)

    while open_list:
        iteration += 1

        # sort open list for clean display (by f, then node name)
        open_list.sort(key=lambda x: (x[0], x[2]))
        f, g, current, path = heapq.heappop(open_list)
        h = heuristic[current]

        print(f"\n--- Iteration {iteration} ---")
        print(f"Current Node : {current}")
        print(f"g({current}) = {g} , h({current}) = {h} , f({current}) = {f}")

        open_display = ", ".join(
            f"{n}(g={gg},h={heuristic[n]},f={ff})"
            for ff, gg, n, _ in sorted(open_list, key=lambda x: (x[0], x[2]))
        )
        print(f"Open List    : [{open_display}]")
        print(f"Closed List  : {closed_list if closed_list else '[]'}")

        # Goal test happens when the goal node is EXPANDED
        # (popped as the lowest f from the open list)
        if current == goal:
            print(f"\n*** Goal node '{goal}' reached with minimum f-value! ***")
            print(f"Optimal Path      : {' -> '.join(path)}")
            print(f"Total Path Cost   : {g}")
            return path, g

        closed_list.append(current)

        for neighbor, cost in graph[current].items():
            if neighbor in closed_list:
                continue
            tentative_g = g + cost
            if neighbor not in best_g or tentative_g < best_g[neighbor]:
                best_g[neighbor] = tentative_g
                new_f = tentative_g + heuristic[neighbor]
                open_list.append((new_f, tentative_g, neighbor, path + [neighbor]))

    return None, float("inf")


# =====================================================================
# QUESTION 2 : MINIMAX WITH ALPHA-BETA PRUNING
# =====================================================================

# ---- Game tree definition -------------------------------------------
# MAX (root)
#   |-- MIN (Left)  -> leaves [3, 5, 6]
#   |-- MIN (Right) -> leaves [9, 1, 2]
game_tree = {
    "MIN_L": [3, 5, 6],
    "MIN_R": [9, 1, 2],
}

pruned_nodes = []
trace_log = []


def minimax_ab(node_label, values, alpha, beta, maximizing):
    """
    Runs Minimax with Alpha-Beta pruning on a single MIN (or MAX) layer
    of leaves, logging every comparison and any pruning event.
    """
    best = float("-inf") if maximizing else float("inf")

    for i, v in enumerate(values):
        leaf_name = f"{node_label}_leaf{i+1}({v})"

        if maximizing:
            best = max(best, v)
            alpha = max(alpha, best)
        else:
            best = min(best, v)
            beta = min(beta, best)

        trace_log.append(
            f"  Visit {leaf_name:<18} -> alpha={alpha}, beta={beta}, "
            f"current best={best}"
        )

        if alpha >= beta:
            remaining = values[i + 1:]
            if remaining:
                for j, rv in enumerate(remaining):
                    pruned_name = f"{node_label}_leaf{i+2+j}(value={rv})"
                    pruned_nodes.append(pruned_name)
                    trace_log.append(
                        f"  PRUNED {pruned_name}  (alpha {alpha} >= beta {beta})"
                    )
            break

    return best, alpha, beta


def run_minimax_alpha_beta():
    print("\n" + "=" * 100)
    print("QUESTION 2 : MINIMAX WITH ALPHA-BETA PRUNING DRY RUN")
    print("=" * 100)

    root_alpha, root_beta = float("-inf"), float("inf")
    print(f"\nRoot MAX node initial: alpha = -inf, beta = +inf")

    # ---- Left MIN subtree ----
    print("\n[Expanding LEFT MIN node : children = 3, 5, 6]")
    left_val, la, lb = minimax_ab("MIN_L", game_tree["MIN_L"],
                                   root_alpha, root_beta, maximizing=False)
    for line in trace_log:
        print(line)
    trace_log.clear()
    print(f"  -> LEFT MIN node value returned to MAX = {left_val}")

    root_alpha = max(root_alpha, left_val)
    print(f"  MAX updates: alpha = max(-inf, {left_val}) = {root_alpha}")

    # ---- Right MIN subtree ----
    print("\n[Expanding RIGHT MIN node : children = 9, 1, 2]")
    right_val, ra, rb = minimax_ab("MIN_R", game_tree["MIN_R"],
                                    root_alpha, float("inf"), maximizing=False)
    for line in trace_log:
        print(line)
    trace_log.clear()
    print(f"  -> RIGHT MIN node value returned to MAX = {right_val}")

    root_alpha = max(root_alpha, right_val)
    final_value = max(left_val, right_val)

    print(f"\nMAX (root) final alpha = {root_alpha}")
    print(f"MAX (root) selects  = max({left_val}, {right_val}) = {final_value}")

    best_move = "LEFT branch (MIN_L, value 3)" if final_value == left_val else "RIGHT branch (MIN_R, value 1)"

    print(f"\nBest Move for MAX      : {best_move}")
    print(f"Final Minimax Value     : {final_value}")
    print(f"Pruned Nodes             : {pruned_nodes if pruned_nodes else 'None'}")

    return final_value, best_move, pruned_nodes, left_val, right_val


# =====================================================================
# VISUALIZATION : output.png  (two-panel figure)
# =====================================================================

def render_output_png(optimal_path, path_cost, final_value, pruned, left_val, right_val,
                       filename="output.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ---------------- Panel 1 : A* Search Graph ----------------------
    ax1 = axes[0]
    positions = {
        "A": (2, 5),
        "B": (0.5, 3.2),
        "C": (3.5, 3.2),
        "D": (0.5, 1.2),
        "E": (3.5, 1.2),
        "G": (3.5, -0.5),
    }
    edges_drawn = set()
    for u in graph:
        for v, w in graph[u].items():
            if (v, u) in edges_drawn:
                continue
            edges_drawn.add((u, v))
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            on_path = (u in optimal_path and v in optimal_path and
                       abs(optimal_path.index(u) - optimal_path.index(v)) == 1)
            color = "#e63946" if on_path else "#888888"
            lw = 3.2 if on_path else 1.3
            ax1.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=1)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax1.text(mx, my, str(w), fontsize=10, color="#1d3557",
                      fontweight="bold", ha="center", va="center",
                      bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="none"))

    for node, (x, y) in positions.items():
        on_path = node in optimal_path
        fc = "#e63946" if on_path else "#a8dadc"
        ax1.scatter([x], [y], s=1500, color=fc, edgecolors="#1d3557",
                    linewidths=2, zorder=2)
        ax1.text(x, y, f"{node}\nh={heuristic[node]}", ha="center", va="center",
                  fontsize=10, fontweight="bold", zorder=3)

    ax1.set_title(f"A* Search — Optimal Path: {' → '.join(optimal_path)}  "
                   f"(Cost = {path_cost})", fontsize=12, fontweight="bold")
    ax1.axis("off")
    ax1.set_xlim(-0.8, 4.5)
    ax1.set_ylim(-1.3, 6)

    # ---------------- Panel 2 : Minimax Game Tree ---------------------
    ax2 = axes[1]
    root_pos = (5, 3)
    min_l_pos = (2.5, 1.8)
    min_r_pos = (7.5, 1.8)
    leaf_l = [(1, 0.3), (2.5, 0.3), (4, 0.3)]
    leaf_r = [(6, 0.3), (7.5, 0.3), (9, 0.3)]
    left_vals = [3, 5, 6]
    right_vals = [9, 1, 2]

    def node_box(pos, label, kind, is_pruned=False):
        x, y = pos
        fc = "#f1c40f" if kind == "MAX" else "#a8dadc"
        if is_pruned:
            fc = "#e0e0e0"
        ec = "#e63946" if is_pruned else "#1d3557"
        ax2.scatter([x], [y], s=1600 if kind != "leaf" else 1100,
                    color=fc, edgecolors=ec, linewidths=2.2,
                    zorder=3, marker="s" if kind == "leaf" else "o")
        txt_color = "#999999" if is_pruned else "black"
        ax2.text(x, y, label, ha="center", va="center", fontsize=10,
                  fontweight="bold", zorder=4, color=txt_color)

    # edges
    for lp in leaf_l:
        ax2.plot([min_l_pos[0], lp[0]], [min_l_pos[1], lp[1]], color="#888", zorder=1)
    for rp in leaf_r:
        ax2.plot([min_r_pos[0], rp[0]], [min_r_pos[1], rp[1]], color="#888", zorder=1)
    ax2.plot([root_pos[0], min_l_pos[0]], [root_pos[1], min_l_pos[1]], color="#888", zorder=1)
    ax2.plot([root_pos[0], min_r_pos[0]], [root_pos[1], min_r_pos[1]], color="#888", zorder=1)

    node_box(root_pos, f"MAX\n{final_value}", "MAX")
    node_box(min_l_pos, f"MIN\n{left_val}", "MIN")
    node_box(min_r_pos, f"MIN\n{right_val}", "MIN")

    for pos, val in zip(leaf_l, left_vals):
        node_box(pos, str(val), "leaf")

    for pos, val in zip(leaf_r, right_vals):
        is_pruned = any(f"value={val})" in p and "MIN_R" in p for p in pruned)
        node_box(pos, str(val), "leaf", is_pruned=is_pruned)
        if is_pruned:
            ax2.text(pos[0], pos[1] - 0.35, "PRUNED", color="#e63946",
                      fontsize=8, ha="center", fontweight="bold")

    ax2.set_title(f"Minimax with Alpha-Beta Pruning — Final Value = {final_value}",
                  fontsize=12, fontweight="bold")
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-0.8, 4)

    legend_elems = [
        mpatches.Patch(color="#f1c40f", label="MAX node"),
        mpatches.Patch(color="#a8dadc", label="MIN node"),
        mpatches.Patch(color="#e0e0e0", label="Pruned leaf"),
    ]
    ax2.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=8,
              bbox_to_anchor=(0.5, -0.15), frameon=False)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"\n[Saved visualization to {filename}]")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    optimal_path, path_cost = a_star_search(graph, heuristic, START, GOAL)
    final_value, best_move, pruned, left_val, right_val = run_minimax_alpha_beta()

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"Q1  Optimal Path (A*)      : {' -> '.join(optimal_path)}")
    print(f"Q1  Total Path Cost        : {path_cost}")
    print(f"Q2  Best Move for MAX      : {best_move}")
    print(f"Q2  Final Minimax Value    : {final_value}")
    print(f"Q2  Pruned Nodes           : {pruned if pruned else 'None'}")

    render_output_png(optimal_path, path_cost, final_value, pruned, left_val, right_val,
                       filename="output.png")
