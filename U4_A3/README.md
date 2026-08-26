# Experiment 2 — Reinforcement Learning: Q-Learning

## Overview
This project implements Q-Learning for a simple 4×4 Grid World environment.

## Environment
- Start state: 0
- Goal state: 15
- Obstacles: 5 and 10
- Actions: Up, Down, Left, Right
- Goal reward: +10
- Normal step: -1
- Obstacle: -5

## Hyperparameters
- Learning rate (alpha): 0.10
- Discount factor (gamma): 0.90
- Exploration rate (epsilon): 0.20
- Episodes: 100

## Files
- `Problem_Statement.pdf`
- `Solution.pdf`
- `q_learning.py`
- `output.png`
- `Report.pdf`
- `README.md`
- `cumulative_reward.png`
- `q_table_evolution.png`
- `q_learning_final_policy.csv`

## Requirements
```bash
pip install numpy matplotlib pandas
```

## Run
```bash
python q_learning.py
```

## Learning
The Q-table starts with zeros. Through repeated interaction, rewards propagate backward from the goal and actions leading toward useful states receive higher Q-values.

## Convergence
Cumulative reward generally improves as the agent learns. Some fluctuation remains because epsilon-greedy exploration is intentionally retained.

## Result
The final policy is obtained by selecting the action with the maximum Q-value for each non-terminal state.
