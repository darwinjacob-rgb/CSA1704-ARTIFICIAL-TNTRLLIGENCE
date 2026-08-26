import numpy as np
import matplotlib.pyplot as plt

# 4x4 Grid World
N = 4
START, GOAL = 0, 15
OBSTACLES = {5, 10}
ACTIONS = ["U", "D", "L", "R"]
MOVES = [(-1,0), (1,0), (0,-1), (0,1)]

alpha, gamma, epsilon = 0.10, 0.90, 0.20
episodes = 100
Q = np.zeros((16, 4))
rng = np.random.default_rng(7)

def step(state, action):
    r, c = divmod(state, N)
    dr, dc = MOVES[action]
    nr, nc = r + dr, c + dc

    if not (0 <= nr < N and 0 <= nc < N):
        return state, -1, False

    ns = nr * N + nc
    if ns in OBSTACLES:
        return ns, -5, False
    if ns == GOAL:
        return ns, 10, True
    return ns, -1, False

snapshots = {}
rewards = []

for ep in range(1, episodes + 1):
    state, total = START, 0

    for _ in range(100):
        if rng.random() < epsilon:
            action = int(rng.integers(4))
        else:
            action = int(rng.choice(np.flatnonzero(Q[state] == Q[state].max())))

        next_state, reward, done = step(state, action)

        future = 0 if done else gamma * np.max(Q[next_state])
        Q[state, action] += alpha * (reward + future - Q[state, action])

        total += reward
        state = next_state

        if done:
            break

    rewards.append(total)

    if ep in [1, 50, 100]:
        snapshots[ep] = Q.copy()

# Representative states: start and state 6
print("\nQ-values at Episodes 1, 50 and 100")
for ep in [1, 50, 100]:
    print(f"Episode {ep}")
    print("State 0 :", np.round(snapshots[ep][0], 3))
    print("State 6 :", np.round(snapshots[ep][6], 3))

# Final policy
print("\nFINAL POLICY")
for s in range(16):
    if s == GOAL:
        p = "G"
    elif s in OBSTACLES:
        p = "X"
    else:
        p = ACTIONS[np.argmax(Q[s])]
    print(f"State {s:2}: {p}")

# Plot cumulative reward
plt.plot(range(1, episodes + 1), rewards)
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Q-Learning: Cumulative Reward")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("cumulative_reward.png", dpi=180)
plt.show()
