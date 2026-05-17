# backend/simulation.py

import numpy as np
import matplotlib.pyplot as plt


def simulate_agent(env, agent, steps=40):
    """
    Run trained agent and record path
    """
    state = env.reset()

    path = [state.copy()]

    for _ in range(steps):
        i, j = state

        # Choose best learned action (NO randomness)
        action = np.argmax(agent.q_table[i, j])

        next_state, reward, done = env.step(action)

        path.append(next_state.copy())
        state = next_state

    return path


def plot_simulation(level_grid, path):
    """
    Plot patrol path over crime grid
    """
    plt.figure(figsize=(7, 7))

    # 🔴 Crime heatmap
    plt.imshow(level_grid, cmap='RdYlGn_r')

    # 🔵 Patrol path
    x = [p[1] for p in path]
    y = [p[0] for p in path]

    plt.plot(x, y, marker='o', linewidth=2, label="Patrol Path")

    # 🟢 Start
    plt.scatter(x[0], y[0], s=120, label="Start")

    # ⚫ End
    plt.scatter(x[-1], y[-1], s=120, label="End")

    plt.title("Smart Crime Patrol Simulation")
    plt.legend()
    plt.grid(False)
    plt.show()