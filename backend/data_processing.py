# backend/data_processing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from backend.grid import CrimeGrid
from backend.utils import plot_crime_levels
from backend.environment import CrimeEnv
from backend.q_learning import QLearningAgent
from backend.simulation import simulate_agent, plot_simulation


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    # Normalize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Select required columns
    df = df[['date', 'latitude', 'longitude', 'primary_type']]

    # Drop missing values
    df = df.dropna()

    # CRITICAL: Filter to Chicago City Boundaries to remove outliers
    # Chicago is roughly between Lat [41.6, 42.1] and Lon [-87.95, -87.5]
    df = df[
        (df['latitude'] >= 41.6) & (df['latitude'] <= 42.1) &
        (df['longitude'] >= -87.95) & (df['longitude'] <= -87.5)
    ]

    # Convert date column
    df['date'] = pd.to_datetime(df['date'], format='mixed')

    return df


def filter_recent_data(df, years=3):
    cutoff_date = df['date'].max() - pd.DateOffset(years=years)
    df = df[df['date'] >= cutoff_date]
    return df


def process_pipeline(file_path):
    df = load_data(file_path)
    df = clean_data(df)
    df = filter_recent_data(df)

    print("Processed Data Shape:", df.shape)

    return df


if __name__ == "__main__":
    # =========================
    # PHASE 1: DATA PROCESSING
    # =========================
    df = process_pipeline("backend/data/crimes.csv")

    # =========================
    # PHASE 2: GRID CREATION
    # =========================
    grid_system = CrimeGrid(df, grid_size=20)
    grid = grid_system.populate_grid()
    grid = grid_system.normalize_grid()

    # =========================
    # PHASE 3: CRIME LEVELS
    # =========================
    level_grid, _ = grid_system.get_crime_levels()

    print("\n===== BASIC INFO =====")
    print("Grid shape:", grid.shape)
    print("Level grid shape:", level_grid.shape)

    print("\n===== SAMPLE GRID (5x5) =====")
    print(level_grid[:5, :5])

    print("\n===== VALUE RANGE =====")
    print("Min value:", np.min(grid))
    print("Max value:", np.max(grid))

    print("\n===== CRIME LEVEL DISTRIBUTION =====")
    unique, counts = np.unique(level_grid, return_counts=True)
    distribution = dict(zip(unique, counts))
    print(distribution)

    print("\n===== PERCENTAGE DISTRIBUTION =====")
    total = level_grid.size
    for level, count in distribution.items():
        print(f"Level {level}: {count} cells ({(count/total)*100:.2f}%)")

    print("\n===== RANDOM CELL CHECK =====")
    for _ in range(5):
        i = np.random.randint(0, level_grid.shape[0])
        j = np.random.randint(0, level_grid.shape[1])
        print(f"Cell ({i},{j}) → Level:", level_grid[i][j])

    # Optional visualization
    # plot_crime_levels(level_grid)

    # =========================
    # PHASE 4: RL ENVIRONMENT TEST
    # =========================
    print("\n===== TESTING RL ENVIRONMENT =====")

    env = CrimeEnv(level_grid)
    state = env.reset()

    print("Start position:", state)

    for step in range(5):
        action = np.random.randint(0, 4)
        next_state, reward, done, _ = env.step(action)

        print(f"Step {step}: Action {action} → Position {next_state}, Reward {reward}")

    # =========================
    # PHASE 5: TRAINING Q-LEARNING AGENT
    # =========================
    print("\n===== TRAINING Q-LEARNING AGENT =====")

    env = CrimeEnv(level_grid)
    agent = QLearningAgent(grid_size=20)

    episodes = 500
    rewards_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(75):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done:
                break

        # Epsilon decay
        agent.epsilon = max(0.05, agent.epsilon * 0.995)

        rewards_per_episode.append(total_reward)

        if ep % 50 == 0:
            print(f"Episode {ep} → Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    # PHASE 6: PLOT TRAINING REWARD GRAPH (SMOOTHED)
    # =========================
    print("\n===== PLOTTING TRAINING REWARD GRAPH =====")

    window_size = 20  # Moving average window
    smoothed_rewards = np.convolve(
        rewards_per_episode,
        np.ones(window_size) / window_size,
        mode='valid'
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(window_size, len(rewards_per_episode) + 1),
        smoothed_rewards,
        linewidth=2
    )
    plt.xlabel("Training Episodes")
    plt.ylabel("Average Total Reward")
    plt.title("Smoothed Reward Trend across Training Episodes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("backend/fig6_training_rewards.png", dpi=300)
    plt.show()

    print("Training reward graph saved as: backend/fig6_training_rewards.png")

    # =========================
    # PHASE 7: SIMULATING TRAINED AGENT
    # =========================
    print("\n===== SIMULATING TRAINED AGENT =====")

    path = simulate_agent(env, agent, steps=40)
    plot_simulation(level_grid, path)