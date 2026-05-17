# backend/q_learning.py

import numpy as np

class QLearningAgent:
    def __init__(self, grid_size, actions=4, alpha=0.15, gamma=0.98, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.998):
        self.grid_size = grid_size
        self.actions = actions

        self.alpha = alpha      # Increased for faster mapping of new rewards
        self.gamma = gamma      # Increased to value long-term path discovery
        self.epsilon = epsilon  
        self.epsilon_min = epsilon_min # Higher floor to ensure continuous patrolling
        self.epsilon_decay = epsilon_decay

        # Boltzmann Exploration (Softmax)
        self.temperature = 1.0  
        self.temp_min = 0.2    # Higher floor for more "natural" stochastic movement
        self.temp_decay = 0.999 

        # Q-table: (x, y, action)
        self.q_table = np.zeros((grid_size, grid_size, actions))
        
        # Training metrics
        self.history = {"rewards": [], "epsilon": [], "temp": []}

    # =========================
    # ACTION SELECTION
    # =========================
    def choose_action(self, state, exploit_only=False, method="softmax"):
        i, j = state
        q_values = self.q_table[i, j]

        if exploit_only:
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)

        if method == "epsilon_greedy":
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.actions)
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
        
        elif method == "softmax":
            # Softmax (Boltzmann) exploration
            # Use a slightly shifted q_values to avoid overflow in exp
            shifted_q = (q_values - np.max(q_values)) / max(self.temperature, 1e-8)
            exp_q = np.exp(shifted_q)
            probs = exp_q / np.sum(exp_q)
            return np.random.choice(self.actions, p=probs)
        
        return np.random.randint(self.actions)

    # =========================
    # Q-LEARNING UPDATE
    # =========================
    def update(self, state, action, reward, next_state):
        i, j = state
        ni, nj = next_state

        # Bellman Equation update
        best_next = np.max(self.q_table[ni, nj])
        
        target = reward + self.gamma * best_next
        self.q_table[i, j, action] += self.alpha * (target - self.q_table[i, j, action])

    # =========================
    # TRAINING LOOP
    # =========================
    def train(self, env, episodes=1000):
        """Train the agent and return episode rewards."""
        episode_rewards = []
        
        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Use epsilon-greedy for core discovery, softmax for refinement
                # Actually, let's use a hybrid approach
                action = self.choose_action(state, method="epsilon_greedy" if ep < episodes * 0.7 else "softmax")
                
                next_state, reward, done, valid = env.step(action)
                
                # Update Q-table
                self.update(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
            
            # Decay exploration params
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.temperature = max(self.temp_min, self.temperature * self.temp_decay)
            
            episode_rewards.append(float(total_reward))
            
            if (ep + 1) % 100 == 0:
                print(f"Episode {ep+1}/{episodes} | Avg Reward: {np.mean(episode_rewards[-100:]):.2f} | Epsilon: {self.epsilon:.3f} | T: {self.temperature:.3f}")

        self.history["rewards"].extend(episode_rewards)
        self.history["epsilon"].append(float(self.epsilon))
        self.history["temp"].append(float(self.temperature))
        
        return episode_rewards

    # =========================
    # PERSISTENCE
    # =========================
    def save(self, path="models/q_table.npy"):
        import os
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path="models/q_table.npy"):
        import os
        if os.path.exists(path):
            self.q_table = np.load(path)
            self.epsilon = self.epsilon_min # Low exploration after loading
            return True
        return False