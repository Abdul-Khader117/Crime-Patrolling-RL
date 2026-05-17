import numpy as np

class CrimeEnv:
    def __init__(self, grid, max_steps=150):
        self.original_grid = grid.copy()
        self.grid_size = grid.shape[0]
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.grid = self.original_grid.copy()
        self.steps = 0
        
        # Start at a random VALID (non-water) position
        valid_indices = np.argwhere(self.grid >= 0)
        if len(valid_indices) > 0:
            idx = np.random.randint(0, len(valid_indices))
            self.agent_pos = list(valid_indices[idx])
        else:
            self.agent_pos = [0, 0]

        # NEW: Visit Tracking (counts how many times we've been to each cell)
        self.visit_counts = np.zeros((self.grid_size, self.grid_size))
        self.visit_counts[tuple(self.agent_pos)] = 1
        
        return self.agent_pos

    def step(self, action):
        i, j = self.agent_pos
        self.steps += 1

        # MOVE
        if action == 0: i -= 1      # Up
        elif action == 1: i += 1    # Down
        elif action == 2: j -= 1    # Left
        elif action == 3: j += 1    # Right

        # 1. BOUNDARY CHECK
        if i < 0 or i >= self.grid_size or j < 0 or j >= self.grid_size:
            reward = -2.0 # Negative feedback for hitting boundary
            done = self.steps >= self.max_steps
            return self.agent_pos, reward, done, False

        crime_level = self.grid[i][j]

        # 2. WATER CHECK (-1 is Water)
        if crime_level == -1:
            reward = -3.0 # Heavy penalty for water areas
            done = self.steps >= self.max_steps
            return self.agent_pos, reward, done, False

        # Valid move
        self.agent_pos = [i, j]
        new_pos = (i, j)

        # 3. REWARD SYSTEM (Refactored for Coverage)
        reward = 0
        
        # A. Crime Detection (Reduced to prevent "camping")
        if crime_level == 2: reward = 1.0     # High risk
        elif crime_level == 1: reward = 0.5   # Mid risk
        else: reward = 0.0                    # Safe zone

        # B. Exploration / Coverage (THE MAIN DRIVER)
        current_visits = self.visit_counts[new_pos]
        
        if current_visits == 0:
            reward += 5.0   # MASSIVE bonus for first-time discovery
        else:
            # C. Stagnation Penalty (Increasingly worse for repeats)
            reward -= 0.5 * current_visits 

        # Update visit tally
        self.visit_counts[new_pos] += 1

        # D. Movement / Step Penalty (Encourages speed)
        reward -= 0.1

        # Episode termination
        done = self.steps >= self.max_steps

        return self.agent_pos, reward, done, True