from flask import Flask, jsonify, request
from backend.data_processing import process_pipeline
from backend.grid import CrimeGrid
from backend.environment import CrimeEnv
from backend.q_learning import QLearningAgent
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# =========================
# INIT SYSTEM
# =========================

GRID_SIZE = 20
df = process_pipeline("backend/data/crimes.csv")

grid_system = CrimeGrid(df, grid_size=GRID_SIZE)
grid = grid_system.populate_grid()
grid = grid_system.normalize_grid()
level_grid, invalid_mask = grid_system.get_crime_levels()

# Initialize environment with max_steps
env = CrimeEnv(level_grid, max_steps=100)
# Initialize agent with improved hyperparameters
agent = QLearningAgent(grid_size=GRID_SIZE, epsilon_decay=0.998)

state = env.reset()
total_reward = 0
training_history = []

# UNIFY BOUNDS with CrimeGrid
lat_min, lat_max = grid_system.min_lat, grid_system.max_lat
lon_min, lon_max = grid_system.min_lon, grid_system.max_lon

# =========================
# CORRECT GRID → LAT/LON
# =========================

def grid_to_latlng(i, j):
    """
    Ensures centering within cells.
    i: latitude index (0 is South, rising North)
    j: longitude index (0 is West, rising East)
    """
    lat_step = (lat_max - lat_min) / GRID_SIZE
    lon_step = (lon_max - lon_min) / GRID_SIZE

    # Centering logic
    lat = float(lat_min + (i + 0.5) * lat_step)
    lon = float(lon_min + (j + 0.5) * lon_step)

    return lat, lon


# =========================
# STATE API
# =========================

@app.route("/state", methods=["GET"])
def get_state():
    lat, lon = grid_to_latlng(state[0], state[1])

    grid_points = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            lat_cell, lon_cell = grid_to_latlng(i, j)
            grid_points.append({
                "lat": lat_cell,
                "lon": lon_cell,
                "level": int(level_grid[i][j]),
                "valid": bool(not invalid_mask[i][j])
            })

    lat_step = (lat_max - lat_min) / GRID_SIZE
    lon_step = (lon_max - lon_min) / GRID_SIZE

    return jsonify({
        "position": [int(state[0]), int(state[1])],
        "lat": lat,
        "lon": lon,
        "total_reward": float(total_reward),
        "grid_points": grid_points,
        "grid_size": int(GRID_SIZE),
        "lat_step": float(lat_step),
        "lon_step": float(lon_step),
        "epsilon": float(agent.epsilon),
        "history": agent.history["rewards"][-1000:] # Last 1k episodes
    })

# =========================
# TRAINING API
# =========================

@app.route("/train", methods=["POST"])
def train():
    episodes = request.json.get("episodes", 1000)
    
    # Run the training loop
    rewards = agent.train(env, episodes=episodes)

    return jsonify({
        "status": "success", 
        "epsilon": float(agent.epsilon),
        "rewards": rewards,
        "message": f"Trained for {episodes} episodes"
    })

@app.route("/save", methods=["POST"])
def save():
    agent.save("models/q_table.npy")
    return jsonify({"status": "saved"})

@app.route("/load", methods=["POST"])
def load():
    if agent.load("models/q_table.npy"):
        return jsonify({"status": "loaded", "epsilon": float(agent.epsilon)})
    return jsonify({"status": "file not found"}), 404


# =========================
# MOVE (MANUAL)
# =========================

@app.route("/move", methods=["POST"])
def move():
    global state, total_reward

    action = request.json["action"]
    state, reward, done, valid = env.step(action)
    total_reward += reward

    lat, lon = grid_to_latlng(state[0], state[1])

    return jsonify({
        "position": [int(state[0]), int(state[1])],
        "lat": lat,
        "lon": lon,
        "reward": float(reward),
        "total_reward": float(total_reward),
        "done": bool(done)
    })


# =========================
# AI MOVE (Fixed Training/Exploration)
# =========================

@app.route("/ai", methods=["POST"])
def ai_move():
    global state, total_reward

    # Select action (can still explore slightly in live mode)
    action = agent.choose_action(state)

    next_state, reward, done, valid = env.step(action)

    # Online learning (optional but helps stability)
    agent.update(state, action, reward, next_state)

    state = next_state
    total_reward += reward

    lat, lon = grid_to_latlng(state[0], state[1])

    return jsonify({
        "position": [int(state[0]), int(state[1])],
        "lat": lat,
        "lon": lon,
        "reward": float(reward),
        "total_reward": float(total_reward),
        "done": bool(done),
        "epsilon": float(agent.epsilon)
    })


# =========================
# RESET
# =========================

@app.route("/reset", methods=["POST"])
def reset():
    global state, total_reward

    state = env.reset()
    total_reward = 0

    lat, lon = grid_to_latlng(state[0], state[1])

    return jsonify({
        "position": [int(state[0]), int(state[1])],
        "lat": lat,
        "lon": lon,
        "total_reward": float(total_reward)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)