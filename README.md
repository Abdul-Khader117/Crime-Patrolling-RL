# Crime Patrolling RL

An advanced Reinforcement Learning (RL) simulation project designed to optimize crime patrol routes using the Chicago Crimes dataset. This project leverages a sophisticated Q-Learning agent operating within a dynamically generated geographical grid, balancing exploration and exploitation to maximize patrol efficiency across high-risk areas.

## 🌟 Key Features

*   **Reinforcement Learning Engine**: Custom Deep Q-Network / Q-Learning agent designed for spatial optimization.
*   **Dynamic Geographic Environment**: A grid-based representation of real-world crime data, incorporating geographical constraints (like land/water masking).
*   **Predictive Crime Modeling**: Integrates historical crime data to generate intensity heatmaps and predictive risk gradients.
*   **Reward Shaping**: Advanced reward structures featuring crime potential gradients to actively guide the agent toward high-risk zones, eliminating local maxima loops.
*   **Real-Time Visualization Dashboard**: A modern React-based frontend providing live telemetry, heatmap visualization, and real-time agent tracking.
*   **Model Persistence**: Save, load, and resume training sessions for continuous improvement.

## 🏗️ Project Architecture

The project is structured into three main components:

### 1. Backend (`/backend`)
A robust Python backend powering the RL environment and simulation logic.
*   **`environment.py`**: Defines the state space, action space, and transition dynamics of the patrol environment.
*   **`q_learning.py`**: Implementation of the Q-learning algorithm, Boltzmann (Softmax) exploration, and experience replay.
*   **`grid.py`**: Spatial processing and grid generation from geographic coordinates.
*   **`api.py`**: FastAPI/Flask server exposing endpoints for frontend communication.
*   **`data_processing.py`**: Ingestion and preprocessing pipelines for the Chicago Crimes dataset.

### 2. Frontend Dashboard (`/frontend-react`)
A responsive and interactive React application for monitoring the agent.
*   Built with React and styled for a modern, dashboard-like aesthetic.
*   Features live maps, real-time metrics (epsilon, cumulative rewards, steps), and training control panels.

### 3. Models (`/models`)
*   Directory for persisting trained agent models and Q-tables for future evaluation or continued training.

## 🚀 Getting Started

### Prerequisites
*   Python 3.8+
*   Node.js 16+
*   npm or yarn

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Abdul-Khader117/Crime-Patrolling-RL.git
    cd Crime-Patrolling-RL
    ```

2.  **Backend Setup:**
    ```bash
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install Python dependencies
    pip install -r requirements.txt
    ```

3.  **Frontend Setup:**
    ```bash
    cd frontend-react
    npm install
    ```

### Running the Application

1.  **Start the Backend Server:**
    ```bash
    # Ensure you are in the virtual environment
    cd backend
    python api.py
    ```

2.  **Start the Frontend Dashboard:**
    ```bash
    cd frontend-react
    npm start
    ```

3.  Open your browser and navigate to `http://localhost:3000` to interact with the simulation.

## 📊 Methodology

1.  **Data Preprocessing**: The Chicago Crimes dataset is cleaned, normalized, and mapped onto a spatial grid.
2.  **State Representation**: The agent's state is defined by its current grid coordinates and the surrounding crime intensity.
3.  **Action Space**: The agent can move in cardinal directions (Up, Down, Left, Right).
4.  **Reward Function**: The agent receives positive rewards for patrolling high-crime areas and negative rewards (penalties) for stepping out of bounds or into invalid zones (e.g., water bodies).
5.  **Training**: The Q-learning agent explores the environment, updating its Q-table based on the Bellman equation. Over time, it learns optimal paths to maximize its cumulative reward.

## 📄 License

This project is licensed under the MIT License.
