# frontend/app.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import time

from backend.data_processing import process_pipeline
from backend.grid import CrimeGrid
from backend.environment import CrimeEnv
from backend.q_learning import QLearningAgent


# =========================
# LOAD ENVIRONMENT
# =========================

@st.cache_data
def load_environment():
    df = process_pipeline("backend/data/crimes.csv")

    grid_system = CrimeGrid(df, grid_size=20)
    grid = grid_system.populate_grid()
    grid = grid_system.normalize_grid()

    level_grid = grid_system.get_crime_levels()

    env = CrimeEnv(level_grid)

    return env, level_grid, df


# =========================
# SESSION STATE
# =========================

if "env" not in st.session_state:
    st.session_state.env, st.session_state.level_grid, st.session_state.df = load_environment()

if "agent_pos" not in st.session_state:
    st.session_state.agent_pos = st.session_state.env.reset()

if "last_reward" not in st.session_state:
    st.session_state.last_reward = 0

if "total_reward" not in st.session_state:
    st.session_state.total_reward = 0

if "reward_history" not in st.session_state:
    st.session_state.reward_history = []

if "current_level" not in st.session_state:
    st.session_state.current_level = 0

if "agent" not in st.session_state:
    st.session_state.agent = QLearningAgent(grid_size=20)

if "ai_running" not in st.session_state:
    st.session_state.ai_running = False


env = st.session_state.env
level_grid = st.session_state.level_grid
df = st.session_state.df
agent = st.session_state.agent


# =========================
# UI TITLE
# =========================

st.title("🚓 Smart Crime Patrol System")
st.markdown("### 🔥 AI + Real-Time Crime Heatmap")


# =========================
# GRID FUNCTION
# =========================

def plot_grid(level_grid, agent_pos):
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(level_grid, cmap='RdYlGn_r')
    ax.scatter(agent_pos[1], agent_pos[0], s=120, label="Agent")

    ax.set_title("Grid View")
    ax.legend()

    return fig


# =========================
# REAL MAP (ADVANCED)
# =========================

def show_advanced_map(df, level_grid, agent_pos, grid_size=20):
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()

    m = folium.Map(location=[41.85, -87.65], zoom_start=10)

    # Heatmap data
    heat_data = []

    for i in range(grid_size):
        for j in range(grid_size):
            intensity = level_grid[i][j]

            if intensity > 0:
                lat = lat_min + (i / grid_size) * (lat_max - lat_min)
                lon = lon_min + (j / grid_size) * (lon_max - lon_min)

                heat_data.append([lat, lon, intensity])

    HeatMap(
        heat_data,
        radius=18,
        blur=25,
        min_opacity=0.25
    ).add_to(m)

    # Agent position
    i, j = agent_pos

    lat = lat_min + (i / grid_size) * (lat_max - lat_min)
    lon = lon_min + (j / grid_size) * (lon_max - lon_min)

    folium.Marker(
        location=[lat, lon],
        popup=f"🚓 Agent: {agent_pos}",
        icon=folium.Icon(color="blue")
    ).add_to(m)

    return m


# =========================
# PERFORMANCE DASHBOARD
# =========================

st.subheader("📊 Performance Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.metric("💰 Total Reward", round(st.session_state.total_reward, 2))

with col2:
    st.metric("📍 Position", str(st.session_state.agent_pos))


fig2, ax2 = plt.subplots()
ax2.plot(st.session_state.reward_history)
ax2.set_title("Reward Over Time")
st.pyplot(fig2)


# =========================
# SIDE-BY-SIDE VISUALIZATION
# =========================

col1, col2 = st.columns(2)

with col1:
    st.pyplot(plot_grid(level_grid, st.session_state.agent_pos))

with col2:
    st_folium(
        show_advanced_map(df, level_grid, st.session_state.agent_pos),
        width=500,
        height=500,
        key="map"
    )


# =========================
# MOVE FUNCTION
# =========================

def move_agent(action):
    next_pos, reward, _ = env.step(action)

    st.session_state.agent_pos = next_pos
    st.session_state.last_reward = reward
    st.session_state.total_reward += reward
    st.session_state.reward_history.append(st.session_state.total_reward)

    i, j = next_pos
    st.session_state.current_level = level_grid[i][j]


# =========================
# CONTROLS
# =========================

st.subheader("🎮 Manual Control")

col1, col2, col3 = st.columns(3)

with col2:
    if st.button("⬆️"):
        move_agent(0)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⬅️"):
        move_agent(2)

with col2:
    if st.button("⬇️"):
        move_agent(1)

with col3:
    if st.button("➡️"):
        move_agent(3)


# =========================
# AI PATROL
# =========================

st.subheader("🤖 AI Patrol")

if st.button("🚀 Start AI Patrol"):
    for _ in range(30):
        state = st.session_state.agent_pos
        action = agent.get_best_action(state)

        move_agent(action)
        time.sleep(0.15)

    st.rerun()


# =========================
# RESET
# =========================

if st.button("🔄 Reset"):
    st.session_state.agent_pos = env.reset()
    st.session_state.total_reward = 0
    st.session_state.reward_history = []
    st.rerun()