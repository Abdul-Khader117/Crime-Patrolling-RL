import React, { useEffect, useState, useCallback, useMemo } from "react";
import axios from "axios";
import {
  MapContainer,
  TileLayer,
  Rectangle,
  Marker,
  Polyline,
  useMap
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { 
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid 
} from 'recharts';
import { 
  Zap, Brain, RotateCcw, Save, FolderOpen, ShieldCheck, 
  ChevronUp, ChevronDown, ChevronLeft, ChevronRight, Activity
} from 'lucide-react';
import "./App.css";

const API = "http://127.0.0.1:5000";

// 🚓 Robust SVG Agent Icon
const policeCarSvg = `
<svg width="45" height="45" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M3 11L5.5 5.5H18.5L21 11H3Z" fill="#38bdf8" stroke="#f8fafc" stroke-width="1.5" stroke-linejoin="round"/>
  <path d="M2.5 11H21.5V17H2.5V11Z" fill="#0ea5e9" stroke="#f8fafc" stroke-width="1.5" stroke-linejoin="round"/>
  <circle cx="6" cy="17" r="2.5" fill="#1e293b" stroke="#f8fafc" stroke-width="1.5"/>
  <circle cx="18" cy="17" r="2.5" fill="#1e293b" stroke="#f8fafc" stroke-width="1.5"/>
  <path d="M11 5.5V3.5H13V5.5" stroke="#ef4444" stroke-width="1.5" stroke-linecap="round"/>
</svg>
`;

const agentIcon = L.divIcon({
  html: policeCarSvg,
  className: "agent-marker",
  iconSize: [45, 45],
  iconAnchor: [22, 22],
});

function CenterMap({ pos }) {
  const map = useMap();
  useEffect(() => {
    if (pos && pos.lat && pos.lon) {
        // Use setView with animate for more reliable tracking than panTo
        map.setView([pos.lat, pos.lon], map.getZoom(), { animate: true });
    }
  }, [pos, map]);
  return null;
}

const LegendItem = ({ color, text, opacity = 0.6 }) => (
    <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px" }}>
        <div style={{ width: "12px", height: "12px", background: color, borderRadius: "3px", opacity }} />
        <span style={{ color: "#94a3b8", fontSize: "11px", fontWeight: "600" }}>{text}</span>
    </div>
);

function App() {
  const [agent, setAgent] = useState({ lat: 41.8781, lon: -87.6298 });
  const [gridPoints, setGridPoints] = useState([]);
  const [totalReward, setTotalReward] = useState(0);
  const [lastReward, setLastReward] = useState(0);
  const [trail, setTrail] = useState([]);
  const [loading, setLoading] = useState(false);
  const [gridSize, setGridSize] = useState(20);
  const [epsilon, setEpsilon] = useState(1.0);
  const [history, setHistory] = useState([]);
  const [status, setStatus] = useState("Idle");
  const [steps, setSteps] = useState({ lat: 0.018, lon: 0.021 });

  const fetchState = useCallback(async () => {
    try {
      const res = await axios.get(`${API}/state`);
      const { 
        lat, lon, total_reward, grid_points, grid_size, epsilon, history, lat_step, lon_step 
      } = res.data;
      
      const newPos = { lat, lon };
      setAgent(newPos);
      setGridPoints(grid_points);
      setTotalReward(total_reward);
      setGridSize(grid_size);
      setEpsilon(epsilon);
      if (lat_step) setSteps({ lat: lat_step, lon: lon_step });
      
      if (history && history.length > 0) {
          // Format history for Recharts
          const chartData = history.map((r, i) => ({ ep: i, reward: r }));
          setHistory(chartData);
      }
      
      setTrail(prev => {
        if (prev.length > 0) {
            const last = prev[prev.length - 1];
            if (last[0] === lat && last[1] === lon) return prev;
        }
        return [...prev, [lat, lon]].slice(-50);
      });
    } catch (err) {
      console.error("API Error:", err);
      setTimeout(() => fetchState(), 2000);
    }
  }, []);

  useEffect(() => {
    fetchState();
  }, [fetchState]);

  const updateAgentState = (data) => {
    const newPos = { lat: data.lat, lon: data.lon };
    setAgent(newPos);
    setTotalReward(data.total_reward);
    setLastReward(data.reward);
    setTrail(prev => [...prev, [newPos.lat, newPos.lon]].slice(-50));
    if (data.done) setStatus("Episode Finished");
  };

  const move = async (action) => {
    try {
        setStatus("Patrolling...");
        const res = await axios.post(`${API}/move`, { action });
        updateAgentState(res.data);
    } catch (err) { console.error(err); }
  };

  const aiMove = async () => {
    try {
        setStatus("AI Thinking...");
        const res = await axios.post(`${API}/ai`);
        updateAgentState(res.data);
    } catch (err) { console.error(err); }
  };

  const train = async () => {
    setLoading(true);
    setStatus("Deep Training RL...");
    try {
        const res = await axios.post(`${API}/train`, { episodes: 1000 });
        setEpsilon(res.data.epsilon);
        fetchState();
        setStatus("Training Complete");
    } catch (err) { setStatus("Training Failed"); }
    setLoading(false);
  };

  const reset = async () => {
    setStatus("Resetting...");
    const res = await axios.post(`${API}/reset`);
    setTrail([]);
    updateAgentState({ ...res.data, reward: 0 });
    setLastReward(0);
    setStatus("Ready");
  };

  // Precise cell bounds calculation
  const getCellBounds = (p) => {
    // Dynamically calculate gaps between points to form a perfect grid
    const { lat: latStep, lon: lonStep } = steps;
    
    // Expand bounds slightly to ensure seamless heatmap coverage
    return [
        [p.lat - (latStep * 0.505), p.lon - (lonStep * 0.505)], 
        [p.lat + (latStep * 0.505), p.lon + (lonStep * 0.505)]
    ];
  };

  const riskColors = {
      2: "#ef4444", // Critical Risk
      1: "#f59e0b", // High Activity
      0: "#10b981", // Low Activity
      "-1": "transparent"
  };

  return (
    <div style={{ display: "flex", height: "100vh", background: "#020617", color: "#f8fafc", overflow: "hidden" }}>
      {/* Sidebar */}
      <div className="glass-panel" style={{
        width: "380px", background: "rgba(15, 23, 42, 0.98)", padding: "30px", 
        display: "flex", flexDirection: "column", zIndex: 1000, boxShadow: "10px 0 30px rgba(0,0,0,0.5)"
      }}>
        <div style={{ marginBottom: "30px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                <div style={{ background: "#38bdf8", padding: "8px", borderRadius: "10px" }}><ShieldCheck size={24} color="#020617" /></div>
                <div>
                    <h1 style={{ fontSize: "22px", fontWeight: "800", letterSpacing: "-0.5px", margin: "0" }}>CRIME PATROL <span style={{ color: "#38bdf8" }}>RL</span></h1>
                    <div style={{ fontSize: "10px", color: "#64748b", textTransform: "uppercase", letterSpacing: "1px", fontWeight: "700" }}>Chicago Tactical Agent</div>
                </div>
            </div>
        </div>

        {/* Stats Grid */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "25px" }}>
            <div className="stat-card">
                <div style={{ fontSize: "10px", color: "#94a3b8", fontWeight: "700", marginBottom: "5px" }}>TOTAL REWARD</div>
                <div style={{ fontSize: "20px", fontWeight: "800", color: totalReward >= 0 ? "#4ade80" : "#fb7185" }}>{totalReward.toFixed(1)}</div>
            </div>
            <div className="stat-card">
                <div style={{ fontSize: "10px", color: "#94a3b8", fontWeight: "700", marginBottom: "5px" }}>EPSILON (ε)</div>
                <div style={{ fontSize: "20px", fontWeight: "800", color: "#38bdf8" }}>{epsilon.toFixed(3)}</div>
            </div>
        </div>

        <div style={{ flex: 1, overflowY: "auto", paddingRight: "5px" }}>
            {/* Controls */}
            <div style={{ marginBottom: "25px" }}>
                <label style={{ fontSize: "12px", color: "#94a3b8", fontWeight: "800", display: "block", marginBottom: "15px" }}>MANUAL DISPATCH</label>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "10px", width: "180px", margin: "0 auto" }}>
                    <div /><button className="ctrl-btn" onClick={() => move(0)}><ChevronUp size={20} /></button><div />
                    <button className="ctrl-btn" onClick={() => move(2)}><ChevronLeft size={20} /></button>
                    <button className="ctrl-btn" onClick={() => move(1)}><ChevronDown size={20} /></button>
                    <button className="ctrl-btn" onClick={() => move(3)}><ChevronRight size={20} /></button>
                </div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginBottom: "30px" }}>
                <button className="action-btn" onClick={aiMove} style={{ background: "#0ea5e9", color: "white" }}>
                    <Zap size={18} fill="currentColor" /> AI Patrol Step
                </button>
                <button className="action-btn" onClick={train} disabled={loading} style={{ background: loading ? "#334155" : "#f59e0b", color: "white" }}>
                    {loading ? <Activity className="animate-spin" size={18} /> : <Brain size={18} />} 
                    {loading ? "Optimizing Policy..." : "Train Agent (1000 Episodes)"}
                </button>
                <div style={{ display: "flex", gap: "10px" }}>
                    <button className="action-btn" onClick={async () => { await axios.post(`${API}/save`); setStatus("Saved"); }} style={{ flex: 1, background: "#059669", color: "white", fontSize: "12px" }}>
                        <Save size={14} /> Save
                    </button>
                    <button className="action-btn" onClick={async () => { await axios.post(`${API}/load`); fetchState(); setStatus("Loaded"); }} style={{ flex: 1, background: "#4f46e5", color: "white", fontSize: "12px" }}>
                        <FolderOpen size={14} /> Load
                    </button>
                </div>
            </div>

            {/* Training Chart */}
            <div style={{ marginTop: "20px" }}>
                <label style={{ fontSize: "12px", color: "#94a3b8", fontWeight: "800", display: "block", marginBottom: "15px" }}>CONVERGENCE HISTORY</label>
                <div style={{ height: "150px", width: "100%", background: "rgba(0,0,0,0.2)", borderRadius: "12px", padding: "10px" }}>
                    {history.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={history.slice(-50)}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="ep" hide />
                                <YAxis hide />
                                <Tooltip 
                                    contentStyle={{ background: "#1e293b", border: "none", borderRadius: "8px", fontSize: "12px" }}
                                    itemStyle={{ color: "#38bdf8" }}
                                />
                                <Line type="monotone" dataKey="reward" stroke="#38bdf8" strokeWidth={2} dot={false} isAnimationActive={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    ) : (
                        <div style={{ height: "100%", display: "flex", alignItems: "center", justifyCenter: "center", fontSize: "11px", color: "#475569" }}>No training data available</div>
                    )}
                </div>
            </div>
        </div>

        <div style={{ paddingTop: "20px", borderTop: "1px solid rgba(255,255,255,0.05)", marginTop: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "11px", color: "#64748b", fontWeight: "700" }}>STATUS</span>
                <span style={{ fontSize: "11px", color: "#38bdf8", fontWeight: "800" }}>{status.toUpperCase()}</span>
            </div>
            <button onClick={reset} style={{ width: "100%", background: "transparent", border: "1px solid #ef4444", color: "#ef4444", borderRadius: "8px", padding: "10px", fontSize: "11px", fontWeight: "800", cursor: "pointer", marginTop: "15px" }}>
                <RotateCcw size={12} style={{ marginRight: "5px" }} /> RE-INITIALIZE ENVIRONMENT
            </button>
        </div>
      </div>

      {/* Map View */}
      <div style={{ flex: 1, position: "relative" }}>
        <MapContainer center={[41.8781, -87.6298]} zoom={11} zoomControl={false} style={{ height: "100%", width: "100%", filter: "grayscale(0.2) contrast(1.1)" }}>
          <TileLayer 
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" 
            attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          />
          
          {/* Grid Visualization */}
          {gridPoints.map((p, idx) => (
            <Rectangle
              key={idx}
              bounds={getCellBounds(p)}
              pathOptions={{
                fillColor: riskColors[p.level] || "#1e293b",
                fillOpacity: p.valid ? (p.level === 2 ? 0.5 : p.level === 1 ? 0.3 : 0.15) : 0.05,
                weight: p.valid ? 1 : 0,
                color: "rgba(255,255,255,0.05)",
                interactive: false
              }}
            />
          ))}

          {/* Path Trail */}
          <Polyline 
            positions={trail} 
            pathOptions={{ 
                color: "#38bdf8", 
                weight: 4, 
                opacity: 0.6, 
                dashArray: '1, 12',
                lineCap: 'round'
            }} 
          />
          
          {/* Agent Marker */}
          <Marker position={[agent.lat, agent.lon]} icon={agentIcon} />
          
          <CenterMap pos={agent} />
        </MapContainer>

        {/* Legend Overlay */}
        <div className="glass-panel" style={{ 
            position: "absolute", bottom: "30px", right: "30px", 
            background: "rgba(15, 23, 42, 0.9)", padding: "20px", 
            borderRadius: "16px", color: "white", zIndex: 1000 
        }}>
          <div style={{ fontWeight: "800", fontSize: "12px", marginBottom: "15px", color: "#94a3b8", letterSpacing: "1px" }}>RISK ASSESSMENT</div>
          <LegendItem color="#ef4444" text="Critical Risk" opacity={0.6} />
          <LegendItem color="#f59e0b" text="High Activity" opacity={0.4} />
          <LegendItem color="#10b981" text="Monitored Area" opacity={0.2} />
          <LegendItem color="#0ea5e9" text="Water/No Access" opacity={0.1} />
        </div>
        
        {/* Floating Reward Notification */}
        {lastReward !== 0 && (
            <div style={{
                position: "absolute", top: "30px", right: "30px",
                background: lastReward > 0 ? "rgba(16, 185, 129, 0.9)" : "rgba(239, 68, 68, 0.9)",
                padding: "10px 20px", borderRadius: "100px", zIndex: 1000,
                fontWeight: "800", fontSize: "14px", boxShadow: "0 10px 25px rgba(0,0,0,0.3)",
                animation: "fadeOut 1s forwards"
            }}>
                {lastReward > 0 ? "+" : ""}{lastReward.toFixed(1)} Reward
            </div>
        )}
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap');
        @keyframes fadeOut {
            0% { opacity: 1; transform: translateY(0); }
            100% { opacity: 0; transform: translateY(-20px); }
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

export default App;