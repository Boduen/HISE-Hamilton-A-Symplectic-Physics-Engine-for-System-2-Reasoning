import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


# Page Configuration (Removed Icon)
st.set_page_config(
    page_title="HISE-Pro Cognitive Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Style Settings ---
plt.style.use('dark_background')


# --- Mock Data Generator ---
def mock_inference_data(steps=100, mode="mixed"):
    """
    Simulates HISE physics engine telemetry.
    """
    t = np.arange(steps)
    
    if mode == "sys1":
        entropy = np.random.normal(0.2, 0.05, steps)
        mass = np.random.normal(0.1, 0.01, steps)
        epsilon = np.ones(steps) * 0.1
        fsi = np.random.normal(2.5, 0.2, steps)
        
    elif mode == "sys2":
        entropy = np.random.normal(2.5, 0.5, steps)
        mass = np.random.normal(2.0, 0.5, steps)
        epsilon = np.ones(steps) * 0.01
        fsi = np.random.normal(0.6, 0.1, steps)
        
    else: 
        entropy = np.concatenate([
            np.random.normal(0.2, 0.1, steps // 2), 
            np.random.normal(2.5, 0.3, steps // 2)
        ])
        mass = np.sqrt(entropy * 1.5) + np.random.normal(0, 0.1, steps)
        fsi = 1.0 / (mass + 1e-6)
        epsilon = 0.1 / (mass + 1.0)


    q = np.cumsum(np.sin(t * 0.1) * epsilon) 
    p = np.cos(t * 0.1) * mass
    
    return pd.DataFrame({
        "Step": t,
        "Entropy (H)": entropy,
        "Semantic Mass (M)": mass,
        "Step Size (epsilon)": epsilon,
        "FSI Score": fsi,
        "Position (q)": q,
        "Momentum (p)": p
    })


# --- Main UI ---
st.title("HISE-Pro: Holographic Inertial Syntax Engine")
st.markdown("### Real-time Cognitive Thermodynamics Monitor")


# Sidebar Controls
st.sidebar.header("Physics Engine Controls")
sim_mode = st.sidebar.selectbox(
    "Simulation Scenario", 
    ["Mixed (System 1->2)", "System 1 (Reflex)", "System 2 (Deep Thought)"]
)
steps = st.sidebar.slider("Generation Steps", 50, 500, 100)
run_btn = st.sidebar.button("Run Inference Simulation")


if run_btn:
    mode_key = "mixed" if "Mixed" in sim_mode else ("sys1" if "Reflex" in sim_mode else "sys2")
    data = mock_inference_data(steps, mode=mode_key)
    
    # --- Top Level KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    avg_mass = data["Semantic Mass (M)"].mean()
    min_fsi = data["FSI Score"].min()
    sys2_active_count = (data['Semantic Mass (M)'] > 1.0).sum()
    
    with col1:
        st.metric(
            "Avg Semantic Mass", 
            f"{avg_mass:.2f}", 
            delta="Heavy" if avg_mass > 1.0 else "Light", 
            delta_color="inverse"
        )
    with col2:
        st.metric(
            "Min FSI Score", 
            f"{min_fsi:.2f}", 
            delta="Risk" if min_fsi < 1.0 else "Safe", 
            delta_color="normal"
        )
    with col3:
        st.metric(
            "System 2 Activation", 
            f"{sys2_active_count / steps * 100:.0f}%"
        )
    with col4:
        is_cooling = data["Step Size (epsilon)"].iloc[-1] < 0.05
        st.metric(
            "Thermodynamic Status", 
            "Cooling (Converging)" if is_cooling else "Ballistic (Flowing)"
        )


    # --- Main Charts ---
    st.markdown("---")
    
    # Row 1: Inertia & Safety
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("1. Cognitive Inertia (Mass Dynamics)")
        st.line_chart(data, x="Step", y=["Semantic Mass (M)", "Step Size (epsilon)"])
        st.caption(
            "Observation: As Semantic Mass (M) increases, the Symplectic Step Size (epsilon) "
            "automatically decreases. This represents the transition to System 2 deliberative thought."
        )
        
    with c2:
        st.subheader("2. Axiom Smuggling Detector (FSI)")
        
        fig_fsi, ax_fsi = plt.subplots(figsize=(10, 4))
        ax_fsi.plot(data["Step"], data["FSI Score"], color='#00ff00', label='FSI Metric')
        ax_fsi.axhline(y=1.0, color='red', linestyle='--', label='Nyquist Limit (Hallucination)')
        
        ax_fsi.fill_between(
            data["Step"], 0, 1.0, 
            alpha=0.2, color='red', 
            where=(data["FSI Score"] < 1.0),
            label='Smuggling Zone'
        )
        
        ax_fsi.legend(loc='upper right')
        ax_fsi.set_ylabel("Fisher Semantic Information")
        ax_fsi.set_xlabel("Token Step")
        
        ax_fsi.set_facecolor('#0e1117')
        fig_fsi.patch.set_facecolor('#0e1117')
        ax_fsi.tick_params(axis='x', colors='white')
        ax_fsi.tick_params(axis='y', colors='white')
        ax_fsi.spines['bottom'].set_color('white')
        ax_fsi.spines['left'].set_color('white')
        
        st.pyplot(fig_fsi)
        st.caption(
            "Red Zone indicates 'Axiom Smuggling'. If the trajectory enters this region, "
            "the RAG Safety Valve is triggered immediately."
        )


    # Row 2: Phase Space Topology
    st.markdown("---")
    st.subheader("3. Semantic Phase Space (Hamiltonian Orbit)")
    
    col_phase, col_desc = st.columns([2, 1])
    
    with col_phase:
        fig_phase, ax_phase = plt.subplots(figsize=(8, 6))
        
        sns.scatterplot(
            data=data, 
            x="Position (q)", 
            y="Momentum (p)", 
            hue="Semantic Mass (M)", 
            palette="rocket_r", 
            ax=ax_phase
        )
        ax_phase.plot(data["Position (q)"], data["Momentum (p)"], color='white', alpha=0.3)
        
        ax_phase.set_title("q-p Trajectory Evolution")
        ax_phase.set_xlabel("Semantic Position (q)")
        ax_phase.set_ylabel("Semantic Momentum (p)")
        
        ax_phase.set_facecolor('#0e1117')
        fig_phase.patch.set_facecolor('#0e1117')
        ax_phase.tick_params(colors='white')
        ax_phase.spines['bottom'].set_color('white')
        ax_phase.spines['left'].set_color('white')
        
        st.pyplot(fig_phase)
        
    with col_desc:
        st.markdown("""
        **Physics Interpretation:**
        
        * **Spiral Sink**: Indicates System 2 is applying "Thermodynamic Friction" to force logical convergence.
        * **Limit Cycle**: Indicates System 1 is in a stable, reflexive generation loop (Grammar flow).
        * **Divergence**: If the trajectory escapes the bounded region, it indicates gradient explosion or physical parameter mismatch.
        """)


else:
    st.info("Awaiting input... Click 'Run Inference Simulation' on the sidebar to visualize HISE-Pro dynamics.")