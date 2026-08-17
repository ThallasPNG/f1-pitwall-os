import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fastf1.plotting
import re

from optimization import (solve_dp, optimize_all_strategies, get_pit_loss_for_lap, 
                          calculate_stint_time, get_base_lap_time, run_monte_carlo, evaluate_fixed_strategy)
from telemetry import (load_fastf1_data, fit_degradation_models, get_head_to_head_laps,
                       get_actual_strategy, modele_lineaire, modele_quadratique, modele_exponentiel)
from report import generate_pdf_report

sns.set_theme(style="whitegrid")
st.set_page_config(page_title="F1 Pit Wall OS", page_icon="🏁", layout="wide")

# State init
if 'custom_strats' not in st.session_state:
    st.session_state.custom_strats = [('Medium', 'Hard'), ('Soft', 'Hard'), ('Soft', 'Medium', 'Soft')]
if 'total_laps' not in st.session_state:
    st.session_state.total_laps = 52
if 'pit_loss_time' not in st.session_state:
    st.session_state.pit_loss_time = 28.0

# Database circuits (Translated to English)
CIRCUITS_DATA = {
    "Bahrain (Sakhir)": {"laps": 57, "pit_loss": 24.0, "length": 5.412, "abrasion": "High", "record": "1:31.447"},
    "Saudi Arabia (Jeddah)": {"laps": 50, "pit_loss": 22.0, "length": 6.174, "abrasion": "Medium", "record": "1:30.734"},
    "Australia (Melbourne)": {"laps": 58, "pit_loss": 20.0, "length": 5.278, "abrasion": "Medium", "record": "1:19.815"},
    "Japan (Suzuka)": {"laps": 53, "pit_loss": 23.0, "length": 5.807, "abrasion": "Very High", "record": "1:30.983"},
    "China (Shanghai)": {"laps": 56, "pit_loss": 24.0, "length": 5.451, "abrasion": "High", "record": "1:32.238"},
    "USA (Miami)": {"laps": 57, "pit_loss": 22.0, "length": 5.412, "abrasion": "Medium", "record": "1:29.708"},
    "Emilia-Romagna (Imola)": {"laps": 63, "pit_loss": 28.0, "length": 4.909, "abrasion": "Medium", "record": "1:15.484"},
    "Monaco (Monte-Carlo)": {"laps": 78, "pit_loss": 25.0, "length": 3.337, "abrasion": "Very Low", "record": "1:12.909"},
    "Canada (Montreal)": {"laps": 70, "pit_loss": 18.0, "length": 4.361, "abrasion": "Low", "record": "1:13.078"},
    "Spain (Barcelona)": {"laps": 66, "pit_loss": 23.0, "length": 4.657, "abrasion": "High", "record": "1:16.330"},
    "Austria (Spielberg)": {"laps": 71, "pit_loss": 20.0, "length": 4.318, "abrasion": "Medium", "record": "1:05.619"},
    "Great Britain (Silverstone)": {"laps": 52, "pit_loss": 28.0, "length": 5.891, "abrasion": "High", "record": "1:27.097"},
    "Hungary (Hungaroring)": {"laps": 70, "pit_loss": 20.0, "length": 4.381, "abrasion": "Medium", "record": "1:16.627"},
    "Belgium (Spa-Francorchamps)": {"laps": 44, "pit_loss": 24.0, "length": 7.004, "abrasion": "High", "record": "1:46.286"},
    "Netherlands (Zandvoort)": {"laps": 72, "pit_loss": 18.0, "length": 4.259, "abrasion": "High", "record": "1:11.097"},
    "Italy (Monza)": {"laps": 53, "pit_loss": 24.0, "length": 5.793, "abrasion": "Low", "record": "1:21.046"},
    "Azerbaijan (Baku)": {"laps": 51, "pit_loss": 21.0, "length": 6.003, "abrasion": "Low", "record": "1:43.009"},
    "Singapore (Marina Bay)": {"laps": 62, "pit_loss": 29.0, "length": 4.940, "abrasion": "Medium", "record": "1:35.867"},
    "USA (Austin)": {"laps": 56, "pit_loss": 20.0, "length": 5.513, "abrasion": "High", "record": "1:36.169"},
    "Mexico (Mexico City)": {"laps": 71, "pit_loss": 22.0, "length": 4.304, "abrasion": "Low", "record": "1:17.774"},
    "Brazil (Interlagos)": {"laps": 71, "pit_loss": 24.0, "length": 4.309, "abrasion": "Medium", "record": "1:10.540"},
    "USA (Las Vegas)": {"laps": 50, "pit_loss": 20.0, "length": 6.201, "abrasion": "Low", "record": "1:35.490"},
    "Qatar (Lusail)": {"laps": 57, "pit_loss": 25.0, "length": 5.419, "abrasion": "Very High", "record": "1:24.319"},
    "Abu Dhabi (Yas Marina)": {"laps": 58, "pit_loss": 23.0, "length": 5.281, "abrasion": "Medium", "record": "1:26.103"}
}

def apply_to_sidebar(comp, base, a, b2):
    prefixes = {"SOFT": "s", "MEDIUM": "m", "HARD": "h"}
    p = prefixes[comp]
    st.session_state[f"{p}b"] = float(base)
    st.session_state[f"{p}a"] = float(a)
    st.session_state[f"{p}b2"] = float(b2)

# --- UI Sidebar ---
st.sidebar.title("🏁 Pit Wall OS")
st.sidebar.markdown("---")

page = st.sidebar.radio("🧭 NAVIGATION", [
    "🏠 Mission Control",
    "🛠️ Strategy Builder",
    "🎲 Monte Carlo & Risk",
    "📈 Telemetry Modeling",
    "🥊 Validation & H2H",
    "🔄 Track History",
    "🌍 Track Database"
])
st.sidebar.markdown("---")

st.sidebar.header("⚙️ GP Parameters")
selected_circuit = st.sidebar.selectbox("Select Race:", list(CIRCUITS_DATA.keys()))
TOTAL_LAPS = CIRCUITS_DATA[selected_circuit]["laps"]
PIT_LOSS_TIME = CIRCUITS_DATA[selected_circuit]["pit_loss"]

FUEL_EFFECT = st.sidebar.slider("Fuel Effect (s/lap gain)", min_value=0.0, max_value=0.15, value=0.06, step=0.01)

with st.sidebar.expander("🚓 Safety Car (SC)", expanded=False):
    sc_active = st.checkbox("Enable SC event", value=False)
    if sc_active:
        sc_start = st.slider("Deployment Lap", 1, TOTAL_LAPS, 20)
        sc_duration = st.slider("SC Duration (laps)", 1, 10, 3)
        sc_pit_loss = st.slider("SC Pit Loss (s)", 10.0, PIT_LOSS_TIME, 15.0, 0.5)
        sc_slowdown = st.slider("Field Slowdown (s/lap)", 10.0, 50.0, 30.0, 1.0)
        sc_deg_factor = st.slider("SC Wear Factor", 0.0, 1.0, 0.25, 0.05)
    else:
        sc_start, sc_duration, sc_pit_loss, sc_slowdown, sc_deg_factor = -1, 0, PIT_LOSS_TIME, 0.0, 1.0

sc_config = {
    'active': sc_active, 'start': sc_start, 'duration': sc_duration, 
    'pit_loss': sc_pit_loss, 'slowdown': sc_slowdown, 'deg_factor': sc_deg_factor
}

st.sidebar.markdown("---")
st.sidebar.header("🏎️ Tyre Properties")
deg_model = st.sidebar.selectbox("Degradation Model", ["Quadratique", "Exponentiel", "Linéaire"])

with st.sidebar.expander("🔴 Soft Tyres", expanded=False):
    s_b = st.number_input("Base time (s) - Soft", value=93.8, key='sb', step=0.1)
    s_a = st.number_input("Parameter a - Soft", value=0.148, format="%.3f", key='sa', step=0.01)
    s_b2 = st.number_input("Parameter b - Soft", value=0.002, format="%.4f", key='sb2', step=0.001)
    s_w = st.number_input("Warm-up (s) - Soft", value=1.0, step=0.5, key='sw')

with st.sidebar.expander("🟡 Medium Tyres", expanded=False):
    m_b = st.number_input("Base time (s) - Med", value=94.13, key='mb', step=0.1)
    m_a = st.number_input("Parameter a - Med", value=0.076, format="%.3f", key='ma', step=0.01)
    m_b2 = st.number_input("Parameter b - Med", value=0.001, format="%.4f", key='mb2', step=0.001)
    m_w = st.number_input("Warm-up (s) - Med", value=2.0, step=0.5, key='mw')

with st.sidebar.expander("⚪ Hard Tyres", expanded=False):
    h_b = st.number_input("Base time (s) - Hard", value=94.8, key='hb', step=0.1)
    h_a = st.number_input("Parameter a - Hard", value=0.056, format="%.3f", key='ha', step=0.01)
    h_b2 = st.number_input("Parameter b - Hard", value=0.000, format="%.4f", key='hb2', step=0.001)
    h_w = st.number_input("Warm-up (s) - Hard", value=3.5, step=0.5, key='hw')

TIRE_MODELS = {
    'Soft':   {'base': s_b, 'a': s_a, 'b': s_b2, 'warmup': s_w},
    'Medium': {'base': m_b, 'a': m_a, 'b': m_b2, 'warmup': m_w},
    'Hard':   {'base': h_b, 'a': h_a, 'b': h_b2, 'warmup': h_w}
}

def format_time(seconds): 
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {seconds % 60:06.3f}s"

def get_full_race_laps(strategy, custom_total_laps=TOTAL_LAPS):
    lap_times = []
    compounds = strategy['compounds']
    pit_laps = strategy['pit_laps']
    current_lap = 1
    
    for i, pit_lap in enumerate(list(pit_laps) + [custom_total_laps]):
        stint_laps = pit_lap - current_lap + 1
        if stint_laps <= 0: continue
        
        stint_times = []
        current_age = 1.0
        params = TIRE_MODELS.get(compounds[i], TIRE_MODELS['Medium'])
        
        for j in range(stint_laps):
            curr_race_lap = current_lap + j
            is_sc = sc_config['active'] and (sc_config['start'] <= curr_race_lap < sc_config['start'] + sc_config['duration'])
            
            base_time = get_base_lap_time(params['base'], params['a'], params['b'], current_age, deg_model)
            base_time -= (curr_race_lap * FUEL_EFFECT) 
            
            if current_lap > 1 and j == 0: 
                base_time += params['warmup'] 
            
            if is_sc:
                stint_times.append(base_time + sc_config['slowdown'])
                current_age += sc_config['deg_factor']
            else:
                stint_times.append(base_time)
                current_age += 1.0
                
        if i < len(pit_laps): 
            stint_times[-1] += get_pit_loss_for_lap(pit_lap, sc_config, PIT_LOSS_TIME)
            
        lap_times.extend(stint_times)
        current_lap = pit_lap + 1
    return np.array(lap_times)


# --- ROUTING ---

if page == "🏠 Mission Control":
    st.title(f"🏠 Mission Control - {selected_circuit}")
    st.markdown("Automated strategic overview generated by the Dynamic Programming Engine.")
    
    if sc_active:
        st.warning(f"🚓 Safety Car Alert: Active from lap {sc_start} to {sc_start + sc_duration - 1}.")
    
    with st.spinner("Calculating DP optimum..."):
        strats = []
        for stops in [1, 2, 3]:
            s = optimize_all_strategies(TOTAL_LAPS, stops, TIRE_MODELS, sc_config, PIT_LOSS_TIME, deg_model, FUEL_EFFECT)
            if s: strats.append(s)
        strats.sort(key=lambda x: x['total_time'])
        best = strats[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distance", f"{TOTAL_LAPS} Laps")
    c2.metric("Optimal Strategy", " ➔ ".join(best['compounds']))
    pits_clean = ", ".join(map(str, best['pit_laps']))
    c3.metric("Pit Windows", f"Laps {pits_clean}")
    c4.metric("Est. Race Time", format_time(best['total_time']))
    
    st.markdown("---")
    st.subheader("📈 Pace Projection")
    
    fig_mc, ax_mc = plt.subplots(figsize=(14, 6))
    laps_mc = get_full_race_laps(best)
    
    ax_mc.plot(np.arange(1, len(laps_mc)+1), laps_mc, label=f"DP Optimum: {' ➔ '.join(best['compounds'])}", color="#2ca02c", lw=3)
    if sc_active: 
        ax_mc.axvspan(sc_start, sc_start + sc_duration - 1, color='gold', alpha=0.3, label='Safety Car Period')
        
    ax_mc.set_title(f"Perfect Race Pace Simulation - {selected_circuit}", fontsize=14, fontweight='bold', pad=15)
    ax_mc.set_xlabel("Lap Number", fontsize=12)
    ax_mc.set_ylabel("Lap Time (s)", fontsize=12)
    ax_mc.legend(fontsize=11)
    st.pyplot(fig_mc)
    
    st.markdown("---")
    st.subheader("📄 Strategy Report")
    
    pdf_bytes = generate_pdf_report(selected_circuit, best, TOTAL_LAPS, PIT_LOSS_TIME, format_time(best['total_time']), fig_mc)
    st.download_button(
        label="📥 Download Strategy Brief (PDF)",
        data=pdf_bytes,
        file_name=f"Strategy_Brief_{selected_circuit}.pdf",
        mime="application/pdf",
        type="primary"
    )

elif page == "🛠️ Strategy Builder":
    st.title("🛠️ Strategy Builder & Comparator")
    
    with st.expander("➕ Build New Strategy", expanded=True):
        st_stops = st.radio("Number of Stops", [1, 2, 3], horizontal=True)
        st_cols = st.columns(st_stops + 1)
        current_build = [st_cols[i].selectbox(f"Stint {i+1}", ["Soft", "Medium", "Hard"], key=f"sel_{i}") for i in range(st_stops + 1)]
        
        if st.button("Add to Comparison", type="primary"):
            st.session_state.custom_strats.append(tuple(current_build))
            st.rerun()

    st.markdown("---")
    if st.session_state.custom_strats:
        for i, strat in enumerate(st.session_state.custom_strats):
            col_s, col_b = st.columns([4, 1])
            col_s.markdown(f"**Strategy {i+1}:** {len(strat)-1} Stop(s) ➔ {' - '.join(strat)}")
            if col_b.button("❌ Remove", key=f"del_{i}"):
                st.session_state.custom_strats.pop(i)
                st.rerun()
                
        if st.button("🗑️ Clear all"):
            st.session_state.custom_strats = []
            st.rerun()

        st.markdown("---")
        results = []
        for comps in st.session_state.custom_strats:
            t_best, pits = solve_dp(comps, TOTAL_LAPS, TIRE_MODELS, sc_config, PIT_LOSS_TIME, deg_model, FUEL_EFFECT)
            pits_clean = ", ".join(map(str, pits))
            results.append({"label": f"{len(comps)-1} Stop(s): {' ➔ '.join(comps)}", "comps": comps, "time": t_best, "pits_clean": pits_clean, "pits": pits})
            
        results.sort(key=lambda x: x["time"], reverse=True)
        
        st.subheader("Global Evaluation")
        fig1, ax1 = plt.subplots(figsize=(12, max(4, len(results) * 0.8)))
        ax1.barh([r["label"] for r in results], [r["time"] for r in results], color='#1f77b4')
        min_time, max_time = min([r["time"] for r in results]), max([r["time"] for r in results])
        ax1.set_xlim(min_time - 5, max_time + 5)
        ax1.set_xlabel("Total Race Time (s)", fontsize=12)
        st.pyplot(fig1)

        st.subheader("Pace Comparison")
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        colors = sns.color_palette("tab10", len(results))
        
        for i, res in enumerate(results): 
            laps = get_full_race_laps({'compounds': res["comps"], 'pit_laps': res["pits"]})
            ax2.plot(np.arange(1, len(laps)+1), laps, label=f"{res['label']} (Pits: {res['pits_clean']})", color=colors[i], lw=2.5)
            
        if sc_active: 
            ax2.axvspan(sc_start, sc_start + sc_duration - 1, color='gold', alpha=0.3, label='Safety Car')
            
        ax2.set_xlabel("Lap Number", fontsize=12)
        ax2.set_ylabel("Lap Time (s)", fontsize=12)
        ax2.legend(fontsize=11)
        st.pyplot(fig2)
    else:
        st.info("No custom strategies loaded.")

elif page == "🎲 Monte Carlo & Risk":
    st.title("🎲 Monte Carlo Risk Assessment")
    n_sims = st.slider("Simulations (N=)", 100, 5000, 1000)
    
    if st.session_state.custom_strats:
        if st.button("Run Simulation", type="primary"):
            with st.spinner(f"Processing {n_sims} permutations..."):
                fig, ax = plt.subplots(figsize=(14, 7))
                stats = []
                colors = sns.color_palette("tab10", len(st.session_state.custom_strats))
                
                for i, comps in enumerate(st.session_state.custom_strats):
                    t_best, pits = solve_dp(comps, TOTAL_LAPS, TIRE_MODELS, sc_config, PIT_LOSS_TIME, deg_model, FUEL_EFFECT)
                    times = run_monte_carlo(comps, pits, TOTAL_LAPS, TIRE_MODELS, PIT_LOSS_TIME, deg_model, FUEL_EFFECT, n_sims)
                    
                    label_name = f"{len(comps)-1} Stop(s): {' ➔ '.join(comps)}"
                    color = colors[i]
                    
                    ax.hist(times, bins=60, alpha=0.5, color=color, label=label_name, density=False)
                    median_time = np.median(times)
                    ax.axvline(median_time, color=color, linestyle='dashed', linewidth=2)
                    
                    stats.append({
                        "Strategy": label_name, 
                        "DP Ideal Time": format_time(t_best), 
                        "MC Median": format_time(median_time),
                        "P95 (Worst Case)": format_time(np.percentile(times, 95)),
                        "Risk Delta": f"+{(median_time - t_best):.2f} s"
                    })
                    
                ax.set_title("Stochastic Race Time Distribution", fontsize=14, fontweight='bold', pad=15)
                ax.set_xlabel("Total Race Time (s)", fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.legend(fontsize=11)
                st.pyplot(fig)
                
                st.table(pd.DataFrame(stats))
    else:
        st.warning("Configure strategies in the Strategy Builder first.")

elif page == "📈 Telemetry Modeling":
    st.title("📈 ML Curve Fitting (FastF1)")
    
    c1, c2, c3, c4 = st.columns(4)
    y2 = c1.number_input("Year", 2018, 2026, 2023, key='y2')
    gp2 = c2.text_input("GP", "Silverstone", key='gp2')
    d2 = c3.text_input("Driver", "HAM", key='d2_tel')
    comp2 = c4.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"])
    
    sess_type = st.selectbox("Session Type", ["Race (R)", "Sprint (S)", "Practice 2 (FP2)"])
    sess_code = re.search(r'\((.*?)\)', sess_type).group(1)
    
    if st.button("Run Fitting Algorithm", type="primary"):
        with st.spinner("Fetching data & SciPy optim..."):
            session = load_fastf1_data(y2, gp2, sess_code)
            laps_driver = session.laps.pick_driver(d2).pick_compounds(comp2)
            
            if laps_driver.empty:
                st.error("No telemetry data found.")
            else:
                longest_stint = laps_driver['Stint'].value_counts().idxmax()
                stint_laps = laps_driver[laps_driver["Stint"] == longest_stint].pick_quicklaps()
                
                if len(stint_laps) < 3:
                    st.error("Not enough clean laps to converge.")
                else:
                    median = stint_laps['LapTime'].dt.total_seconds().median()
                    correct_laps = stint_laps[(stint_laps['LapTime'].dt.total_seconds() < median + 2) & (stint_laps["IsAccurate"] == 1)]
                    
                    tours_abs = correct_laps['LapNumber'].values
                    tours_rel = tours_abs - tours_abs.min()
                    t_carburant = tours_abs if sess_code in ['R', 'S'] else tours_rel
                    
                    temps_corriges = correct_laps['LapTime'].dt.total_seconds().values + (t_carburant * FUEL_EFFECT)
                    popt_lin, popt_quad, popt_exp = fit_degradation_models(tours_rel, temps_corriges)
                    
                    fig, ax = plt.subplots(figsize=(14, 6))
                    
                    t_lisses_rel = np.linspace(min(tours_rel), max(tours_rel)+15, 300)
                    t_lisses_abs = t_lisses_rel + tours_abs.min()
                    
                    ax.scatter(tours_abs, temps_corriges, color='teal', s=60, edgecolor='white', label='Raw Data (Fuel adjusted)')
                    ax.plot(t_lisses_abs, modele_lineaire(t_lisses_rel, *popt_lin), color='darkorange', linestyle=':', lw=2, label='Linear Fit')
                    ax.plot(t_lisses_abs, modele_quadratique(t_lisses_rel, *popt_quad), color='crimson', linestyle='--', lw=2, label='Quadratic Fit')
                    ax.plot(t_lisses_abs, modele_exponentiel(t_lisses_rel, *popt_exp), color='purple', lw=3, label='Exponential Fit')
                    
                    ax.axvline(max(tours_abs), color="gray", linestyle="--")
                    ax.set_title(f"Degradation curve: {comp2} - {d2} ({gp2} {y2})", fontsize=14, fontweight='bold', pad=15)
                    ax.set_xlabel("Lap Number", fontsize=12)
                    ax.set_ylabel("Lap Time (s)", fontsize=12)
                    ax.legend(fontsize=11)
                    st.pyplot(fig)
                    
                    if deg_model == "Linéaire": base_v, a_v, b_v = popt_lin[0], popt_lin[1], 0.0
                    elif deg_model == "Exponentiel": base_v, a_v, b_v = popt_exp[0], popt_exp[1], popt_exp[2]
                    else: base_v, a_v, b_v = popt_quad[0], popt_quad[1], popt_quad[2]
                    
                    st.button(
                        f"Inject parameters to {comp2} model", 
                        on_click=apply_to_sidebar, 
                        args=(comp2, base_v, a_v, b_v), 
                        type="primary",
                        use_container_width=True
                    )

elif page == "🥊 Validation & H2H":
    st.title("✅ Backtesting Engine & H2H")
    
    c1, c2, c3, c4 = st.columns(4)
    y_val = c1.number_input("Year", 2018, 2026, 2024, key="y_val")
    gp_val = c2.text_input("GP", "Bahrain", key="gp_val")
    d1 = c3.text_input("Driver 1", "VER", key="d1_val")
    d2 = c4.text_input("Driver 2", "PER", key="d2_val")
    
    if st.button("Run Audit", type="primary"):
        with st.spinner("Processing telemetry..."):
            session = load_fastf1_data(y_val, gp_val, "R")
            actual_strat = get_actual_strategy(session, d1)
            
            st.subheader(f"Strategy Audit ({d1})")
            if actual_strat:
                valid_compounds = ["Soft", "Medium", "Hard"]
                if not all(c in valid_compounds for c in actual_strat["compounds"]):
                    st.warning("Non-slick tyres detected. Skipping validation.")
                else:
                    laps_done = actual_strat["total_laps"]
                    t_actual = evaluate_fixed_strategy(actual_strat["compounds"], actual_strat["pit_laps"], laps_done, TIRE_MODELS, sc_config, PIT_LOSS_TIME, deg_model, FUEL_EFFECT)
                    
                    strats_val = []
                    for stops in [1, 2, 3]:
                        s = optimize_all_strategies(laps_done, stops, TIRE_MODELS, sc_config, PIT_LOSS_TIME, deg_model, FUEL_EFFECT)
                        if s: strats_val.append(s)
                    strats_val.sort(key=lambda x: x['total_time'])
                    opt = strats_val[0]
                    
                    act_pits_clean = ", ".join(map(str, actual_strat['pit_laps']))
                    opt_pits_clean = ", ".join(map(str, opt['pit_laps']))
                    
                    col_a, col_b, col_c = st.columns(3)
                    col_a.markdown(f"**Actual ({d1})**<br>Comps: {' ➔ '.join(actual_strat['compounds'])}<br>Pits: L{act_pits_clean}<br>Time: **{format_time(t_actual)}**", unsafe_allow_html=True)
                    col_b.markdown(f"**DP Optimum**<br>Comps: {' ➔ '.join(opt['compounds'])}<br>Pits: L{opt_pits_clean}<br>Time: **{format_time(opt['total_time'])}**", unsafe_allow_html=True)
                    
                    delta = t_actual - opt['total_time']
                    if delta < 0.5:
                        col_c.metric("Delta", f"{delta:.2f} s", delta="Optimal", delta_color="normal")
                    else:
                        col_c.metric("Delta", f"+{delta:.2f} s", delta="Suboptimal", delta_color="inverse")
                    
                    fig_val, ax_val = plt.subplots(figsize=(14, 6))
                    laps_act_plot = get_full_race_laps(actual_strat, custom_total_laps=laps_done)
                    laps_opt_plot = get_full_race_laps(opt, custom_total_laps=laps_done)
                    
                    ax_val.plot(np.arange(1, len(laps_act_plot)+1), laps_act_plot, label=f"Actual Pace ({d1})", color="black", lw=2, linestyle='--')
                    ax_val.plot(np.arange(1, len(laps_opt_plot)+1), laps_opt_plot, label="DP Optimum Pace", color="#2ca02c", lw=3)
                    ax_val.set_title("Actual vs Theoretical Pace", fontsize=14, fontweight='bold', pad=15)
                    ax_val.set_xlabel("Lap Number", fontsize=12)
                    ax_val.set_ylabel("Lap Time (s)", fontsize=12)
                    ax_val.legend(fontsize=11)
                    st.pyplot(fig_val)
            else:
                st.error("Actual strategy data not found.")

            st.markdown("---")
            st.subheader(f"Head-to-Head: {d1} vs {d2}")
            laps_d1, laps_d2 = get_head_to_head_laps(session, d1, d2)
            
            if laps_d1.empty or laps_d2.empty:
                st.error("Missing telemetry for one driver.")
            else:
                med_d1 = laps_d1['LapTime'].dt.total_seconds().median()
                laps_d1_clean = laps_d1[laps_d1['LapTime'].dt.total_seconds() < med_d1 + 5]
                med_d2 = laps_d2['LapTime'].dt.total_seconds().median()
                laps_d2_clean = laps_d2[laps_d2['LapTime'].dt.total_seconds() < med_d2 + 5]
                
                fig_h2h, ax_h2h = plt.subplots(figsize=(14, 6))
                ax_h2h.plot(laps_d1_clean['LapNumber'], laps_d1_clean['LapTime'].dt.total_seconds(), label=d1, color='teal', lw=2, marker='o', markersize=4)
                ax_h2h.plot(laps_d2_clean['LapNumber'], laps_d2_clean['LapTime'].dt.total_seconds(), label=d2, color='darkorange', lw=2, marker='x', markersize=4)
                
                ax_h2h.set_xlabel("Lap Number", fontsize=12)
                ax_h2h.set_ylabel("Lap Time (s)", fontsize=12)
                ax_h2h.legend(fontsize=11)
                st.pyplot(fig_h2h)

elif page == "🔄 Track History":
    st.title("🔄 Race Strategy History")
    
    c1, c2, c3 = st.columns(3)
    y_hist = c1.number_input("Year", 2018, 2026, 2024, key="y_hist")
    gp_hist = c2.text_input("GP", "Bahrain", key="gp_hist")
    sess_hist = c3.selectbox("Session", ["Race (R)", "Sprint (S)", "Qualifying (Q)"], key="s_hist")
    
    sess_code = re.search(r'\((.*?)\)', sess_hist).group(1)
    
    if st.button("Load Grid Data", type="primary"):
        with st.spinner("Loading driver stints..."):
            try:
                session = load_fastf1_data(y_hist, gp_hist, sess_code)
                if session.laps.empty:
                    st.error("No lap data available.")
                else:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    for driver in session.drivers:
                        driver_abbr = session.get_driver(driver)["Abbreviation"]
                        stints = session.laps.pick_driver(driver)[["Stint", "Compound", "LapNumber"]].groupby(["Stint", "Compound"]).count().reset_index()
                        
                        prev = 0
                        for _, r in stints.iterrows():
                            compound_name = str(r["Compound"])
                            try:
                                color = fastf1.plotting.get_compound_color(compound_name, session=session)
                            except Exception:
                                color = "grey"
                                
                            ax.barh(driver_abbr, r["LapNumber"], left=prev, color=color, edgecolor="black", height=0.6)
                            prev += r["LapNumber"]
                            
                    ax.invert_yaxis()
                    ax.set_title(f"Stint History - {gp_hist} {y_hist}", fontsize=14, fontweight='bold', pad=15)
                    ax.set_xlabel("Lap Number", fontsize=12)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error : {e}")

elif page == "🌍 Track Database":
    st.title("🌍 Track Database")
    
    df = pd.DataFrame.from_dict(CIRCUITS_DATA, orient='index').reset_index().rename(
        columns={"index": "Track", "laps": "Laps", "pit_loss": "Pit Loss (s)", "length": "Length (km)", "abrasion": "Abrasion", "record": "Lap Record"}
    )
    st.dataframe(df, use_container_width=True, hide_index=True)