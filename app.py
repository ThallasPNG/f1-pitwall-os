import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # NOUVEAU : Import de seaborn
import fastf1.plotting
import re

# Application du thème clair et professionnel globalement
sns.set_theme(style="whitegrid")

# Importation de nos propres modules modulaires
from optimization import (solve_dp, optimize_all_strategies, get_pit_loss_for_lap, 
                          calculate_stint_time, get_base_lap_time, run_monte_carlo, evaluate_fixed_strategy)
from telemetry import (load_fastf1_data, fit_degradation_models, get_head_to_head_laps,
                       get_actual_strategy, modele_lineaire, modele_quadratique, modele_exponentiel)
from report import generate_pdf_report

# --- CONFIGURATION DE LA PAGE STREAMLIT ---
st.set_page_config(page_title="F1 Pit Wall OS", page_icon="🏁", layout="wide")

# Initialisation de la mémoire de session
if 'custom_strats' not in st.session_state:
    st.session_state.custom_strats = [('Medium', 'Hard'), ('Soft', 'Hard'), ('Soft', 'Medium', 'Soft')]
if 'total_laps' not in st.session_state:
    st.session_state.total_laps = 52
if 'pit_loss_time' not in st.session_state:
    st.session_state.pit_loss_time = 28.0

# --- BASE DE DONNÉES DES CIRCUITS F1 ---
CIRCUITS_DATA = {
    "Bahreïn (Sakhir)": {"laps": 57, "pit_loss": 24.0, "length": 5.412, "abrasion": "Élevée", "record": "1:31.447"},
    "Arabie Saoudite (Jeddah)": {"laps": 50, "pit_loss": 22.0, "length": 6.174, "abrasion": "Moyenne", "record": "1:30.734"},
    "Australie (Melbourne)": {"laps": 58, "pit_loss": 20.0, "length": 5.278, "abrasion": "Moyenne", "record": "1:19.815"},
    "Japon (Suzuka)": {"laps": 53, "pit_loss": 23.0, "length": 5.807, "abrasion": "Très Élevée", "record": "1:30.983"},
    "Chine (Shanghai)": {"laps": 56, "pit_loss": 24.0, "length": 5.451, "abrasion": "Élevée", "record": "1:32.238"},
    "USA (Miami)": {"laps": 57, "pit_loss": 22.0, "length": 5.412, "abrasion": "Moyenne", "record": "1:29.708"},
    "Émilie-Romagne (Imola)": {"laps": 63, "pit_loss": 28.0, "length": 4.909, "abrasion": "Moyenne", "record": "1:15.484"},
    "Monaco (Monte-Carlo)": {"laps": 78, "pit_loss": 25.0, "length": 3.337, "abrasion": "Très Faible", "record": "1:12.909"},
    "Canada (Montréal)": {"laps": 70, "pit_loss": 18.0, "length": 4.361, "abrasion": "Faible", "record": "1:13.078"},
    "Espagne (Barcelone)": {"laps": 66, "pit_loss": 23.0, "length": 4.657, "abrasion": "Élevée", "record": "1:16.330"},
    "Autriche (Spielberg)": {"laps": 71, "pit_loss": 20.0, "length": 4.318, "abrasion": "Moyenne", "record": "1:05.619"},
    "Royaume-Uni (Silverstone)": {"laps": 52, "pit_loss": 28.0, "length": 5.891, "abrasion": "Élevée", "record": "1:27.097"},
    "Hongrie (Hungaroring)": {"laps": 70, "pit_loss": 20.0, "length": 4.381, "abrasion": "Moyenne", "record": "1:16.627"},
    "Belgique (Spa-Francorchamps)": {"laps": 44, "pit_loss": 24.0, "length": 7.004, "abrasion": "Élevée", "record": "1:46.286"},
    "Pays-Bas (Zandvoort)": {"laps": 72, "pit_loss": 18.0, "length": 4.259, "abrasion": "Élevée", "record": "1:11.097"},
    "Italie (Monza)": {"laps": 53, "pit_loss": 24.0, "length": 5.793, "abrasion": "Faible", "record": "1:21.046"},
    "Azerbaïdjan (Bakou)": {"laps": 51, "pit_loss": 21.0, "length": 6.003, "abrasion": "Faible", "record": "1:43.009"},
    "Singapour (Marina Bay)": {"laps": 62, "pit_loss": 29.0, "length": 4.940, "abrasion": "Moyenne", "record": "1:35.867"},
    "USA (Austin)": {"laps": 56, "pit_loss": 20.0, "length": 5.513, "abrasion": "Élevée", "record": "1:36.169"},
    "Mexique (Mexico)": {"laps": 71, "pit_loss": 22.0, "length": 4.304, "abrasion": "Faible", "record": "1:17.774"},
    "Brésil (Interlagos)": {"laps": 71, "pit_loss": 24.0, "length": 4.309, "abrasion": "Moyenne", "record": "1:10.540"},
    "USA (Las Vegas)": {"laps": 50, "pit_loss": 20.0, "length": 6.201, "abrasion": "Faible", "record": "1:35.490"},
    "Qatar (Lusail)": {"laps": 57, "pit_loss": 25.0, "length": 5.419, "abrasion": "Très Élevée", "record": "1:24.319"},
    "Abou Dabi (Yas Marina)": {"laps": 58, "pit_loss": 23.0, "length": 5.281, "abrasion": "Moyenne", "record": "1:26.103"}
}

def apply_to_sidebar(comp, base, a, b2):
    prefixes = {"SOFT": "s", "MEDIUM": "m", "HARD": "h"}
    p = prefixes[comp]
    st.session_state[f"{p}b"] = float(base)
    st.session_state[f"{p}a"] = float(a)
    st.session_state[f"{p}b2"] = float(b2)

# ==========================================
# BARRE LATÉRALE - MENU ET CONFIGURATION
# ==========================================
st.sidebar.title("🏁 Pit Wall OS")
st.sidebar.markdown("---")

page = st.sidebar.radio("🧭 NAVIGATION", [
    "🏠 Mission Control",
    "🛠️ Constructeur de Stratégie",
    "🎲 Monte-Carlo & Risques",
    "📈 Modélisation Télémétrie",
    "🥊 Face-à-Face & Validation",
    "🔄 Historique des Stratégies",
    "🌍 Base de données Circuits"
])
st.sidebar.markdown("---")

st.sidebar.header("⚙️ Paramètres du GP")
selected_circuit = st.sidebar.selectbox("Sélectionnez la course :", list(CIRCUITS_DATA.keys()))
TOTAL_LAPS = CIRCUITS_DATA[selected_circuit]["laps"]
PIT_LOSS_TIME = CIRCUITS_DATA[selected_circuit]["pit_loss"]

FUEL_EFFECT = st.sidebar.slider("Effet Carburant (gain en s/tour)", min_value=0.0, max_value=0.15, value=0.06, step=0.01)

with st.sidebar.expander("🚓 Voiture de Sécurité (SC)", expanded=False):
    sc_active = st.checkbox("Activer un événement SC", value=False)
    if sc_active:
        sc_start = st.slider("Tour de déploiement", 1, TOTAL_LAPS, 20)
        sc_duration = st.slider("Durée de la SC (tours)", 1, 10, 3)
        sc_pit_loss = st.slider("Perte aux stands sous SC (s)", 10.0, PIT_LOSS_TIME, 15.0, 0.5)
        sc_slowdown = st.slider("Ralentissement du peloton (s/tour)", 10.0, 50.0, 30.0, 1.0)
        sc_deg_factor = st.slider("Facteur d'usure SC (0.25 = 25%)", 0.0, 1.0, 0.25, 0.05)
    else:
        sc_start, sc_duration, sc_pit_loss, sc_slowdown, sc_deg_factor = -1, 0, PIT_LOSS_TIME, 0.0, 1.0

sc_config = {
    'active': sc_active, 'start': sc_start, 'duration': sc_duration, 
    'pit_loss': sc_pit_loss, 'slowdown': sc_slowdown, 'deg_factor': sc_deg_factor
}

st.sidebar.markdown("---")
st.sidebar.header("🏎️ Propriétés des Gommes")
deg_model = st.sidebar.selectbox("Loi Mathématique de Dégradation", ["Quadratique", "Exponentiel", "Linéaire"])

with st.sidebar.expander("🔴 Pneus Soft (Tendres)", expanded=False):
    s_b = st.number_input("Temps de base initial (s)", value=93.8, key='sb', step=0.1)
    s_a = st.number_input("Usure linéaire (Paramètre a)", value=0.148, format="%.3f", key='sa', step=0.01)
    s_b2 = st.number_input("Chute thermique (Paramètre b)", value=0.002, format="%.4f", key='sb2', step=0.001)
    s_w = st.number_input("Pénalité de Warm-up (s)", value=1.0, step=0.5, key='sw')

with st.sidebar.expander("🟡 Pneus Medium (Médiums)", expanded=False):
    m_b = st.number_input("Temps de base initial (s)", value=94.13, key='mb', step=0.1)
    m_a = st.number_input("Usure linéaire (Paramètre a)", value=0.076, format="%.3f", key='ma', step=0.01)
    m_b2 = st.number_input("Chute thermique (Paramètre b)", value=0.001, format="%.4f", key='mb2', step=0.001)
    m_w = st.number_input("Pénalité de Warm-up (s)", value=2.0, step=0.5, key='mw')

with st.sidebar.expander("⚪ Pneus Hard (Durs)", expanded=False):
    h_b = st.number_input("Temps de base initial (s)", value=94.8, key='hb', step=0.1)
    h_a = st.number_input("Usure linéaire (Paramètre a)", value=0.056, format="%.3f", key='ha', step=0.01)
    h_b2 = st.number_input("Chute thermique (Paramètre b)", value=0.000, format="%.4f", key='hb2', step=0.001)
    h_w = st.number_input("Pénalité de Warm-up (s)", value=3.5, step=0.5, key='hw')

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


# ==========================================
# GESTION DES PAGES
# ==========================================

if page == "🏠 Mission Control":
    st.title(f"🏠 Mission Control - {selected_circuit}")
    st.markdown("Vue d'ensemble stratégique générée automatiquement par la Programmation Dynamique.")
    
    if sc_active:
        st.warning(f"🚓 Alerte Safety Car : Active du tour {sc_start} à {sc_start + sc_duration - 1}.")
    
    with st.spinner("Calcul de la stratégie mathématique parfaite en cours..."):
        strats = []
        for stops in [1, 2, 3]:
            s = optimize_all_strategies(TOTAL_LAPS, stops, TIRE_MODELS, sc_config, PIT_LOSS_TIME, deg_model, FUEL_EFFECT)
            if s: strats.append(s)
        strats.sort(key=lambda x: x['total_time'])
        best = strats[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distance", f"{TOTAL_LAPS} Tours")
    c2.metric("Meilleure Stratégie", " ➔ ".join(best['compounds']))
    
    # Nettoyage de l'affichage des tours
    pits_clean = ", ".join(map(str, best['pit_laps']))
    c3.metric("Fenêtres d'arrêts", f"Tours {pits_clean}")
    c4.metric("Temps de course estimé", format_time(best['total_time']))
    
    st.markdown("---")
    st.subheader("📈 Projection de Rythme (Pace)")
    
    fig_mc, ax_mc = plt.subplots(figsize=(14, 6))
    laps_mc = get_full_race_laps(best)
    
    # Couleur adaptée pour le fond blanc
    ax_mc.plot(np.arange(1, len(laps_mc)+1), laps_mc, label=f"Optimum Mathématique : {' ➔ '.join(best['compounds'])}", color="#2ca02c", lw=3)
    if sc_active: 
        ax_mc.axvspan(sc_start, sc_start + sc_duration - 1, color='gold', alpha=0.3, label='Période Safety Car')
        
    ax_mc.set_title(f"Simulation du rythme de course parfait - {selected_circuit}", fontsize=14, fontweight='bold', pad=15)
    ax_mc.set_xlabel("Numéro du Tour", fontsize=12)
    ax_mc.set_ylabel("Temps au tour (secondes)", fontsize=12)
    ax_mc.legend(fontsize=11)
    st.pyplot(fig_mc)
    
    st.markdown("---")
    st.subheader("📄 Rapport de Stratégie (Brief Pilote)")
    st.info("Générez un rapport PDF contenant la stratégie optimale et le graphique de rythme, prêt à être imprimé ou partagé.")
    
    pdf_bytes = generate_pdf_report(selected_circuit, best, TOTAL_LAPS, PIT_LOSS_TIME, format_time(best['total_time']), fig_mc)
    st.download_button(
        label="📥 Télécharger le Brief Stratégique (PDF)",
        data=pdf_bytes,
        file_name=f"Strategy_Brief_{selected_circuit}.pdf",
        mime="application/pdf",
        type="primary"
    )

elif page == "🛠️ Constructeur de Stratégie":
    st.title("🛠️ Constructeur et Comparateur de Stratégies")
    st.markdown("Testez vos propres idées stratégiques et comparez-les visuellement et mathématiquement.")
    
    with st.expander("➕ Créer une stratégie manuelle", expanded=True):
        st_stops = st.radio("Nombre d'arrêts", [1, 2, 3], horizontal=True)
        st_cols = st.columns(st_stops + 1)
        current_build = [st_cols[i].selectbox(f"Relais {i+1}", ["Soft", "Medium", "Hard"], key=f"sel_{i}") for i in range(st_stops + 1)]
        
        if st.button("Ajouter à la comparaison", type="primary"):
            st.session_state.custom_strats.append(tuple(current_build))
            st.rerun()

    st.markdown("---")
    st.subheader("🗂️ Stratégies Actuelles")
    if st.session_state.custom_strats:
        for i, strat in enumerate(st.session_state.custom_strats):
            col_s, col_b = st.columns([4, 1])
            col_s.markdown(f"**Stratégie {i+1} :** {len(strat)-1} Arrêt(s) ➔ {' - '.join(strat)}")
            if col_b.button("❌ Supprimer", key=f"del_{i}"):
                st.session_state.custom_strats.pop(i)
                st.rerun()
                
        if st.button("🗑️ Tout effacer"):
            st.session_state.custom_strats = []
            st.rerun()

        st.markdown("---")
        results = []
        for comps in st.session_state.custom_strats:
            t_best, pits = solve_dp(comps, TOTAL_LAPS, TIRE_MODELS, sc_config, PIT_LOSS_TIME, deg_model, FUEL_EFFECT)
            # Nettoyage des chaînes pour l'affichage
            pits_clean = ", ".join(map(str, pits))
            results.append({"label": f"{len(comps)-1} Stop(s) : {' ➔ '.join(comps)}", "comps": comps, "time": t_best, "pits_clean": pits_clean, "pits": pits})
            
        results.sort(key=lambda x: x["time"], reverse=True)
        
        st.subheader("1. Évaluation Globale")
        fig1, ax1 = plt.subplots(figsize=(12, max(4, len(results) * 0.8)))
        # Couleur bleu standard seaborn
        ax1.barh([r["label"] for r in results], [r["time"] for r in results], color='#1f77b4')
        min_time, max_time = min([r["time"] for r in results]), max([r["time"] for r in results])
        ax1.set_xlim(min_time - 5, max_time + 5)
        ax1.set_xlabel("Temps total de course (secondes)", fontsize=12)
        st.pyplot(fig1)

        st.subheader("2. Comparaison du Rythme (Pace)")
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        # Palette de couleurs professionnelle
        colors = sns.color_palette("tab10", len(results))
        
        for i, res in enumerate(results): 
            laps = get_full_race_laps({'compounds': res["comps"], 'pit_laps': res["pits"]})
            ax2.plot(np.arange(1, len(laps)+1), laps, label=f"{res['label']} (Pits: {res['pits_clean']})", color=colors[i], lw=2.5)
            
        if sc_active: 
            ax2.axvspan(sc_start, sc_start + sc_duration - 1, color='gold', alpha=0.3, label='Safety Car')
            
        ax2.set_xlabel("Numéro du Tour", fontsize=12)
        ax2.set_ylabel("Temps au tour (s)", fontsize=12)
        ax2.legend(fontsize=11)
        st.pyplot(fig2)
    else:
        st.info("Aucune stratégie personnalisée. Ajoutez-en une via le constructeur ci-dessus.")

elif page == "🎲 Monte-Carlo & Risques":
    st.title("🎲 Simulation Monte-Carlo (Évaluation des Risques)")
    st.markdown("La théorie c'est bien, la réalité c'est mieux. Simulez des milliers de courses en injectant le chaos du monde réel : **variance du rythme au tour (±0.3s)** et probabilité de **5% de rater un arrêt aux stands (+3s à +8s de perte)**.")
    
    n_sims = st.slider("Nombre de courses parallèles à simuler", 100, 5000, 1000)
    
    if st.session_state.custom_strats:
        if st.button("Lancer la Matrice de Risque", type="primary"):
            with st.spinner(f"Génération de {n_sims} futurs possibles..."):
                fig, ax = plt.subplots(figsize=(14, 7))
                stats = []
                colors = sns.color_palette("tab10", len(st.session_state.custom_strats))
                
                for i, comps in enumerate(st.session_state.custom_strats):
                    t_best, pits = solve_dp(comps, TOTAL_LAPS, TIRE_MODELS, sc_config, PIT_LOSS_TIME, deg_model, FUEL_EFFECT)
                    times = run_monte_carlo(comps, pits, TOTAL_LAPS, TIRE_MODELS, PIT_LOSS_TIME, deg_model, FUEL_EFFECT, n_sims)
                    
                    label_name = f"{len(comps)-1} Arrêt(s) : {' ➔ '.join(comps)}"
                    color = colors[i]
                    
                    ax.hist(times, bins=60, alpha=0.5, color=color, label=label_name, density=False)
                    median_time = np.median(times)
                    ax.axvline(median_time, color=color, linestyle='dashed', linewidth=2)
                    
                    stats.append({
                        "Stratégie": label_name, 
                        "Temps Idéal (DP)": format_time(t_best), 
                        "Médiane (Réalité)": format_time(median_time),
                        "Pire Scénario (P95)": format_time(np.percentile(times, 95)),
                        "Risque Moyen Ajouté": f"+{(median_time - t_best):.2f} s"
                    })
                    
                ax.set_title(f"Distribution des temps de course ({n_sims} simulations)", fontsize=14, fontweight='bold', pad=15)
                ax.set_xlabel("Temps total de course (secondes)", fontsize=12)
                ax.set_ylabel("Fréquence d'occurrence", fontsize=12)
                ax.legend(fontsize=11)
                st.pyplot(fig)
                
                st.markdown("#### 📊 Bilan de Robustesse")
                st.table(pd.DataFrame(stats))
    else:
        st.warning("⚠️ Ajoutez des stratégies via le **Constructeur de Stratégie** avant de lancer Monte-Carlo.")

elif page == "📈 Modélisation Télémétrie":
    st.title("📈 Fit Télémétrie (Machine Learning via SciPy)")
    st.markdown("Extrayez les temps réels d'une session passée et laissez l'algorithme trouver les coefficients mathématiques parfaits pour vos pneus.")
    
    c1, c2, c3, c4 = st.columns(4)
    y2 = c1.number_input("Année", 2018, 2026, 2023, key='y2')
    gp2 = c2.text_input("GP", "Silverstone", key='gp2')
    d2 = c3.text_input("Pilote (ex: HAM)", "HAM", key='d2_tel')
    comp2 = c4.selectbox("Gomme analysée", ["SOFT", "MEDIUM", "HARD"])
    
    sess_type = st.selectbox("Type de Session", ["Race (R)", "Sprint (S)", "Practice 2 (FP2)"])
    sess_code = re.search(r'\((.*?)\)', sess_type).group(1)
    
    if st.button("📊 Analyser et Modéliser la Gomme", type="primary"):
        with st.spinner("Téléchargement FastF1 et ajustement SciPy en cours..."):
            session = load_fastf1_data(y2, gp2, sess_code)
            laps_driver = session.laps.pick_driver(d2).pick_compounds(comp2)
            
            if laps_driver.empty:
                st.error("Aucune donnée trouvée pour ce pilote et cette gomme.")
            else:
                longest_stint = laps_driver['Stint'].value_counts().idxmax()
                stint_laps = laps_driver[laps_driver["Stint"] == longest_stint].pick_quicklaps()
                
                if len(stint_laps) < 3:
                    st.error("Pas assez de tours propres pour entraîner le modèle mathématique.")
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
                    
                    ax.scatter(tours_abs, temps_corriges, color='teal', s=60, edgecolor='white', label='Temps Réels Corrigés (Dégradation pure)')
                    ax.plot(t_lisses_abs, modele_lineaire(t_lisses_rel, *popt_lin), color='darkorange', linestyle=':', lw=2, label='Fit Linéaire')
                    ax.plot(t_lisses_abs, modele_quadratique(t_lisses_rel, *popt_quad), color='crimson', linestyle='--', lw=2, label='Fit Quadratique')
                    ax.plot(t_lisses_abs, modele_exponentiel(t_lisses_rel, *popt_exp), color='purple', lw=3, label='Fit Exponentiel')
                    
                    ax.axvline(max(tours_abs), color="gray", linestyle="--", label="Fin des données réelles")
                    
                    ax.set_title(f"Ajustement Mathématique de la Dégradation ({comp2}) - {d2} ({gp2} {y2})", fontsize=14, fontweight='bold', pad=15)
                    ax.set_xlabel("Numéro du Tour Absolu", fontsize=12)
                    ax.set_ylabel("Temps au tour (s)", fontsize=12)
                    ax.legend(fontsize=11)
                    st.pyplot(fig)
                    
                    if deg_model == "Linéaire": base_v, a_v, b_v = popt_lin[0], popt_lin[1], 0.0
                    elif deg_model == "Exponentiel": base_v, a_v, b_v = popt_exp[0], popt_exp[1], popt_exp[2]
                    else: base_v, a_v, b_v = popt_quad[0], popt_quad[1], popt_quad[2]
                    
                    st.success("Modélisation réussie ! Les paramètres ont été trouvés.")
                    st.info("💡 Cliquez ci-dessous pour injecter ces données scientifiques directement dans la barre latérale pour la gomme étudiée.")
                    
                    st.button(
                        f"📥 Transférer les propriétés vers le pneu {comp2}", 
                        on_click=apply_to_sidebar, 
                        args=(comp2, base_v, a_v, b_v), 
                        type="primary",
                        use_container_width=True
                    )

elif page == "🥊 Face-à-Face & Validation":
    st.title("✅ Validation (Backtesting) & Face-à-Face")
    st.markdown("Comparez le modèle mathématique à la réalité, et analysez les batailles de rythme entre pilotes.")
    
    c1, c2, c3, c4 = st.columns(4)
    y_val = c1.number_input("Année", 2018, 2026, 2024, key="y_val")
    gp_val = c2.text_input("Grand Prix", "Bahrain", key="gp_val")
    d1 = c3.text_input("Pilote Principal (ex: VER)", "VER", key="d1_val")
    d2 = c4.text_input("Adversaire (ex: PER)", "PER", key="d2_val")
    
    if st.button("⚖️ Lancer l'Analyse Croisée", type="primary"):
        with st.spinner("Récupération Télémétrie FastF1 & Calculs..."):
            session = load_fastf1_data(y_val, gp_val, "R")
            actual_strat = get_actual_strategy(session, d1)
            
            st.subheader(f"1. Validation de la Stratégie ({d1})")
            if actual_strat:
                valid_compounds = ["Soft", "Medium", "Hard"]
                if not all(c in valid_compounds for c in actual_strat["compounds"]):
                    st.warning("Le pilote a utilisé des pneus pluie. Backtesting impossible (Slicks uniquement).")
                else:
                    laps_done = actual_strat["total_laps"]
                    t_actual = evaluate_fixed_strategy(actual_strat["compounds"], actual_strat["pit_laps"], laps_done, TIRE_MODELS, sc_config, PIT_LOSS_TIME, deg_model, FUEL_EFFECT)
                    opt = optimize_all_strategies(laps_done, len(actual_strat["compounds"])-1, TIRE_MODELS, sc_config, PIT_LOSS_TIME, deg_model, FUEL_EFFECT)
                    
                    # Formattage propre des listes sans np.int64
                    act_pits_clean = ", ".join(map(str, actual_strat['pit_laps']))
                    opt_pits_clean = ", ".join(map(str, opt['pit_laps']))
                    
                    col_a, col_b, col_c = st.columns(3)
                    col_a.markdown(f"**🏎️ Réalité ({d1})**<br>Gommes : {' ➔ '.join(actual_strat['compounds'])}<br>Pits : T{act_pits_clean}<br>Temps Modèle : **{format_time(t_actual)}**", unsafe_allow_html=True)
                    col_b.markdown(f"**💻 Optimum Mathématique**<br>Gommes : {' ➔ '.join(opt['compounds'])}<br>Pits : T{opt_pits_clean}<br>Temps Calculé : **{format_time(opt['total_time'])}**", unsafe_allow_html=True)
                    
                    delta = t_actual - opt['total_time']
                    if delta < 0.5:
                        col_c.metric("Écart", f"{delta:.2f} s", delta="Parfait !", delta_color="normal")
                    else:
                        col_c.metric("Écart", f"+{delta:.2f} s", delta="Améliorable", delta_color="inverse")
                    
                    fig_val, ax_val = plt.subplots(figsize=(14, 6))
                    laps_act_plot = get_full_race_laps(actual_strat, custom_total_laps=laps_done)
                    laps_opt_plot = get_full_race_laps(opt, custom_total_laps=laps_done)
                    
                    # Sur fond blanc, la ligne blanche devient noire pour être visible
                    ax_val.plot(np.arange(1, len(laps_act_plot)+1), laps_act_plot, label=f"Stratégie Réelle de {d1}", color="black", lw=2, linestyle='--')
                    ax_val.plot(np.arange(1, len(laps_opt_plot)+1), laps_opt_plot, label="Optimum DP", color="#2ca02c", lw=3)
                    ax_val.set_title(f"Superposition Réalité vs Optimisation Mathématique", fontsize=14, fontweight='bold', pad=15)
                    ax_val.set_xlabel("Tour", fontsize=12)
                    ax_val.set_ylabel("Temps (s)", fontsize=12)
                    ax_val.legend(fontsize=11)
                    st.pyplot(fig_val)
            else:
                st.error("Données stratégiques réelles introuvables.")

            st.markdown("---")
            st.subheader(f"2. Bataille de Rythme : {d1} vs {d2}")
            laps_d1, laps_d2 = get_head_to_head_laps(session, d1, d2)
            
            if laps_d1.empty or laps_d2.empty:
                st.error("Télémétrie manquante pour l'un des pilotes.")
            else:
                med_d1 = laps_d1['LapTime'].dt.total_seconds().median()
                laps_d1_clean = laps_d1[laps_d1['LapTime'].dt.total_seconds() < med_d1 + 5]
                med_d2 = laps_d2['LapTime'].dt.total_seconds().median()
                laps_d2_clean = laps_d2[laps_d2['LapTime'].dt.total_seconds() < med_d2 + 5]
                
                fig_h2h, ax_h2h = plt.subplots(figsize=(14, 6))
                ax_h2h.plot(laps_d1_clean['LapNumber'], laps_d1_clean['LapTime'].dt.total_seconds(), label=d1, color='teal', lw=2, marker='o', markersize=4)
                ax_h2h.plot(laps_d2_clean['LapNumber'], laps_d2_clean['LapTime'].dt.total_seconds(), label=d2, color='darkorange', lw=2, marker='x', markersize=4)
                
                ax_h2h.set_title(f"Face-à-Face en piste : {d1} vs {d2}", fontsize=14, fontweight='bold', pad=15)
                ax_h2h.set_xlabel("Tour", fontsize=12)
                ax_h2h.set_ylabel("Temps au tour (s)", fontsize=12)
                ax_h2h.legend(fontsize=11)
                st.pyplot(fig_h2h)

elif page == "🔄 Historique des Stratégies":
    st.title("🔄 Historique Global des Stratégies (Session)")
    st.markdown("Visualisez les stratégies de pneus réellement adoptées par **tous les pilotes** lors d'une session passée.")
    
    c1, c2, c3 = st.columns(3)
    y_hist = c1.number_input("Année", 2018, 2026, 2024, key="y_hist")
    gp_hist = c2.text_input("Grand Prix", "Bahrain", key="gp_hist")
    sess_hist = c3.selectbox("Session", ["Race (R)", "Sprint (S)", "Qualifying (Q)"], key="s_hist")
    
    sess_code = re.search(r'\((.*?)\)', sess_hist).group(1)
    
    if st.button("📊 Afficher la Grille des Stratégies", type="primary"):
        with st.spinner("Analyse du peloton via FastF1..."):
            try:
                session = load_fastf1_data(y_hist, gp_hist, sess_code)
                if session.laps.empty:
                    st.error("Aucune donnée de tour trouvée.")
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
                    ax.set_title(f"Historique des Relais - {gp_hist} {y_hist} ({sess_code})", fontsize=14, fontweight='bold', pad=15)
                    ax.set_xlabel("Numéro du Tour", fontsize=12)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur : {e}")

elif page == "🌍 Base de données Circuits":
    st.title("🌍 Encyclopédie des Circuits F1")
    st.markdown("Consultez les informations clés pour configurer rapidement votre stratégie.")
    
    df = pd.DataFrame.from_dict(CIRCUITS_DATA, orient='index').reset_index().rename(
        columns={"index": "Circuit", "laps": "Tours", "pit_loss": "Perte aux Stands (s)", "length": "Longueur (km)", "abrasion": "Abrasion", "record": "Record du Tour"}
    )
    st.dataframe(df, use_container_width=True, hide_index=True)