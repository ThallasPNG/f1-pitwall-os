import os
import numpy as np
import pandas as pd
import streamlit as st
import fastf1
from scipy.optimize import curve_fit

# Configuration globale de FastF1
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

@st.cache_data(show_spinner=False)
def load_fastf1_data(year, gp, s_code):
    """Télécharge et met en cache les données de la session F1."""
    session = fastf1.get_session(year, gp, s_code)
    session.load(telemetry=False, weather=False)
    return session

# --- Fonctions des modèles mathématiques ---
def modele_lineaire(t, t0, a): 
    return t0 + a * t

def modele_quadratique(t, t0, a, b): 
    return t0 + a * t + b * t**2

def modele_exponentiel(t, t0, a, b): 
    return t0 + a * (np.exp(b * t) - 1)

def fit_degradation_models(tours_relatifs, temps_corriges):
    """Entraîne les 3 modèles mathématiques sur les données réelles."""
    popt_lin, _ = curve_fit(modele_lineaire, tours_relatifs, temps_corriges)
    
    limites_quad = ([0, 0, 0], [np.inf, np.inf, np.inf])
    popt_quad, _ = curve_fit(modele_quadratique, tours_relatifs, temps_corriges, bounds=limites_quad)
    
    try:
        limites_exp = ([0, 0, 0], [np.inf, 10, 0.5])
        popt_exp, _ = curve_fit(modele_exponentiel, tours_relatifs, temps_corriges, bounds=limites_exp)
    except:
        popt_exp = [popt_lin[0], popt_lin[1], 0]
        
    return popt_lin, popt_quad, popt_exp

def get_head_to_head_laps(session, driver1, driver2):
    """Récupère les données de deux pilotes pour l'onglet Bataille."""
    laps_d1 = session.laps.pick_driver(driver1)
    laps_d2 = session.laps.pick_driver(driver2)
    return laps_d1, laps_d2

def get_actual_strategy(session, driver):
    """Extrait la stratégie réelle (Gommes et Pits) exécutée par un pilote."""
    laps = session.laps.pick_driver(driver)
    if laps.empty:
        return None
        
    stints = laps.groupby('Stint').agg({
        'Compound': 'first', 
        'LapNumber': 'max'
    }).reset_index()
    
    compounds = [str(c).capitalize() for c in stints['Compound']]
    # CORRECTION : cast explicite en int Python natif pour éviter les np.int64
    pit_laps = [int(x) for x in stints['LapNumber'].tolist()[:-1]]
    total_laps_done = int(stints['LapNumber'].max())
    
    return {
        "compounds": compounds,
        "pit_laps": pit_laps,
        "total_laps": total_laps_done
    }