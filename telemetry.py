import os
import numpy as np
import pandas as pd
import streamlit as st
import fastf1
from scipy.optimize import curve_fit

# Init cache FastF1 pour éviter de spammer l'API
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

@st.cache_data(show_spinner=False)
def load_fastf1_data(year, gp, s_code):
    session = fastf1.get_session(year, gp, s_code)
    session.load(telemetry=False, weather=False)
    return session

# Modèles de dégradation théoriques
def modele_lineaire(t, t0, a): 
    return t0 + a * t

def modele_quadratique(t, t0, a, b): 
    return t0 + a * t + b * t**2

def modele_exponentiel(t, t0, a, b): 
    return t0 + a * (np.exp(b * t) - 1)

def fit_degradation_models(tours_relatifs, temps_corriges):
    # Fit lineaire (fallback si les autres échouent)
    popt_lin, _ = curve_fit(modele_lineaire, tours_relatifs, temps_corriges)
    
    # Fit quadratique (borné > 0 pour éviter les inversions de courbe)
    limites_quad = ([0, 0, 0], [np.inf, np.inf, np.inf])
    popt_quad, _ = curve_fit(modele_quadratique, tours_relatifs, temps_corriges, bounds=limites_quad)
    
    # Fit exponentiel (limité pour éviter des overflows de Numpy)
    try:
        limites_exp = ([0, 0, 0], [np.inf, 10, 0.5])
        popt_exp, _ = curve_fit(modele_exponentiel, tours_relatifs, temps_corriges, bounds=limites_exp)
    except:
        popt_exp = [popt_lin[0], popt_lin[1], 0]
        
    return popt_lin, popt_quad, popt_exp

def get_head_to_head_laps(session, driver1, driver2):
    laps_d1 = session.laps.pick_driver(driver1)
    laps_d2 = session.laps.pick_driver(driver2)
    return laps_d1, laps_d2

def get_actual_strategy(session, driver):
    laps = session.laps.pick_driver(driver)
    if laps.empty:
        return None
        
    stints = laps.groupby('Stint').agg({
        'Compound': 'first', 
        'LapNumber': 'max'
    }).reset_index()
    
    compounds = [str(c).capitalize() for c in stints['Compound']]
    
    # Cast explicite en int pour éviter les fuites de np.int64 dans l'UI
    pit_laps = [int(x) for x in stints['LapNumber'].tolist()[:-1]]
    total_laps_done = int(stints['LapNumber'].max())
    
    return {
        "compounds": compounds,
        "pit_laps": pit_laps,
        "total_laps": total_laps_done
    }