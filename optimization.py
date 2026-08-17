import numpy as np
import itertools

def get_base_lap_time(base, a, b, age, model_type):
    if model_type == "Linéaire":
        return base + (a * age)
    elif model_type == "Exponentiel":
        return base + (a * (np.exp(b * age) - 1))
    else: 
        return base + (a * age) + (b * (age**2))

def calculate_stint_time(compound, laps_in_stint, start_lap, tire_models, sc_config, deg_model, fuel_effect):
    if laps_in_stint <= 0: return 0.0
    params = tire_models[compound]
    
    total_time = 0.0
    current_age = 1.0 
    
    for i in range(laps_in_stint):
        current_race_lap = start_lap + i
        is_sc = sc_config['active'] and (sc_config['start'] <= current_race_lap < sc_config['start'] + sc_config['duration'])
        
        base_lap_time = get_base_lap_time(params['base'], params['a'], params['b'], current_age, deg_model)
        base_lap_time -= (current_race_lap * fuel_effect)
        
        if start_lap > 1 and i == 0:
            base_lap_time += params['warmup']
        
        if is_sc:
            total_time += (base_lap_time + sc_config['slowdown'])
            current_age += sc_config['deg_factor']
        else:
            total_time += base_lap_time
            current_age += 1.0
            
    return total_time

def get_pit_loss_for_lap(lap, sc_config, normal_pit_loss):
    if sc_config['active'] and (sc_config['start'] <= lap < sc_config['start'] + sc_config['duration']):
        return sc_config['pit_loss']
    return normal_pit_loss

def solve_dp(comps, total_laps, tire_models, sc_config, normal_pit_loss, deg_model, fuel_effect):
    n_stints = len(comps)
    dp = np.full((n_stints, total_laps + 1), np.inf)
    parent = np.zeros((n_stints, total_laps + 1), dtype=int)
    
    memo_stints = {}
    def get_stint(c, laps, start):
        if (c, laps, start) not in memo_stints:
            memo_stints[(c, laps, start)] = calculate_stint_time(c, laps, start, tire_models, sc_config, deg_model, fuel_effect)
        return memo_stints[(c, laps, start)]

    for lap in range(1, total_laps + 1):
        dp[0][lap] = get_stint(comps[0], lap, 1)

    for stint_idx in range(1, n_stints):
        comp = comps[stint_idx]
        for curr_lap in range(stint_idx + 1, total_laps + 1):
            best_cost = np.inf
            best_prev = -1
            
            for prev_lap in range(stint_idx, curr_lap):
                pit_loss = get_pit_loss_for_lap(prev_lap, sc_config, normal_pit_loss)
                stint_t = get_stint(comp, curr_lap - prev_lap, prev_lap + 1)
                
                cost = dp[stint_idx - 1][prev_lap] + pit_loss + stint_t
                if cost < best_cost:
                    best_cost = cost
                    best_prev = prev_lap
                    
            dp[stint_idx][curr_lap] = best_cost
            parent[stint_idx][curr_lap] = best_prev

    best_total_time = dp[n_stints - 1][total_laps]
    pit_laps = []
    curr = total_laps
    for i in range(n_stints - 1, 0, -1):
        prev = parent[i][curr]
        # CORRECTION : cast explicite en int natif
        pit_laps.append(int(prev))
        curr = prev

    pit_laps.reverse()
    return best_total_time, pit_laps

def optimize_all_strategies(total_laps, n_stops, tire_models, sc_config, normal_pit_loss, deg_model, fuel_effect):
    compounds = list(tire_models.keys())
    combinations = list(itertools.product(compounds, repeat=n_stops+1))
    best_time = float('inf')
    best_strategy = None
    
    for comps in combinations:
        if len(set(comps)) < 2: 
            continue
        t, pits = solve_dp(comps, total_laps, tire_models, sc_config, normal_pit_loss, deg_model, fuel_effect)
        if t < best_time:
            best_time = t
            best_strategy = {'stops': n_stops, 'compounds': comps, 'pit_laps': pits, 'total_time': t}
    return best_strategy

def run_monte_carlo(comps, pit_laps, total_laps, tire_models, normal_pit_loss, deg_model, fuel_effect, n_simulations=1000):
    total_times = np.zeros(n_simulations)
    current_lap = 1
    pit_laps_full = list(pit_laps) + [total_laps]
    
    for i, pit_lap in enumerate(pit_laps_full):
        stint_laps = pit_lap - current_lap + 1
        if stint_laps <= 0: 
            continue
        
        params = tire_models[comps[i]]
        age = np.arange(1, stint_laps + 1)
        base_times = get_base_lap_time(params['base'], params['a'], params['b'], age, deg_model)
        race_laps = np.arange(current_lap, pit_lap + 1)
        base_times -= (race_laps * fuel_effect)
        
        if current_lap > 1:
            base_times[0] += params['warmup']
        
        lap_variances = np.random.normal(loc=0.0, scale=0.3, size=(n_simulations, stint_laps))
        stint_times = np.sum(base_times + lap_variances, axis=1)
        total_times += stint_times
        
        if i < len(pit_laps_full) - 1:
            pit_times = np.random.normal(loc=normal_pit_loss, scale=0.6, size=n_simulations)
            bad_luck = np.random.rand(n_simulations) < 0.05
            pit_times[bad_luck] += np.random.uniform(3.0, 8.0, size=np.sum(bad_luck))
            total_times += pit_times
            
        current_lap = pit_lap + 1
        
    return total_times

def evaluate_fixed_strategy(comps, pit_laps, total_laps, tire_models, sc_config, normal_pit_loss, deg_model, fuel_effect):
    total_time = 0.0
    current_lap = 1
    pit_laps_full = list(pit_laps) + [total_laps]
    
    for i, p_lap in enumerate(pit_laps_full):
        stint_laps = p_lap - current_lap + 1
        if stint_laps > 0:
            total_time += calculate_stint_time(comps[i], stint_laps, current_lap, tire_models, sc_config, deg_model, fuel_effect)
        if i < len(pit_laps_full) - 1:
            total_time += get_pit_loss_for_lap(p_lap, sc_config, normal_pit_loss)
        current_lap = p_lap + 1
        
    return total_time