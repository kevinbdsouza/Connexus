import os
import json
import shutil
import time
import copy
import random
import math
import itertools # Used potentially in imported EC code
import numpy as np
import pandas as pd
import geopandas as gpd
from skopt import gp_minimize
from skopt.space import Real, Integer # Integer might be needed for binary flags if added later
from skopt.utils import use_named_args # Optional helper
import matplotlib.pyplot as plt # For optional plotting
from skopt.plots import plot_convergence # For optional plotting
from ei_policy import main_run_pyomo, run_single_optimization_policy
from graph_connectivity import solve_connectivity_ilp, solve_reposition_ilp
from ei_policy import save_gdf_for_ec, parse_geojson
from config import Config
from collections import defaultdict
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from functools import partial
import pyomo.environ as pyo
import matplotlib.ticker as mticker

RANDOM_SEED = 123

# Set random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

FONT_SIZE = 25
plt.rcParams['axes.labelsize'] = FONT_SIZE  # For x and y labels
plt.rcParams['axes.titlesize'] = FONT_SIZE  # For the subplot title
plt.rcParams['xtick.labelsize'] = FONT_SIZE  # For x-axis tick labels
plt.rcParams['ytick.labelsize'] = FONT_SIZE  # For y-axis tick labels
plt.rcParams['legend.fontsize'] = FONT_SIZE  # For the legend
plt.rcParams['figure.titlesize'] = FONT_SIZE

# --- Helper Function to Identify Crops ---
def get_unique_crop_labels(config_ids, base_farm_dir, sample_farms_per_config=2):
    """
    Scans sample farm input GeoJSONs to find unique crop labels on 'ag_plot' types.
    """
    unique_labels = set()
    print(f"Scanning for unique crop labels across {len(config_ids)} configs...")

    for config_id in config_ids:
        config_path = os.path.join(base_farm_dir, f"config_{config_id}")
        if not os.path.isdir(config_path):
            continue

        farm_dirs = [d for d in os.listdir(config_path)
                     if os.path.isdir(os.path.join(config_path, d)) and d.startswith("farm_")]
        if not farm_dirs:
            continue

        # Sample a few farms from this config
        sample_size = min(sample_farms_per_config, len(farm_dirs))
        sampled_farm_dirs = random.sample(farm_dirs, sample_size)

        for farm_dir_name in sampled_farm_dirs:
            farm_path = os.path.join(config_path, farm_dir_name)
            geojson_path = os.path.join(farm_path, "input.geojson")

            if os.path.exists(geojson_path):
                try:
                    gdf = gpd.read_file(geojson_path)
                    # Ensure 'type' and 'label' columns exist
                    if 'type' in gdf.columns and 'label' in gdf.columns:
                         # Filter for agricultural plots and get their labels
                         ag_labels = gdf.loc[gdf['type'] == 'ag_plot', 'label'].unique()
                         for label in ag_labels:
                             if pd.notna(label) and label.strip() != "":
                                 unique_labels.add(str(label)) # Add label as string
                    else:
                        # print(f"Warning: 'type' or 'label' column missing in {geojson_path}")
                        pass

                except Exception as e:
                    print(f"Warning: Could not read or process {geojson_path}: {e}")

    print(f"Scan complete. Found labels: {unique_labels}")
    # Ensure 'Unknown' is not treated as a crop to optimize premium for, if it's just a fallback
    unique_labels.discard("Unknown")
    return sorted(list(unique_labels))


def calculate_baseline_metrics_per_config(config_ids, base_farm_dir, cfg, ec_params_baseline, ec_params_repos,
                                          neighbor_dist, exit_tol, penalty_coef):
    baseline_metrics = {}
    print(f"Calculating baseline metrics (Conn + Farm NPV dict) per config for {len(config_ids)} configurations...")
    # ... (Implementation is identical to the previous response's version) ...
    for config_id in config_ids:
        print(f"  Processing Baseline Config: {config_id}", end='\r')
        config_path = os.path.join(base_farm_dir, f"config_{config_id}")
        config_results_dir = os.path.join(config_path, "policy_bo_temp")
        os.makedirs(config_results_dir, exist_ok=True)
        baseline_gdfs_config = []
        num_farms = sum(1 for item in os.listdir(config_path) if os.path.isdir(os.path.join(config_path, item)) and item.startswith("farm_"))
        if num_farms == 0: baseline_metrics[config_id] = None; continue
        farm_id_map_for_config = {}
        for farm_id in range(1, num_farms + 1):
            if farm_mode == "syn_farms":
                farm_path = os.path.join(config_path, f"farm_{farm_id}")
            else:
                farm_path = os.path.join(config_path, f"farm_{farm_id}", "ei")
            geojson_path = os.path.join(farm_path, "input.geojson")
            farm_results_dir = os.path.join(farm_path, "policy_bo_temp")
            os.makedirs(farm_results_dir, exist_ok=True)
            baseline_farm_json_path = os.path.join(farm_results_dir, f"farm_{farm_id}_baseline_EI_temp.geojson")
            if not os.path.exists(geojson_path): continue
            try:
                base_farm_gdf_processed, _ = main_run_pyomo(cfg, geojson_path, None, baseline_farm_json_path, neighbor_dist, exit_tol, penalty_coef)
                if base_farm_gdf_processed is not None and not base_farm_gdf_processed.empty:
                    current_farm_id = farm_id
                    if 'farm_id' not in base_farm_gdf_processed.columns: base_farm_gdf_processed['farm_id'] = current_farm_id
                    baseline_gdfs_config.append(base_farm_gdf_processed)
            except Exception as e: continue
        if not baseline_gdfs_config: baseline_metrics[config_id] = None; continue
        try:
            combined_baseline_gdf = pd.concat(baseline_gdfs_config, ignore_index=False)
            if combined_baseline_gdf.empty: baseline_metrics[config_id] = None; continue
            combined_baseline_geojson_path = os.path.join(config_results_dir, f"config_{config_id}_combined_baseline_temp.geojson")
            save_gdf_for_ec(combined_baseline_gdf, combined_baseline_geojson_path)
            base_ec_plots = parse_geojson(combined_baseline_geojson_path)
            if not base_ec_plots: raise ValueError(f"Parsed baseline plots empty Config {config_id}.")
            _, _, conn_val_optim, _, optimized_plot_npvs_dict = solve_connectivity_ilp(base_ec_plots, **ec_params_baseline)
            _, _, conn_val_repos, plot_repos_npv = solve_reposition_ilp(base_ec_plots, cfg.params, **ec_params_repos)
            farm_npvs_dict_for_this_config = defaultdict(float)
            farm_npvs_repos_dict_for_this_config = defaultdict(float)
            if isinstance(optimized_plot_npvs_dict, dict) and optimized_plot_npvs_dict:
                index_to_farm_id = {idx: p.get('farm_id', 'unknown_farm') for idx, p in enumerate(base_ec_plots)}
                for plot_key, plot_npv in optimized_plot_npvs_dict.items():
                    farm_id = index_to_farm_id.get(plot_key)
                    if farm_id is not None and farm_id != 'unknown_farm':
                        farm_npvs_dict_for_this_config[farm_id] += plot_npv
                        farm_npvs_repos_dict_for_this_config[farm_id] += plot_repos_npv[plot_key]
            if not farm_npvs_dict_for_this_config: print(f"Warning: No farm NPVs aggregated for baseline config {config_id}")
            baseline_metrics[config_id] = {'connectivity': conn_val_optim, 'repos_connectivity': conn_val_repos,
                                           'farm_npvs': dict(farm_npvs_dict_for_this_config), 'farm_npvs_repos': dict(farm_npvs_repos_dict_for_this_config)}
        except Exception as e:
            print(f"  Error during Baseline Processing for Config {config_id}: {e}")
            baseline_metrics[config_id] = None
    print("." * 80)
    print(f"Finished calculating baseline metrics for {len(baseline_metrics)} configurations.")
    successful_count = sum(1 for v in baseline_metrics.values() if v is not None)
    print(f"Successfully calculated metrics for {successful_count} configurations.")
    print("-" * 30)
    return baseline_metrics


# --- Global cache for objective function --- <<< Re-enabled
objective_cache = {}


def evaluate_policy_for_bo_avg_config_obj(policy_param_list, param_order, crop_param_indices, base_cfg,
                                          all_config_ids, # List of all possible config IDs
                                          n_samples,      # Number of configs to sample
                                          baseline_metrics, # Dict of precalculated baseline values PER CONFIG
                                          base_farm_dir,
                                          neighbor_dist, exit_tol, penalty_coef,
                                          ec_params_repos,
                                          objective_strategy, weight_conn, weight_npv, max_budget):

    global objective_cache # Use cache

    policy_params_tuple = tuple(policy_param_list)
    if policy_params_tuple in objective_cache:
        return objective_cache[policy_params_tuple] # Return cached average objective

    # --- 1. Reconstruct policy_params dictionary ---
    policy_params = {'subsidy': {}, 'payment': {}, 'mandate': {}, 'eco_premium': {}}
    crop_factors = {}; maint_subsidy_margin = 0.0; maint_subsidy_habitat = 0.0
    if len(policy_param_list) != len(param_order): return 1e13
    for i, param_name in enumerate(param_order):
        value = policy_param_list[i]
        if param_name == 'adj_hab_factor_margin': policy_params['subsidy']['adj_hab_factor_margin'] = value
        elif param_name == 'adj_hab_factor_habitat': policy_params['subsidy']['adj_hab_factor_habitat'] = value
        elif param_name == 'maint_subsidy_factor_margin': policy_params['subsidy']['maint_factor_margin'] = value
        elif param_name == 'maint_subsidy_factor_habitat': policy_params['subsidy']['maint_factor_habitat'] = value
        elif param_name == 'hab_per_ha': policy_params['payment']['hab_per_ha'] = value
        elif param_name == 'min_total_hab_area': policy_params['mandate']['min_total_hab_area'] = value
        elif param_name == 'min_margin_frac_adj_hab': policy_params['mandate']['min_margin_frac_adj_hab'] = value
        elif param_name.startswith('eco_premium_factor_'): crop_factors[param_name.replace('eco_premium_factor_', '')] = value
    policy_params['eco_premium']['crop_factors'] = crop_factors

    print(f"\nBO Evaluating Policy (on {n_samples} Samples):")

    # --- 2. Create a deep copy of config ONCE and modify costs/prices ---
    cfg_copy = copy.deepcopy(base_cfg)
    try: # Apply maintenance subsidies
        cfg_copy.params['costs']['margin']['maintenance'] = base_cfg.params['costs']['margin']['maintenance'] * (1.0 - policy_params['subsidy']['maint_factor_margin'])
        cfg_copy.params['costs']['habitat']['maintenance'] = base_cfg.params['costs']['habitat']['maintenance'] * (1.0 - policy_params['subsidy']['maint_factor_habitat'])
    except Exception as e: pass
    if 'eco_premium' in policy_params and 'crop_factors' in policy_params['eco_premium']: # Apply eco-premium
        if isinstance(cfg_copy.params.get('crops'), dict):
            base_prices = {}
            for crop in policy_params['eco_premium']['crop_factors']:
                 if crop in cfg_copy.params['crops'] and isinstance(cfg_copy.params['crops'][crop], dict): base_prices[crop] = cfg_copy.params['crops'][crop].get('p_c', 0)
            for crop, factor in policy_params['eco_premium']['crop_factors'].items():
                 if crop in base_prices: cfg_copy.params['crops'][crop]['p_c'] = base_prices[crop] * factor
        else: pass


    # --- 3. Sample configurations ---
    if not all_config_ids: return 1e14 # Error
    if len(all_config_ids) <= n_samples:
        sampled_config_ids = all_config_ids
    else:
        sampled_config_ids = random.sample(all_config_ids, n_samples)
    print(f"  Running evaluation on {len(sampled_config_ids)} sampled configs.")

    # --- Accumulator for per-config objective scores ---
    per_config_objective_values = []
    per_config_total_policy_costs = []
    budget_exceeded_in_any_sample = False
    policy_id_str = f"bo_eval_{random.randint(0, 99999):05d}" # Unique ID for temp files per BO step

    # --- 4. Loop over sampled configurations ---
    for config_id in sampled_config_ids:
        print(f"    Processing Config {config_id}...", end='\r')

        # --- 4a. Retrieve baseline metrics for this config ---
        config_baseline_data = baseline_metrics.get(config_id)
        if config_baseline_data is None:
            print(f"  Warn: Baseline data missing for config {config_id}. Skipping.")
            continue # Skip this config

        target_connectivity_score = config_baseline_data['connectivity']
        target_baseline_farm_npvs_dict = config_baseline_data.get('farm_npvs', {})
        if not target_baseline_farm_npvs_dict and objective_strategy == 'scalarized_pareto' and weight_npv > 0:
             print(f"  Warn: Baseline farm NPV data missing for config {config_id}. Skipping (NPV weighted).")
             continue # Skip if NPV matters but baseline is missing


        # --- 4b. Run Policy EI & EC for this config ---
        config_path = os.path.join(base_farm_dir, f"config_{config_id}")
        config_policy_results_dir = os.path.join(config_path, "policy_bo_temp")
        os.makedirs(config_policy_results_dir, exist_ok=True)
        policy_gdfs_config = []
        config_policy_farm_npvs_dict = {} # Store policy NPV per farm {farm_id: npv}
        config_total_policy_cost = 0.0

        num_farms = sum(1 for item in os.listdir(config_path) if os.path.isdir(os.path.join(config_path, item)) and item.startswith("farm_"))
        if num_farms == 0: continue

        # Run Policy EI
        for farm_id_loop in range(1, num_farms + 1):
            farm_id = farm_id_loop # Assuming farm_1, farm_2...
            if farm_mode == "syn_farms":
                farm_path = os.path.join(config_path, f"farm_{farm_id}")
            else:
                farm_path = os.path.join(config_path, f"farm_{farm_id}", "ei")
            geojson_path = os.path.join(farm_path, "input.geojson")
            farm_results_dir = os.path.join(farm_path, "policy_bo_temp")
            os.makedirs(farm_results_dir, exist_ok=True)
            if not os.path.exists(geojson_path): continue
            try:
                policy_farm_gdf, policy_farm_data, _, farm_policy_costs = run_single_optimization_policy(cfg_copy, geojson_path, neighbor_dist, exit_tol, penalty_coef, policy_params)
                if policy_farm_gdf is not None and not policy_farm_gdf.empty:
                    current_farm_id = farm_id
                    if 'farm_id' not in policy_farm_gdf.columns: policy_farm_gdf['farm_id'] = current_farm_id
                    policy_gdfs_config.append(policy_farm_gdf)
                    config_policy_farm_npvs_dict[current_farm_id] = policy_farm_data.get('policy_npv', 0.0) if policy_farm_data else 0.0

                    if farm_policy_costs and 'total_policy_cost_npv' in farm_policy_costs:
                        config_total_policy_cost += farm_policy_costs['total_policy_cost_npv']
                    else:
                        # Handle case where cost calculation failed for a farm
                        print(f"  Warn: Could not get policy costs for farm {farm_id} in config {config_id}.")

            except Exception as e: pass # Ignore farm EI errors, proceed

        # --- Budget Check for this Configuration ---
        print(f"    Config {config_id} Total Policy Cost (NPV): {config_total_policy_cost:.2f}")  # Log cost
        per_config_total_policy_costs.append(config_total_policy_cost)  # Store for potential averaging later

        # Run EC Repositioning
        policy_conn_for_this_config = None
        if not policy_gdfs_config: continue # Skip if all EI failed

        try:
            combined_policy_gdf = pd.concat(policy_gdfs_config, ignore_index=True)
            if combined_policy_gdf.empty: raise ValueError("Combined GDF empty.")
            temp_policy_geojson = os.path.join(config_policy_results_dir, f"config_{config_id}_combined_policy_{policy_id_str}.geojson")
            save_gdf_for_ec(combined_policy_gdf, temp_policy_geojson)
            policy_ec_plots = parse_geojson(temp_policy_geojson)
            if not policy_ec_plots: raise ValueError("Parsed plots empty.")
            _, _, conn_val_policy, _ = solve_reposition_ilp(policy_ec_plots, cfg_copy.params, **ec_params_repos)
            policy_conn_for_this_config = conn_val_policy
        except Exception as e:
            print(f"  Warn: EC Repos failed for Config {config_id}: {e}")
            # policy_conn_for_this_config remains None


        # --- 4c. Calculate Objective Components for THIS config ---
        if policy_conn_for_this_config is None and (objective_strategy == 'connectivity_only' or (objective_strategy == 'scalarized_pareto' and weight_conn > 0)):
            print(f"  Skipping config {config_id} objective: Connectivity failed.")
            continue # Skip config if connectivity failed but was needed

        conn_diff = 1e9 # Default penalty
        if policy_conn_for_this_config is not None:
             conn_diff = abs(policy_conn_for_this_config - target_connectivity_score)

        # Calculate mean absolute farm NPV diff for this config
        farm_npv_abs_diffs_this_config = []
        farms_processed_npv = 0
        if config_policy_farm_npvs_dict: # Check if we got any policy NPVs
             for farm_id, policy_npv in config_policy_farm_npvs_dict.items():
                 baseline_npv = target_baseline_farm_npvs_dict.get(farm_id)
                 if baseline_npv is not None:
                     farm_npv_abs_diffs_this_config.append(abs(policy_npv - baseline_npv))
                     farms_processed_npv += 1

        npv_diff = 1e9 # Default penalty
        if not farm_npv_abs_diffs_this_config and objective_strategy == 'scalarized_pareto' and weight_npv > 0 :
              print(f"  Skipping config {config_id} objective: No farm NPV diffs calculated.")
              continue # Skip config if NPV diff needed but failed
        elif farm_npv_abs_diffs_this_config: # Only calculate mean if list is not empty
             npv_diff = np.mean(farm_npv_abs_diffs_this_config)


        # --- 4d. Calculate weighted objective for THIS config ---
        config_objective = weight_conn * conn_diff + weight_npv * npv_diff
        per_config_objective_values.append(config_objective)
        # Optional: Log per-config results
        # print(f"    Config {config_id}: ConnDiff={conn_diff:.4f}, NPVDiff={npv_diff:.0f}, Obj={config_objective:.4f}")


    # --- 5. Calculate Average Objective Score ---
    print("") # Newline after progress indicator
    avg_policy_cost = np.mean(per_config_total_policy_costs) if per_config_total_policy_costs else 0.0
    if avg_policy_cost > max_budget:
        print(f"BUDGET EXCEEDED in at least one sample. Assigning high penalty objective.")
        final_objective = 1e15  # Very high penalty for budget violation
    elif not per_config_objective_values:
        print(f"BO Eval failed: Policy={policy_params} - No valid objective scores calculated across samples.")
        final_objective = 1e12  # High penalty for evaluation failure
    else:
        final_objective = np.mean(per_config_objective_values)  # Average the per-config objectives
        if np.isnan(final_objective): final_objective = 1e11

    print(
        f"  Average Objective ({objective_strategy}) over {len(per_config_objective_values)} valid configs: {final_objective:.6f}")
    print(
        f"  Average Policy Cost (NPV) over {len(per_config_total_policy_costs)} sampled configs: {avg_policy_cost:.2f}")
    print("-" * 20)

    objective_cache[policy_params_tuple] = final_objective # Cache the average objective
    return final_objective


# --- Helper Function to Re-evaluate Policy Performance ---
def evaluate_policy_performance_for_plotting(policy_params, param_order, crop_param_indices, base_cfg,
                                              all_config_ids, baseline_metrics, # Pass full baseline data
                                              base_farm_dir, neighbor_dist, exit_tol, penalty_coef,
                                              ec_params_repos):
    """
    Runs a given policy across ALL specified configurations to get stable average
    connectivity and average farm NPV for plotting.
    """
    print(f"Re-evaluating policy for plotting across {len(all_config_ids)} configs...")
    # 1. Reconstruct policy params (if passed as list) or use directly if dict
    if isinstance(policy_params, (list, tuple)):
         # Assumes policy_params is a list matching param_order
        policy_params_dict = {'subsidy': {}, 'payment': {}, 'mandate': {}, 'eco_premium': {}}
        crop_factors = {}; maint_subsidy_margin = 0.0; maint_subsidy_habitat = 0.0
        if len(policy_params) != len(param_order): raise ValueError("Length mismatch in policy params list")
        for i, param_name in enumerate(param_order):
            value = policy_params[i]
            if param_name == 'adj_hab_factor_margin': policy_params_dict['subsidy']['adj_hab_factor_margin'] = value
            elif param_name == 'adj_hab_factor_habitat': policy_params_dict['subsidy']['adj_hab_factor_habitat'] = value
            elif param_name == 'maint_subsidy_factor_margin': policy_params_dict['subsidy']['maint_factor_margin'] = value
            elif param_name == 'maint_subsidy_factor_habitat': policy_params_dict['subsidy']['maint_factor_habitat'] = value
            elif param_name == 'hab_per_ha': policy_params_dict['payment']['hab_per_ha'] = value
            elif param_name == 'min_total_hab_area': policy_params_dict['mandate']['min_total_hab_area'] = value
            elif param_name == 'min_margin_frac_adj_hab': policy_params_dict['mandate']['min_margin_frac_adj_hab'] = value
            elif param_name.startswith('eco_premium_factor_'): crop_factors[param_name.replace('eco_premium_factor_', '')] = value
        policy_params_dict['eco_premium']['crop_factors'] = crop_factors

    # 2. Create modified config
    cfg_copy = copy.deepcopy(base_cfg)
    try:
        cfg_copy.params['costs']['margin']['maintenance'] = base_cfg.params['costs']['margin']['maintenance'] * (1.0 - policy_params_dict['subsidy']['maint_factor_margin'])
        cfg_copy.params['costs']['habitat']['maintenance'] = base_cfg.params['costs']['habitat']['maintenance'] * (1.0 - policy_params_dict['subsidy']['maint_factor_habitat'])
    except Exception: pass
    if 'eco_premium' in policy_params_dict and 'crop_factors' in policy_params_dict['eco_premium']:
        if isinstance(cfg_copy.params.get('crops'), dict):
            base_prices = {}
            for crop in policy_params_dict['eco_premium']['crop_factors']:
                 if crop in cfg_copy.params['crops'] and isinstance(cfg_copy.params['crops'][crop], dict): base_prices[crop] = cfg_copy.params['crops'][crop].get('p_c', 0)
            for crop, factor in policy_params_dict['eco_premium']['crop_factors'].items():
                 if crop in base_prices: cfg_copy.params['crops'][crop]['p_c'] = base_prices[crop] * factor
        else: pass

    # 3. Accumulators
    all_farm_npvs_for_policy = []
    all_conn_scores_for_policy = []
    all_conn_diffs_for_policy = []
    all_farm_npv_diffs_for_policy = []
    all_total_policy_costs_for_policy = []
    successful_configs_conn = 0
    successful_configs_npv = 0
    successful_configs_cost = 0

    # 4. Loop over ALL specified configurations
    for config_id in all_config_ids:
        print(f"    Re-evaluating Config {config_id}...", end='\r')
        # Retrieve baseline (optional, mainly needed for diff calc if desired)
        # config_baseline_data = baseline_metrics.get(config_id)
        # if config_baseline_data is None: continue # Skip if baseline missing

        config_baseline_data = baseline_metrics.get(config_id)
        baseline_conn_repos = config_baseline_data.get('connectivity')
        baseline_farm_npvs_repos_dict = config_baseline_data.get('farm_npvs')  # Dict {farm_id: npv}

        config_path = os.path.join(base_farm_dir, f"config_{config_id}")
        config_policy_results_dir = os.path.join(config_path, "policy_plotting_temp") # Use different temp dir
        os.makedirs(config_policy_results_dir, exist_ok=True)
        policy_gdfs_config = []
        config_policy_farm_npvs_dict = {} # Collect {farm_id: npv} for this config
        config_total_policy_cost = 0.0  # <<< Cost for THIS config
        config_cost_calculated = False

        num_farms = sum(1 for item in os.listdir(config_path) if os.path.isdir(os.path.join(config_path, item)) and item.startswith("farm_"))
        if num_farms == 0: continue

        # Run Policy EI
        farms_processed_npv_this_config = 0
        for farm_id_loop in range(1, num_farms + 1):
            farm_id = farm_id_loop
            if farm_mode == "syn_farms":
                farm_path = os.path.join(config_path, f"farm_{farm_id}")
            else:
                farm_path = os.path.join(config_path, f"farm_{farm_id}", "ei")
            geojson_path = os.path.join(farm_path, "input.geojson")
            farm_results_dir = os.path.join(farm_path, "policy_plotting_temp")
            os.makedirs(farm_results_dir, exist_ok=True)
            if not os.path.exists(geojson_path): continue
            try:
                policy_farm_gdf, policy_farm_data, _, farm_policy_costs = run_single_optimization_policy(cfg_copy, geojson_path, neighbor_dist, exit_tol, penalty_coef, policy_params_dict)
                if policy_farm_gdf is not None and not policy_farm_gdf.empty:
                    current_farm_id = farm_id
                    if 'farm_id' not in policy_farm_gdf.columns: policy_farm_gdf['farm_id'] = current_farm_id
                    policy_gdfs_config.append(policy_farm_gdf)
                    policy_npv = policy_farm_data.get('policy_npv', 0.0) if policy_farm_data else 0.0
                    config_policy_farm_npvs_dict[current_farm_id] = policy_npv

                    if baseline_farm_npvs_repos_dict is not None:
                        baseline_npv = baseline_farm_npvs_repos_dict.get(current_farm_id)
                        if baseline_npv is not None:
                            all_farm_npv_diffs_for_policy.append(abs(policy_npv - baseline_npv))
                            farms_processed_npv_this_config += 1
                        else:
                            pass

                    if farm_policy_costs and 'total_policy_cost_npv' in farm_policy_costs:
                        cost_val = farm_policy_costs['total_policy_cost_npv']
                        if pd.notna(cost_val):
                            config_total_policy_cost += cost_val
                            config_cost_calculated = True  # Mark cost as successfully calculated for this config
                    else:
                        pass
            except Exception: pass

        # Collect all farm NPVs for averaging later
        if config_policy_farm_npvs_dict:
             all_farm_npvs_for_policy.extend(config_policy_farm_npvs_dict.values())
             successful_configs_npv += 1 # Count configs where NPVs were generated

        if config_cost_calculated:
            all_total_policy_costs_for_policy.append(config_total_policy_cost)
            successful_configs_cost += 1

        # Run EC Repositioning for Connectivity
        if not policy_gdfs_config: continue
        try:
            combined_policy_gdf = pd.concat(policy_gdfs_config, ignore_index=True)
            if combined_policy_gdf.empty: raise ValueError("Combined GDF empty.")
            policy_id_str = f"plot_eval_{random.randint(0, 99999):05d}_{config_id}"
            temp_policy_geojson = os.path.join(config_policy_results_dir, f"config_{config_id}_combined_policy_{policy_id_str}.geojson")
            save_gdf_for_ec(combined_policy_gdf, temp_policy_geojson)
            policy_ec_plots = parse_geojson(temp_policy_geojson)
            if not policy_ec_plots: raise ValueError("Parsed plots empty.")
            _, _, conn_val_policy, _ = solve_reposition_ilp(policy_ec_plots, cfg_copy.params, **ec_params_repos)
            all_conn_scores_for_policy.append(conn_val_policy)

            if baseline_conn_repos is not None and pd.notna(baseline_conn_repos):
                all_conn_diffs_for_policy.append(abs(conn_val_policy - baseline_conn_repos))
                successful_configs_conn += 1  # Count successful conn diff calcs
            else:
                pass

        except Exception as e:
            print(f"  Warn: Plotting Re-eval EC Repos failed Config {config_id}: {e}")

    print("." * 80) # Clear progress line

    # 5. Calculate overall averages
    avg_policy_conn = np.mean(all_conn_scores_for_policy) if all_conn_scores_for_policy else None
    avg_policy_farm_npv = np.mean(all_farm_npvs_for_policy) if all_farm_npvs_for_policy else None
    avg_conn_diff = np.mean(all_conn_diffs_for_policy) if all_conn_diffs_for_policy else None
    avg_npv_diff = np.mean(all_farm_npv_diffs_for_policy) if all_farm_npv_diffs_for_policy else None
    avg_policy_cost = np.nanmean(
        all_total_policy_costs_for_policy) if all_total_policy_costs_for_policy else None

    # --- Print results safely --- <<< CORRECTED PRINT
    conn_log = f"{avg_policy_conn:.4f}" if avg_policy_conn is not None else "N/A"
    npv_log = f"{avg_policy_farm_npv:.0f}" if avg_policy_farm_npv is not None else "N/A"
    cost_log = f"{avg_policy_cost:.0f}" if pd.notna(avg_policy_cost) else "N/A"  # <<< Log cost
    print(
        f"Re-evaluation complete: AvgConn={conn_log} ({successful_configs_conn} configs), AvgFarmNPV={npv_log} ({successful_configs_npv} configs), AvgPolicyCost={cost_log} ({successful_configs_cost} configs)")

    return avg_policy_conn, avg_policy_farm_npv, avg_conn_diff, avg_npv_diff, avg_policy_cost


def visualize_bo_policy_comparison(baseline_avg_conn, baseline_avg_repos_conn, baseline_avg_farm_npv, baseline_avg_farm_npv_repos,
                                   top_policies_results, output_dir):
    print("--- Generating BO Policy Comparison Plot (Simplified Legend) ---")
    plt.figure(figsize=(12, 8))
    plotted_policies = False
    policy_data_points = [] # Store (npv, conn, id) for annotation

    # 1. Plot Baseline Average Points (remains the same)
    baseline_labels_handles = []
    baseline_plotted = False
    if pd.notna(baseline_avg_farm_npv) and pd.notna(baseline_avg_conn):
        plt.scatter(baseline_avg_farm_npv, baseline_avg_conn, color='green', marker='*', s=300, label='Baseline - Optimized', zorder=10) # Label here for legend
        plt.axhline(baseline_avg_conn, color='grey', linestyle='--', linewidth=0.8, zorder=1)
        plt.axvline(baseline_avg_farm_npv, color='grey', linestyle='--', linewidth=0.8, zorder=1)
        baseline_plotted = True
    if pd.notna(baseline_avg_farm_npv_repos) and pd.notna(baseline_avg_repos_conn):
        plt.scatter(baseline_avg_farm_npv_repos, baseline_avg_repos_conn, color='blue', marker='*', s=300, label='Baseline - Repositioned', zorder=10) # Label here for legend
        plt.axhline(baseline_avg_repos_conn, color='grey', linestyle='--', linewidth=0.8, zorder=1) # Use repos conn line? Or keep optim? Keep optim for reference.
        plt.axvline(baseline_avg_farm_npv_repos, color='grey', linestyle='--', linewidth=0.8, zorder=1)
        baseline_plotted = True


    # 2. Extract and Plot Top BO Policies (WITHOUT individual labels initially)
    valid_policies = []
    for idx, policy_data in enumerate(top_policies_results):
        avg_npv = policy_data.get('avg_farm_npv')
        avg_conn = policy_data.get('avg_conn')
        policy_id = policy_data.get('policy_id')
        if pd.notna(avg_npv) and pd.notna(avg_conn):
            valid_policies.append({'npv': avg_npv, 'conn': avg_conn, 'id': policy_id, 'original_index': idx})
        else:
            print(f"Warning: Skipping plot point for {policy_id} due to missing data (Avg NPV={avg_npv}, Avg Conn={avg_conn}).")

    if not valid_policies and not baseline_plotted:
        print("Skipping plot generation - no valid data points.")
        plt.close()
        return

    if valid_policies:
        df = pd.DataFrame(valid_policies)
        cmap = plt.get_cmap('viridis')
        colors = [cmap(p['original_index'] / max(1, len(top_policies_results) - 1)) for p in valid_policies]

        # Plot all policy points WITHOUT individual labels
        plt.scatter(df['npv'], df['conn'], s=150, c=colors, alpha=0.9, zorder=5)
        plotted_policies = True

        # 3. Annotate Min/Max Y-axis points
        min_conn_idx = df['conn'].idxmin()
        max_conn_idx = df['conn'].idxmax()
        min_point = df.loc[min_conn_idx]
        max_point = df.loc[max_conn_idx]

        min_color = cmap(min_point['original_index'] / max(1, len(top_policies_results) - 1))
        max_color = cmap(max_point['original_index'] / max(1, len(top_policies_results) - 1))

        #plt.text(min_point['npv'] * 1.01, min_point['conn'], "Policy - Lowest Conn", fontsize=9, va='center', zorder=6)
        #plt.text(max_point['npv'] * 1.01, max_point['conn'], "Policy - Highest Conn", fontsize=9, va='center', zorder=6)

        min_conn_policy_handle = mlines.Line2D([], [], color=min_color, marker='o', markersize=10,
                                               linestyle='None', label="Policy - Lowest Conn")
        max_conn_policy_handle = mlines.Line2D([], [], color=max_color, marker='o', markersize=10,
                                               linestyle='None', label="Policy - Highest Conn")

        # Optional: Highlight min/max points if desired (e.g., with different markers)


    # 4. Configure Plot Appearance
    plt.xlabel("Average Farm NPV")
    plt.ylabel("Average Connectivity Score")
    #plt.title("Performance of Top BO Policies vs. Baseline Average")
    #plt.grid(True, linestyle=':', alpha=0.6)

    # 5. Create Simplified Legend
    # Get handles/labels from baseline points (already labeled)
    handles, labels = plt.gca().get_legend_handles_labels()
    if min_conn_policy_handle:
        handles.append(min_conn_policy_handle)
        labels.append("Policy - Lowest Conn")
    if max_conn_policy_handle:
        handles.append(max_conn_policy_handle)
        labels.append("Policy - Highest Conn")


    # Add a generic entry for the BO policies if any were plotted
    if plotted_policies:
         # Create a proxy artist for the legend
         bo_policy_marker = plt.Line2D([0], [0], marker='o', color='grey', linestyle='None', # Use a neutral color/marker
                                       markersize=10, label='Top BO Policies')
         handles.append(bo_policy_marker)
         labels.append('Top BO Policies')

    # Place legend outside plot
    plt.legend(handles=handles, labels=labels, title="Setting", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend

    # 6. Save Plot
    plot_filename = os.path.join(output_dir, f"bo_top_policies_vs_baseline.svg") # Use strategy in name
    try:
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f"Comparison plot saved to: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot {plot_filename}: {e}")
    plt.close()


def visualize_bo_policy_differences(top_policies_results, output_dir): # Pass strategy
    print("--- Generating BO Policy Difference Plot (Simplified Legend) ---")
    plt.figure(figsize=(10, 7))
    plotted_policies = False

    # 1. Extract and Plot Top BO Policy Differences (WITHOUT labels initially)
    valid_policies = []
    for idx, policy_data in enumerate(top_policies_results):
        avg_npv_diff = policy_data.get('avg_npv_diff')
        avg_conn_diff = policy_data.get('avg_conn_diff')
        policy_id = policy_data.get('policy_id')
        if pd.notna(avg_npv_diff) and pd.notna(avg_conn_diff):
            valid_policies.append({'npv_diff': avg_npv_diff, 'conn_diff': avg_conn_diff, 'id': policy_id, 'original_index': idx})
        else:
            print(f"Warning: Skipping difference plot point for {policy_id} due to missing diff data (AvgNPVDiff={avg_npv_diff}, AvgConnDiff={avg_conn_diff}).")

    if not valid_policies:
        print("Skipping difference plot generation - no valid data points.")
        plt.close()
        return

    df = pd.DataFrame(valid_policies)
    cmap = plt.get_cmap('plasma')
    colors = [cmap(p['original_index'] / max(1, len(top_policies_results) - 1)) for p in valid_policies]

    plt.scatter(df['npv_diff'], df['conn_diff'], c=colors, s=150, alpha=0.9, zorder=5)
    plotted_policies = True

    # 2. Annotate Min/Max Y-axis points (Min Conn Diff is best)
    min_diff_idx = df['conn_diff'].idxmin()
    max_diff_idx = df['conn_diff'].idxmax()
    min_point = df.loc[min_diff_idx]
    max_point = df.loc[max_diff_idx]

    min_color = cmap(min_point['original_index'] / max(1, len(top_policies_results) - 1))
    max_color = cmap(max_point['original_index'] / max(1, len(top_policies_results) - 1))

    #plt.text(min_point['npv_diff'] * 1.01, min_point['conn_diff'], "Policy - Min Diff", fontsize=9, va='center', zorder=6)
    #plt.text(max_point['npv_diff'] * 1.01, max_point['conn_diff'], "Policy - Max Diff", fontsize=9, va='center', zorder=6)

    min_diff_policy_handle = mlines.Line2D([], [], color=min_color, marker='o', markersize=10, linestyle='None', label="Policy - Min Diff")
    max_diff_policy_handle = mlines.Line2D([], [], color=max_color, marker='o', markersize=10, linestyle='None', label="Policy - Max Diff")

    # 3. Configure Plot Appearance
    plt.xlabel("Average Absolute Farm  NPV Difference (vs Optimized Baseline)")
    plt.ylabel("Average Absolute Connectivity Difference (vs Optimized Baseline)")
    #plt.title("Policy Performance Difference from Baseline (Lower is Better)")
    #plt.grid(True, linestyle=':', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.7, zorder=1, label='No Difference')
    plt.axvline(0, color='black', linestyle='--', linewidth=0.7, zorder=1)


    # Add baseline reference point at (0,0)
    baseline_marker = plt.scatter(0, 0, color='red', marker='x', s=100, label='Baseline - Optimized', zorder=10)

    # 4. Create Simplified Legend
    handles, labels = plt.gca().get_legend_handles_labels() # Get handles from axhline and baseline scatter
    handles.append(min_diff_policy_handle)
    labels.append("Policy - Min Diff")
    handles.append(max_diff_policy_handle)
    labels.append("Policy - Max Diff")

    # Add generic entry for BO policies
    bo_policy_marker = plt.Line2D([0], [0], marker='o', color='grey', linestyle='None',
                                  markersize=10, label='Top BO Policies')
    handles.append(bo_policy_marker)
    labels.append('Top BO Policies')

    plt.legend(handles=handles, labels=labels, title="Setting", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout

    # 5. Save Plot
    plot_filename = os.path.join(output_dir, f"bo_top_policies_differences.svg") # Use strategy
    try:
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f"Policy difference plot saved to: {plot_filename}")
    except Exception as e:
        print(f"Error saving difference plot {plot_filename}: {e}")
    plt.close()


def visualize_conn_vs_cost(baseline_avg_conn, baseline_avg_repos_conn, top_policies_results, output_dir):
    """ Plots Average Connectivity Score vs Average Policy Cost """
    print("--- Generating Connectivity vs. Policy Cost Plot ---")
    plt.figure(figsize=(10, 7))
    plotted_policies = False
    policy_data_points = [] # Store (cost, conn, id) for annotation

    baseline_plotted = False
    plt.axhline(baseline_avg_conn, color='green', linestyle='--', linewidth=0.8, label="Baseline - Optimized", zorder=1)
    plt.axhline(baseline_avg_repos_conn, color='blue', linestyle='--', linewidth=0.8, label="Baseline - Repositioned", zorder=1)


    # 1. Extract Data and Filter Nones/NaNs
    valid_policies = []
    for policy_data in top_policies_results:
        cost = policy_data.get('avg_policy_cost')
        conn = policy_data.get('avg_conn')
        policy_id = policy_data.get('policy_id', 'Unknown')
        if pd.notna(cost) and pd.notna(conn):
            valid_policies.append({'cost': cost, 'conn': conn, 'id': policy_id})
        else:
             print(f"Warning: Skipping Conn vs Cost plot point for {policy_id} due to missing data (Cost={cost}, Conn={conn}).")

    if not valid_policies:
        print("Skipping Conn vs Cost plot generation - no valid data points.")
        plt.close()
        return

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(valid_policies)

    # 2. Plot all valid points
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(df))) # Color points (optional, could use single color)

    plt.scatter(df['cost'], df['conn'], c=colors, s=100, alpha=0.8, zorder=5)
    plotted_policies = True

    # 3. Identify and Annotate Min/Max Y-axis points
    if plotted_policies:
        min_conn_idx = df['conn'].idxmin()
        max_conn_idx = df['conn'].idxmax()

        min_point = df.loc[min_conn_idx]
        max_point = df.loc[max_conn_idx]

        # Add text annotation near min/max points
        #plt.text(min_point['cost'] * 1.01, min_point['conn'], "Policy - Lowest Conn", fontsize=9, va='center', zorder=6)
        #plt.text(max_point['cost'] * 1.01, max_point['conn'], "Policy - Highest Conn", fontsize=9, va='center', zorder=6)

        # Highlight min/max points (optional)
        plt.scatter(min_point['cost'], min_point['conn'], c='red', s=150, marker='x', zorder=10)
        plt.scatter(max_point['cost'], max_point['conn'], c='blue', s=150, marker='*', zorder=10)


    # 4. Configure Plot Appearance
    plt.xlabel("Average Policy Cost (NPV of Government Expenditures)")
    plt.ylabel("Average Connectivity Score (IIC)")
    #plt.title("Policy Performance: Connectivity vs. Cost")
    #plt.grid(True, linestyle=':', alpha=0.6)

    # Format x-axis to avoid scientific notation if costs are large
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    plt.xticks(rotation=30)

    # 5. Create Minimal Legend (if needed, e.g., for highlighted points)
    # Example: Legend for highlighted min/max markers
    handles, labels = plt.gca().get_legend_handles_labels()
    bo_policy_marker = plt.Line2D([0], [0], marker='o', color='grey', linestyle='None',  # Use a neutral color/marker
                                  markersize=10, label='Top BO Policies')
    handles.append(bo_policy_marker)
    labels.append('Top BO Policies')

    #high_con = plt.Line2D([0], [0], marker='*', color='blue', linestyle='None', markersize=10, label='Policy - Highest Conn')
    #low_con = plt.Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=10, label='Policy - Lowest Conn')

    #handles.append(high_con)
    #labels.append('Policy - Highest Conn')
    #handles.append(low_con)
    #labels.append('Policy - Lowest Conn')

    plt.legend(handles=handles, labels=labels, title="Policies", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    # 6. Save Plot
    plot_filename = os.path.join(output_dir, f"bo_conn_vs_cost.svg")
    try:
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f"Connectivity vs Cost plot saved to: {plot_filename}")
    except Exception as e:
        print(f"Error saving Conn vs Cost plot {plot_filename}: {e}")
    plt.close()


def visualize_conn_diff_vs_cost(baseline_avg_conn, baseline_avg_repos_conn, top_policies_results, output_dir):
    """ Plots Average Absolute Connectivity Difference vs Average Policy Cost """
    print("--- Generating Connectivity Difference vs. Policy Cost Plot ---")
    plt.figure(figsize=(10, 7))
    plotted_policies = False
    policy_data_points = [] # Store (cost, conn_diff, id) for annotation

    #plt.axhline(baseline_avg_conn, color='green', linestyle='--', linewidth=0.8, label="Baseline - Optimized", zorder=1)
    #plt.axhline(baseline_avg_repos_conn, color='blue', linestyle='--', linewidth=0.8, label="Baseline - Repositioned",
    #            zorder=1)

    # 1. Extract Data and Filter Nones/NaNs
    valid_policies = []
    for policy_data in top_policies_results:
        cost = policy_data.get('avg_policy_cost')
        conn_diff = policy_data.get('avg_conn_diff') # Use the difference
        policy_id = policy_data.get('policy_id', 'Unknown')
        if pd.notna(cost) and pd.notna(conn_diff):
             valid_policies.append({'cost': cost, 'conn_diff': conn_diff, 'id': policy_id})
        else:
             print(f"Warning: Skipping Conn Diff vs Cost plot point for {policy_id} due to missing data (Cost={cost}, ConnDiff={conn_diff}).")

    if not valid_policies:
        print("Skipping Conn Diff vs Cost plot generation - no valid data points.")
        plt.close()
        return

    # Convert to DataFrame
    df = pd.DataFrame(valid_policies)

    # 2. Plot all valid points
    cmap = plt.get_cmap('plasma') # Different colormap
    colors = cmap(np.linspace(0, 1, len(df)))

    plt.scatter(df['cost'], df['conn_diff'], c=colors, s=100, alpha=0.8, zorder=5)
    plotted_policies = True

    # 3. Identify and Annotate Min/Max Y-axis points (Min Conn Diff is best)
    if plotted_policies:
        min_diff_idx = df['conn_diff'].idxmin()
        max_diff_idx = df['conn_diff'].idxmax()

        min_point = df.loc[min_diff_idx]
        max_point = df.loc[max_diff_idx]

        #plt.text(min_point['cost'] * 1.01, min_point['conn_diff'], "Policy - Min Diff", fontsize=9, va='center', zorder=6)
        #plt.text(max_point['cost'] * 1.01, max_point['conn_diff'], "Policy - Max Diff", fontsize=9, va='center', zorder=6)

        # Highlight points (optional)
        plt.scatter(min_point['cost'], min_point['conn_diff'], c='green', s=150, marker='*', zorder=10) # Best
        plt.scatter(max_point['cost'], max_point['conn_diff'], c='orange', s=150, marker='x', zorder=10) # Worst

    # 4. Configure Plot Appearance
    plt.xlabel("Average Policy Cost (NPV of Government Expenditures)")
    plt.ylabel("Average Absolute Connectivity Difference (vs Baseline)")
    #plt.title("Policy Performance: Connectivity Difference vs. Cost")
    #plt.grid(True, linestyle=':', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.7, zorder=1, label='No Difference') # Target line at Y=0

    # Format x-axis
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    plt.xticks(rotation=30)


    # 5. Create Minimal Legend
    legend_elements = [
         plt.Line2D([0], [0], marker='*', color='green', linestyle='None', markersize=10, label='Policy - Lowest Diff'),
         plt.Line2D([0], [0], marker='x', color='orange', linestyle='None', markersize=10, label='Policy - Highest Diff'),
         plt.Line2D([0], [0], color='black', linestyle='--', linewidth=0.7, label='No Difference')
        ]
    plt.legend(handles=legend_elements, title="Policies", loc='best')

    plt.tight_layout()

    # 6. Save Plot
    plot_filename = os.path.join(output_dir, f"bo_conn_diff_vs_cost.svg")
    try:
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f"Connectivity Difference vs Cost plot saved to: {plot_filename}")
    except Exception as e:
        print(f"Error saving Conn Diff vs Cost plot {plot_filename}: {e}")
    plt.close()


def delete_temp_files(all_config_ids_global):
    for config_id in all_config_ids_global:
        config_path = os.path.join(base_farm_dir_global, f"config_{config_id}")
        num_farms = sum(1 for item in os.listdir(config_path) if
                        os.path.isdir(os.path.join(config_path, item)) and item.startswith("farm_"))

        temp_dir = os.path.join(config_path, "policy_bo_temp")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        temp_dir = os.path.join(config_path, "policy_plotting_temp")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        for farm_id in range(1, num_farms + 1):
            farm_path = os.path.join(config_path, f"farm_{farm_id}")
            temp_dir = os.path.join(farm_path, "policy_bo_temp")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

            temp_dir = os.path.join(farm_path, "ei", "policy_bo_temp")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

            temp_dir = os.path.join(farm_path, "ei", "policy_plotting_temp")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

if __name__ == "__main__":
    print("--- Bayesian Optimization Script Start ---")
    start_time_main = time.time()
    OBJECTIVE_STRATEGY = 'scalarized_pareto'
    WEIGHT_CONNECTIVITY = 1
    WEIGHT_NPV = 0
    N_CONFIG_SAMPLES = 20
    N_INITIAL_POINTS_BO = 15
    N_CALLS_BO = 100
    farm_mode = "real_farms"
    MAX_POLICY_BUDGET_PER_CONFIG = 500000

    # 1. Setup Configuration and Paths
    base_cfg = Config()

    # --- Essential Parameters ---
    exit_tol_global = 1e-6
    neighbor_dist_global = 1000
    penalty_coef_global = 1e3
    if farm_mode == "syn_farms":
        base_farm_dir_global = os.path.join(base_cfg.disk_dir, "crop_inventory", "syn_farms", "mc")
    else:
        base_farm_dir_global = os.path.join(base_cfg.disk_dir, "crop_inventory", "farms_config_s")
    num_configs_iterable = range(1, 571)
    all_config_ids_global = list(num_configs_iterable)
    output_dir_main = os.path.join(base_cfg.disk_dir, "crop_inventory", "syn_farms", "plots", "policy")
    os.makedirs(output_dir_main, exist_ok=True)
    print(f"Processing {len(all_config_ids_global)} configurations from: {base_farm_dir_global}")
    print(f"Saving results/plots to: {output_dir_main}")
    print(f"Objective Strategy: {OBJECTIVE_STRATEGY}")
    if OBJECTIVE_STRATEGY == 'scalarized_pareto': print(f"  Weights: Conn={WEIGHT_CONNECTIVITY}, NPV={WEIGHT_NPV}")

    delete_temp_files(all_config_ids_global)

    # --- EC Parameters ---
    ec_params_baseline_global = { 'al_factor': 1e-9, 'neib_dist': 1000, 'exit_tol': 1e-6, 'params': base_cfg.params, 'connectivity_metric': 'IIC', 'max_loss_ratio': 0.2, 'adjacency_dist': 0.0, 'boundary_seg_count': 4, 'interior_cell_count': 4, 'margin_weight': 50}
    ec_params_repos_global = { 'adjacency_dist': 0.0, 'boundary_seg_count': 4, 'interior_cell_count': 4, 'connectivity_metric': 'IIC', 'al_factor': 1e-9, 'neib_dist': 1000}

    # 1b. Identify Unique Crop Labels
    print("\n--- Identifying Unique Crop Labels ---")
    unique_crops = get_unique_crop_labels(all_config_ids_global, base_farm_dir_global)

    # 2. Phase 1: Calculate Baseline Metrics PER Config
    print("\n--- Phase 1: Calculating Baseline Metrics per Config ---")
    baseline_metrics_global = calculate_baseline_metrics_per_config(
        all_config_ids_global, base_farm_dir_global, base_cfg, ec_params_baseline_global, ec_params_repos_global,
        neighbor_dist_global, exit_tol_global, penalty_coef_global
    )
    valid_config_ids_for_bo = [cid for cid, data in baseline_metrics_global.items() if data is not None and data.get('farm_npvs')]
    if not valid_config_ids_for_bo:
        print("Error: No valid baseline metrics could be calculated. Exiting.")
        exit()
    print(f"Proceeding with {len(valid_config_ids_for_bo)} configurations for BO sampling.")

    # Calculate OVERALL baseline averages for plotting
    all_baseline_conn = [m['connectivity'] for m in baseline_metrics_global.values() if m and pd.notna(m.get('connectivity'))]
    all_baseline_repos = [m['repos_connectivity'] for m in baseline_metrics_global.values() if m and pd.notna(m.get('repos_connectivity'))]
    all_baseline_farm_npvs_list = []
    all_baseline_farm_npvs_repos_list = []
    for m in baseline_metrics_global.values():
        if m and isinstance(m.get('farm_npvs'), dict):
            all_baseline_farm_npvs_list.extend(m['farm_npvs'].values())
            all_baseline_farm_npvs_repos_list.extend(m['farm_npvs_repos'].values())

    overall_avg_baseline_conn = np.mean(all_baseline_conn) if all_baseline_conn else None
    overall_avg_baseline_repos_conn = np.mean(all_baseline_repos) if all_baseline_repos else None
    overall_avg_baseline_farm_npv = np.mean(all_baseline_farm_npvs_list) if all_baseline_farm_npvs_list else None
    overall_avg_baseline_farm_npv_repos = np.mean(all_baseline_farm_npvs_repos_list) if all_baseline_farm_npvs_repos_list else None
    print(f"Overall Avg Baseline Connectivity: {overall_avg_baseline_conn:.4f}")
    print(f"Overall Avg Baseline Farm NPV:     {overall_avg_baseline_farm_npv:.0f}")

    # 3. Phase 2: Define Dynamic Search Space for BO
    print("\n--- Phase 2: Defining Search Space ---")
    # ... (Define search_space and param_order as before) ...
    search_space = []
    param_order = []
    search_space.extend([ Real(0.0, 0.5, name='adj_hab_factor_margin'), Real(0.0, 0.5, name='adj_hab_factor_habitat'), Real(0.0, 0.5, name='maint_subsidy_factor_margin'), Real(0.0, 0.5, name='maint_subsidy_factor_habitat'), Real(0, 150, name='hab_per_ha'), Real(0, 10, name='min_total_hab_area'), Real(0.0, 0.3, name='min_margin_frac_adj_hab') ])
    param_order.extend([ 'adj_hab_factor_margin', 'adj_hab_factor_habitat', 'maint_subsidy_factor_margin', 'maint_subsidy_factor_habitat', 'hab_per_ha', 'min_total_hab_area', 'min_margin_frac_adj_hab' ])
    crop_param_indices = {}
    # print("Adding eco-premium factors for crops:")
    for crop in unique_crops:
        param_name = f'eco_premium_factor_{crop}'
        search_space.append(Real(1.0, 1.3, name=param_name))
        param_order.append(param_name)
        crop_param_indices[crop] = len(param_order) - 1
        # print(f"  - {param_name} (Range: [1.0, 1.3])")
    print(f"Total search space dimensions: {len(search_space)}")


    # 4. Phase 4: Run Bayesian Optimization Loop
    print("\n--- Phase 4: Running Bayesian Optimization (Avg Config Obj Eval) ---")
    print(f"Configuration Sampling per Evaluation: {N_CONFIG_SAMPLES}")
    print(f"BO Settings: n_calls={N_CALLS_BO}, n_initial_points={N_INITIAL_POINTS_BO}")

    start_time_bo = time.time()

    # Use partial to pass fixed arguments
    objective_wrapper = partial(evaluate_policy_for_bo_avg_config_obj, # Use the correct objective fn
                                param_order=param_order, crop_param_indices=crop_param_indices,
                                base_cfg=base_cfg, all_config_ids=valid_config_ids_for_bo,
                                n_samples = N_CONFIG_SAMPLES,
                                baseline_metrics=baseline_metrics_global,
                                base_farm_dir=base_farm_dir_global,
                                neighbor_dist=neighbor_dist_global, exit_tol=exit_tol_global,
                                penalty_coef=penalty_coef_global, ec_params_repos=ec_params_repos_global,
                                objective_strategy=OBJECTIVE_STRATEGY, weight_conn=WEIGHT_CONNECTIVITY, weight_npv=WEIGHT_NPV,
                                max_budget=MAX_POLICY_BUDGET_PER_CONFIG)

    result = gp_minimize(
        func=objective_wrapper, dimensions=search_space,
        n_calls=N_CALLS_BO, n_initial_points=N_INITIAL_POINTS_BO,
        acq_func='EI', random_state=RANDOM_SEED, verbose=True
    )
    end_time_bo = time.time()
    print(f"Bayesian Optimization finished in {end_time_bo - start_time_bo:.2f} seconds.")


    # 5. Phase 5: Analyze and Report Results
    print("\n--- Phase 5: Analyzing Results ---")
    best_params_list = result.x
    best_objective_value = result.fun
    print(f"\nBest Average Objective Value Found ({OBJECTIVE_STRATEGY}): {best_objective_value:.6f}")

    # Reconstruct the best policy dictionary
    # ... (Reconstruction logic remains the same) ...
    best_policy_params = {'subsidy': {}, 'payment': {}, 'mandate': {}, 'eco_premium': {}}
    best_crop_factors = {}
    for i, param_name in enumerate(param_order):
        value = best_params_list[i]
        if param_name == 'adj_hab_factor_margin': best_policy_params['subsidy']['adj_hab_factor_margin'] = value
        elif param_name == 'adj_hab_factor_habitat': best_policy_params['subsidy']['adj_hab_factor_habitat'] = value
        elif param_name == 'maint_subsidy_factor_margin': best_policy_params['subsidy']['maint_factor_margin'] = value
        elif param_name == 'maint_subsidy_factor_habitat': best_policy_params['subsidy']['maint_factor_habitat'] = value
        elif param_name == 'hab_per_ha': best_policy_params['payment']['hab_per_ha'] = value
        elif param_name == 'min_total_hab_area': best_policy_params['mandate']['min_total_hab_area'] = value
        elif param_name == 'min_margin_frac_adj_hab': best_policy_params['mandate']['min_margin_frac_adj_hab'] = value
        elif param_name.startswith('eco_premium_factor_'): best_crop_factors[param_name.replace('eco_premium_factor_', '')] = value
    if 'eco_premium' in best_policy_params: best_policy_params['eco_premium']['crop_factors'] = best_crop_factors
    else: best_policy_params['eco_premium'] = {'crop_factors': best_crop_factors}
    if not best_crop_factors: best_policy_params.pop('eco_premium', None)
    if not best_policy_params.get('subsidy'): best_policy_params.pop('subsidy', None)
    if not best_policy_params.get('payment'): best_policy_params.pop('payment', None)
    if not best_policy_params.get('mandate'): best_policy_params.pop('mandate', None)

    print("\nBest Policy Parameters Found (Dictionary):")
    print(json.dumps(best_policy_params, indent=2, default=lambda x: round(x, 4) if isinstance(x, float) else x))

    # --- Find Top 5 Policies and Re-evaluate for Plotting ---
    print("\n--- Re-evaluating Top 5 Policies for Plotting ---")
    N_TOP_POLICIES = 50
    top_policies_results_for_plot = []

    if hasattr(result, 'x_iters') and hasattr(result, 'func_vals'):
        evaluations = list(zip(result.x_iters, result.func_vals))
        # Create dict to store best score for unique param vectors
        unique_evals = {}
        for params_list, score in evaluations:
            params_tuple = tuple(params_list)
            if params_tuple not in unique_evals or score < unique_evals[params_tuple]:
                unique_evals[params_tuple] = score

        # Sort unique evaluations by score (ascending)
        sorted_unique_evals = sorted(unique_evals.items(), key=lambda item: item[1])

        # Get top N unique parameter vectors
        top_n_params_list = [list(params_tuple) for params_tuple, score in sorted_unique_evals[:N_TOP_POLICIES]]

        # Re-evaluate each top policy across ALL valid configurations
        for i, params_list in enumerate(top_n_params_list):
            policy_id = f"BO Policy {i+1}"
            print(f"\nRe-evaluating {policy_id}...")
            # Note: evaluate_policy_performance_for_plotting needs access to many globals or pass them
            # It also needs param_order, crop_param_indices, base_cfg etc.
            avg_conn, avg_farm_npv, avg_conn_diff, avg_npv_diff, avg_policy_cost = evaluate_policy_performance_for_plotting(
                policy_params=params_list, # Pass the parameter list
                param_order=param_order,
                crop_param_indices=crop_param_indices,
                base_cfg=base_cfg,
                all_config_ids=valid_config_ids_for_bo, # Use valid configs
                baseline_metrics=baseline_metrics_global,
                base_farm_dir=base_farm_dir_global,
                neighbor_dist=neighbor_dist_global,
                exit_tol=exit_tol_global,
                penalty_coef=penalty_coef_global,
                ec_params_repos=ec_params_repos_global
            )

            # Reconstruct the policy dict just for storage/reference
            policy_dict_for_plot = {'subsidy': {}, 'payment': {}, 'mandate': {}, 'eco_premium': {}}
            cf = {}; mm = mh = 0.0
            # ... (condensed reconstruction - same logic as above) ...
            for j, pname in enumerate(param_order):
                v=params_list[j];n=pname
                if n=='adj_hab_factor_margin': policy_dict_for_plot['subsidy']['adj_hab_factor_margin']=v
                elif n=='adj_hab_factor_habitat': policy_dict_for_plot['subsidy']['adj_hab_factor_habitat']=v
                elif n=='maint_subsidy_factor_margin': mm=max(0.0,min(1.0,v))
                elif n=='maint_subsidy_factor_habitat': mh=max(0.0,min(1.0,v))
                elif n=='hab_per_ha': policy_dict_for_plot['payment']['hab_per_ha']=v
                elif n=='min_total_hab_area': policy_dict_for_plot['mandate']['min_total_hab_area']=v
                elif n=='min_margin_frac_adj_hab': policy_dict_for_plot['mandate']['min_margin_frac_adj_hab']=v
                elif n.startswith('eco_premium_factor_'): cf[n.replace('eco_premium_factor_','')]=v
            policy_dict_for_plot['eco_premium']['crop_factors']=cf; policy_dict_for_plot['subsidy']['maint_factor_margin']=mm; policy_dict_for_plot['subsidy']['maint_factor_habitat']=mh
            if not policy_dict_for_plot.get('eco_premium',{}).get('crop_factors'): policy_dict_for_plot.pop('eco_premium',None)
            if not policy_dict_for_plot.get('subsidy'): policy_dict_for_plot.pop('subsidy',None) # Add cleanup if needed
            if not policy_dict_for_plot.get('payment'): policy_dict_for_plot.pop('payment',None)
            if not policy_dict_for_plot.get('mandate'): policy_dict_for_plot.pop('mandate',None)


            top_policies_results_for_plot.append({
                'policy_id': policy_id,
                'avg_conn': avg_conn,
                'avg_farm_npv': avg_farm_npv,
                'avg_conn_diff': avg_conn_diff,
                'avg_npv_diff': avg_npv_diff,
                'avg_policy_cost': avg_policy_cost,
                'params_list': params_list,
                'params_dict': policy_dict_for_plot
            })
    else:
        print("Warning: Could not retrieve optimization history ('x_iters', 'func_vals') from result object. Skipping top policies plot.")


    # --- Visualize Top Policies vs Baseline ---
    if top_policies_results_for_plot:
        visualize_bo_policy_comparison(
            baseline_avg_conn=overall_avg_baseline_conn,
            baseline_avg_repos_conn=overall_avg_baseline_repos_conn,
            baseline_avg_farm_npv=overall_avg_baseline_farm_npv,
            baseline_avg_farm_npv_repos=overall_avg_baseline_farm_npv_repos,
            top_policies_results=top_policies_results_for_plot,
            output_dir=output_dir_main
        )

        visualize_bo_policy_differences(
            top_policies_results=top_policies_results_for_plot,
            output_dir=output_dir_main
        )

        visualize_conn_vs_cost(baseline_avg_conn=overall_avg_baseline_conn, baseline_avg_repos_conn=overall_avg_baseline_repos_conn,
                               top_policies_results=top_policies_results_for_plot, output_dir=output_dir_main)

        visualize_conn_diff_vs_cost(baseline_avg_conn=overall_avg_baseline_conn, baseline_avg_repos_conn=overall_avg_baseline_repos_conn,
                                    top_policies_results=top_policies_results_for_plot, output_dir=output_dir_main)
    else:
         print("Skipping top policies visualization as no top policies were evaluated.")


    # --- Save Final Summary ---
    print("\n--- Saving Final Summary ---")
    results_summary = {
        'objective_strategy': OBJECTIVE_STRATEGY,
        'weight_connectivity': WEIGHT_CONNECTIVITY if OBJECTIVE_STRATEGY == 'scalarized_pareto' else None,
        'weight_npv': WEIGHT_NPV if OBJECTIVE_STRATEGY == 'scalarized_pareto' else None,
        'max_policy_budget_per_config': MAX_POLICY_BUDGET_PER_CONFIG,
        'best_avg_objective_value_found': best_objective_value,
        'best_parameters_list': best_params_list,
        'best_policy_dict': best_policy_params, # Policy dict for the single best point
        'top_policies_evaluated_for_plot': top_policies_results_for_plot, # Add top N results
        'parameter_order': param_order,
        'overall_avg_baseline_connectivity': overall_avg_baseline_conn, # Store overall baseline avg
        'overall_avg_baseline_farm_npv': overall_avg_baseline_farm_npv, # Store overall baseline avg
        'unique_crops_optimized': unique_crops,
        'bo_type': 'average_of_config_objectives',
        'n_configs_sampled_per_eval': N_CONFIG_SAMPLES,
        'bo_n_calls': N_CALLS_BO,
        'bo_n_initial_points': N_INITIAL_POINTS_BO,
        'bo_duration_seconds': end_time_bo - start_time_bo,
        'total_script_duration_seconds': time.time() - start_time_main
        # Optionally save baseline_metrics_global if space allows and needed later
    }
    results_filename = os.path.join(output_dir_main, f"bo_results_avg_cfg_obj.json")
    try:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                # Handle other non-serializable types if necessary
                try:
                    return super(NpEncoder, self).default(obj)
                except TypeError:
                    return str(obj) # Fallback to string representation
        with open(results_filename, 'w') as f: json.dump(results_summary, f, indent=2, cls=NpEncoder)
        print(f"\nResults summary saved to: {results_filename}")
    except Exception as e: print(f"\nError saving results summary: {e}")

    # Optional: Plot convergence
    try:
        plot_convergence(result)
        #plt.title(f"BO Convergence ({OBJECTIVE_STRATEGY})")
        plot_filename = os.path.join(output_dir_main, f"bo_convergence_avg_cfg_obj.svg")
        plt.grid(False)
        plt.savefig(plot_filename)
        print(f"Convergence plot saved to: {plot_filename}")
        plt.close()
    except ImportError: print("\nInstall matplotlib for convergence plot.")
    except Exception as e: print(f"\nError generating convergence plot: {e}")

    print("\n--- Bayesian Optimization Script End ---")


    delete_temp_files(all_config_ids_global)



