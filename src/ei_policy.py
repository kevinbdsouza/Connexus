import os
import json
import time
import copy
import math
import itertools
import random
import numpy as np
import pandas as pd
from config import Config

# Geometry
import geopandas as gpd
from shapely.geometry import shape, Polygon, Point, LineString, MultiPolygon, MultiPoint
from shapely import unary_union, voronoi_polygons
from graph_connectivity import solve_reposition_ilp, solve_connectivity_ilp
from shapely import wkt as shapely_wkt
from scipy.spatial import KDTree

# Optimization
import pyomo.environ as pyo
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon  # EC plotting
from matplotlib.collections import PatchCollection  # EC plotting
import matplotlib.lines as mlines  # EC plotting


def precompute_inputs(farm_gdf, params, neighbor_dist=15.0):
    plot_ids_orig = list(farm_gdf['id'])  # Keep original IDs if needed

    id_to_idx = {pid: i for i, pid in enumerate(plot_ids_orig)}
    idx_to_id = {i: pid for i, pid in enumerate(plot_ids_orig)}

    plot_ids = list(range(len(plot_ids_orig)))  # Use 0-based indices internally
    n_plots = len(plot_ids)

    # Extract centroids safely
    centroids = []
    for i, row in farm_gdf.iterrows():
        centroids.append(row.geometry.centroid.coords[0])
    centroids = np.array(centroids)

    # Pairwise distances
    dx = centroids[:, 0, np.newaxis] - centroids[:, 0]
    dy = centroids[:, 1, np.newaxis] - centroids[:, 1]
    distances = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(distances, 0.0)  # Ensure self-distance is zero

    # Create neighbor sets based on threshold
    neighbors = {}
    for i in range(n_plots):
        # Find indices where distance is > 0 and < threshold
        neighbor_indices = np.where((distances[i, :] > 1e-9) & (distances[i, :] < neighbor_dist))[0]
        neighbors[i] = list(neighbor_indices)  # Store as list of indices

    # Basic columns
    yields_ = farm_gdf['yield'].fillna(0.0).values

    # Calculate areas, ensuring geometry is valid
    areas = np.zeros(n_plots, dtype=float)
    for i, geom in enumerate(farm_gdf.geometry):
        if geom and not geom.is_empty and geom.is_valid:
            areas[i] = geom.area / 10000  # Convert sqm to Ha
        else:
            print(f"Warning: Invalid or empty geometry for plot index {i}. Area set to 0.")
            areas[i] = 0.0

    plot_types = farm_gdf['type'].values
    if 'label' in farm_gdf.columns:
        plot_labels = farm_gdf['label'].fillna("Unknown").values  # Fill NaNs
    else:
        plot_labels = np.array(["Unknown"] * n_plots)

    # Time discounting
    r = params['r']
    T = params['t']
    t_arr = np.arange(1, T + 1)
    discount_factors = (1 + r) ** (-t_arr)

    # Precompute time-factor cache for each gamma/zeta
    gamma_values = set()
    zeta_values = set()  # Separate set for clarity
    if 'crops' in params:
        for crop_def in params['crops'].values():
            if isinstance(crop_def, dict):  # Ensure it's a dictionary
                for intervention in ['margin', 'habitat']:
                    if intervention in crop_def:
                        gamma_values.add(crop_def[intervention].get('gamma', 0.1))  # Use default if missing
                        zeta_values.add(crop_def[intervention].get('zeta', 0.1))  # Use default if missing

    time_factor_cache_gamma = {}
    for g in gamma_values:
        if g > 0:
            time_factor_cache_gamma[g] = (1 - np.exp(-g * t_arr))
        else:
            time_factor_cache_gamma[g] = np.zeros_like(t_arr)  # Handle g=0 case

    time_factor_cache_zeta = {}
    for z in zeta_values:
        if z > 0:
            time_factor_cache_zeta[z] = (1 - np.exp(-z * t_arr))
        else:
            time_factor_cache_zeta[z] = np.zeros_like(t_arr)

    # Combine caches for simplicity in the model function (or keep separate)
    time_factor_cache = {**time_factor_cache_gamma, **time_factor_cache_zeta}

    # --- Adjacency Calculation (for Policy Evaluation) ---
    is_adjacent_to_habitat = {}
    for i in range(n_plots):
        is_adjacent_to_habitat[i] = False
        if i in neighbors:
            for j in neighbors[i]:
                if 0 <= j < n_plots and plot_types[j] == 'hab_plots':
                    is_adjacent_to_habitat[i] = True
                    break

    # Pack everything in a dictionary
    farm_data = {
        'plot_ids': plot_ids,
        'id_to_idx': id_to_idx,
        'idx_to_id': idx_to_id,
        'distances': distances,
        'neighbors': neighbors,
        'yields': yields_,
        'areas': areas,
        'plot_types': plot_types,
        'plot_labels': plot_labels,
        'discount_factors': discount_factors,
        'time_factor_cache': time_factor_cache,
        'neighbor_dist': neighbor_dist,
        'params': params,
        'is_adjacent_to_habitat': is_adjacent_to_habitat,
        'n_plots': n_plots
    }
    return farm_data


def build_and_solve_pyomo_model(farm_data, params, penalty_coef, exit_tol):
    """
    Creates and solves the baseline Pyomo model (no explicit policies).
    Returns the solved model.
    """
    plot_ids = farm_data['plot_ids']  # 0-based indices
    n_plots = farm_data['n_plots']
    distances = farm_data['distances']
    neighbors = farm_data['neighbors']  # index -> list of neighbor indices
    yields_ = farm_data['yields']
    areas = farm_data['areas']  # Ha
    plot_types = farm_data['plot_types']
    plot_labels = farm_data['plot_labels']
    discount_factors_array = farm_data['discount_factors']
    time_factor_cache = farm_data['time_factor_cache']

    # Costs from params
    cost_margin_impl = params['costs']['margin']['implementation']
    cost_margin_maint = params['costs']['margin']['maintenance']
    cost_habitat_impl = params['costs']['habitat']['implementation']
    cost_habitat_maint = params['costs']['habitat']['maintenance']
    cost_existing_hab = params['costs']['habitat']['existing_hab']
    cost_ag_maint = params['costs']['agriculture']['maintenance']

    # Build the Pyomo model
    model = pyo.ConcreteModel("FarmOptimizationBaseline")

    # Sets
    model.I = pyo.Set(initialize=plot_ids)  # 0-based indices

    # Subsets for convenience (using indices)
    ag_plots = [i for i in plot_ids if plot_types[i] == 'ag_plot']
    hab_plots = [i for i in plot_ids if plot_types[i] == 'hab_plots']
    model.AgPlots = pyo.Set(initialize=ag_plots)
    model.HabPlots = pyo.Set(initialize=hab_plots)

    # Variables: margin[i], habitat[i] in [0,1]
    model.margin = pyo.Var(model.I, bounds=(0.0, 1.0), initialize=0.0)
    model.habitat = pyo.Var(model.I, bounds=(0.0, 1.0), initialize=0.0)

    # Numeric parameters
    model.yield_ = pyo.Param(model.I, initialize=lambda m, i: yields_[i], within=pyo.NonNegativeReals)
    model.area = pyo.Param(model.I, initialize=lambda m, i: areas[i], within=pyo.NonNegativeReals)

    # Distance param (2D)
    def dist_init(m, i, j):
        # Check indices are within bounds
        if 0 <= i < n_plots and 0 <= j < n_plots:
            return distances[i, j]
        return 99999  # Default large distance

    model.distance = pyo.Param(model.I, model.I, initialize=dist_init, within=pyo.NonNegativeReals, default=99999)

    # NPV Expression
    @model.Expression(model.I)
    def NPV(m, i):
        """ Computes the NPV for plot i (baseline). """
        p_type = plot_types[i]
        A = m.area[i]  # Area in Ha

        if p_type == 'ag_plot':
            c_label = plot_labels[i]
            if c_label not in params['crops']:
                print(f"Warning: Crop label '{c_label}' for plot {i} not found in params. Using 'Unknown'.")
                c_label = "Unknown"  # Use fallback crop
            crop_def = params['crops'].get(c_label)

            # Margin parameters
            margin_params = crop_def.get('margin', {})
            alpha = margin_params.get('alpha', 0.0)
            beta = margin_params.get('beta', 1.0)  # Use 1 if 0 to avoid exp(0) issues if alpha is 0
            gamma = margin_params.get('gamma', 0.1)
            delta_ = margin_params.get('delta', 0.0)
            epsilon_ = margin_params.get('epsilon', 1.0)
            zeta_ = margin_params.get('zeta', 0.1)

            # Habitat parameters
            hab_params = crop_def.get('habitat', {})
            hab_alpha = hab_params.get('alpha', 0.0)
            hab_beta = hab_params.get('beta', 1.0)
            hab_gamma = hab_params.get('gamma', 0.1)
            hab_delta = hab_params.get('delta', 0.0)
            hab_epsilon = hab_params.get('epsilon', 1.0)
            hab_zeta = hab_params.get('zeta', 0.1)

            # Time factor arrays from cache
            margin_time_factors_gamma = time_factor_cache.get(gamma, np.zeros_like(discount_factors_array))
            margin_time_factors_zeta = time_factor_cache.get(zeta_, np.zeros_like(discount_factors_array))
            hab_time_factors_gamma = time_factor_cache.get(hab_gamma, np.zeros_like(discount_factors_array))
            hab_time_factors_zeta = time_factor_cache.get(hab_zeta, np.zeros_like(discount_factors_array))

            base_yield = m.yield_[i]
            p_c = crop_def.get('p_c', 0.0)

            # Implementation cost (up-front)
            impl_cost = A * (
                    m.margin[i] * cost_margin_impl
                    + m.habitat[i] * cost_habitat_impl
            )

            # Annual maintenance (constant each year)
            margin_maint = m.margin[i] * cost_margin_maint * A
            habitat_maint = m.habitat[i] * cost_habitat_maint * A
            ag_maint = (1 - m.habitat[i]) * cost_ag_maint * A
            total_maint = margin_maint + habitat_maint + ag_maint

            # Yield loss if we convert to habitat
            yield_loss_by_habitat = base_yield * p_c * A * m.habitat[i]

            T_len = len(discount_factors_array)
            npv_val = -impl_cost  # up-front cost (not discounted)

            # Loop over each time step t_idx
            for t_idx in range(T_len):
                # =========== 1) Compute total pollination at this time step ============
                pollination_t = 0.0
                dist_ii = m.distance[i, i]  # Should be 0

                pollination_t += alpha * m.margin[i] * pyo.exp(-beta * dist_ii) * margin_time_factors_gamma[t_idx]

                # Neighbor effects
                if i in neighbors:
                    for j in neighbors[i]:
                        if not (0 <= j < n_plots): continue  # Index check
                        dist_ij = m.distance[i, j]
                        neighbor_type = plot_types[j]

                        if neighbor_type == 'ag_plot':
                            # Neighbor's margin effect on pollination
                            pollination_t += alpha * m.margin[j] * pyo.exp(-beta * dist_ij) * \
                                                 margin_time_factors_gamma[t_idx]
                            # Neighbor's habitat conversion effect on pollination
                            pollination_t += m.habitat[j] * hab_alpha * pyo.exp(-hab_beta * dist_ij) * \
                                                 hab_time_factors_gamma[t_idx]
                        elif neighbor_type == 'hab_plots':
                            # Existing habitat effect on pollination
                            pollination_t += hab_alpha * pyo.exp(-hab_beta * dist_ij) * hab_time_factors_gamma[
                                    t_idx]

                # =========== 2) Compute total pest control at this time step ============
                pest_t = 0.0
                pest_t += delta_ * m.margin[i] * pyo.exp(-epsilon_ * dist_ii) * margin_time_factors_zeta[t_idx]

                # Neighbor effects
                if i in neighbors:
                    for j in neighbors[i]:
                        if not (0 <= j < n_plots): continue
                        dist_ij = m.distance[i, j]
                        neighbor_type = plot_types[j]

                        if neighbor_type == 'ag_plot':
                            # Neighbor's margin effect on pest control
                            pest_t += delta_ * m.margin[j] * pyo.exp(-epsilon_ * dist_ij) * \
                                          margin_time_factors_zeta[t_idx]
                            # Neighbor's habitat conversion effect on pest control
                            pest_t += m.habitat[j] * hab_delta * pyo.exp(-hab_epsilon * dist_ij) * \
                                          hab_time_factors_zeta[t_idx]
                        elif neighbor_type == 'hab_plots':
                            # Existing habitat effect on pest control
                            pest_t += hab_delta * pyo.exp(-hab_epsilon * dist_ij) * hab_time_factors_zeta[t_idx]

                # =========== 3) Combine yield at time t_idx ============
                combined_yield_t = base_yield * (1.0 + pollination_t + pest_t)

                # =========== 4) Compute revenue at time t_idx ============
                revenue_t = combined_yield_t * p_c * A * (1 - m.habitat[i])

                # =========== 5) Subtract maintenance + yield_loss_by_habitat, and discount ============
                yearly_cf = revenue_t - total_maint - yield_loss_by_habitat
                df = discount_factors_array[t_idx]
                npv_val += yearly_cf * df
            return npv_val

        elif p_type == 'hab_plots':
            # "existing habitat" – annual cost is negative revenue
            npv_val = 0.0
            for t_idx, df in enumerate(discount_factors_array):
                npv_val += -cost_existing_hab * A * df  # Cost is negative flow
            return npv_val
        else:
            # If there's a third type, set NPV=0
            return 0.0

    # Penalty Term
    @model.Expression()
    def penalty(m):
        return sum(m.margin[i] ** 2 + m.habitat[i] ** 2 for i in m.I)

    # Objective: maximize sum of NPV minus penalty
    def total_npv_rule(m):
        return pyo.summation(m.NPV) - penalty_coef * m.penalty

    model.Obj = pyo.Objective(rule=total_npv_rule, sense=pyo.maximize)

    # Solve
    solver = pyo.SolverFactory("ipopt")  # IPOPT is suitable for NLP
    solver.options['acceptable_tol'] = exit_tol
    result = solver.solve(model, tee=False)  # Set tee=True for detailed logs
    model.solutions.load_from(result)
    return model


def build_and_solve_pyomo_model_policy(farm_data, params, penalty_coef, exit_tol, policy_params):
    """
    Creates and solves a Pyomo model incorporating policy interventions.

    Args:
        farm_data: Dictionary from precompute_inputs.
        params: Original configuration parameters (potentially modified for eco-premium).
        penalty_coef: Coefficient for regularization penalty.
        exit_tol: Solver exit tolerance.
        policy_params: Dictionary defining active policies and their parameters.
                       Example: {'subsidy': {'adj_hab_factor_margin': 0.3, 'adj_hab_factor_habitat': 0.5},
                                 'mandate': {'min_total_hab_area': 10.0}, # Area in Ha
                                 'payment': {'hab_per_ha': 50}} # Annual payment per Ha

    Returns:
        The solved Pyomo model.
    """
    plot_ids = farm_data['plot_ids']  # 0-based indices
    n_plots = farm_data['n_plots']
    distances = farm_data['distances']
    neighbors = farm_data['neighbors']  # index -> list of neighbor indices
    yields_ = farm_data['yields']
    areas = farm_data['areas']  # Ha
    plot_types = farm_data['plot_types']
    plot_labels = farm_data['plot_labels']
    discount_factors_array = farm_data['discount_factors']
    time_factor_cache = farm_data['time_factor_cache']
    is_adjacent_to_habitat = farm_data['is_adjacent_to_habitat']  # Precomputed

    # --- Base Costs ---
    cost_margin_impl_base = params['costs']['margin']['implementation']
    cost_margin_maint = params['costs']['margin']['maintenance']
    cost_habitat_impl_base = params['costs']['habitat']['implementation']
    cost_habitat_maint = params['costs']['habitat']['maintenance']
    cost_existing_hab = params['costs']['habitat']['existing_hab']
    cost_ag_maint = params['costs']['agriculture']['maintenance']

    # --- Build the Pyomo model ---
    model = pyo.ConcreteModel("FarmOptimizationPolicy")

    # Sets
    model.I = pyo.Set(initialize=plot_ids)  # 0-based indices
    ag_plots = [i for i in plot_ids if plot_types[i] == 'ag_plot']
    hab_plots = [i for i in plot_ids if plot_types[i] == 'hab_plots']
    model.AgPlots = pyo.Set(initialize=ag_plots)
    model.HabPlots = pyo.Set(initialize=hab_plots)

    # Variables
    model.margin = pyo.Var(model.I, bounds=(0.0, 1.0), initialize=0.0)
    model.habitat = pyo.Var(model.I, bounds=(0.0, 1.0), initialize=0.0)

    # Parameters
    model.yield_ = pyo.Param(model.I, initialize=lambda m, i: yields_[i], within=pyo.NonNegativeReals)
    model.area = pyo.Param(model.I, initialize=lambda m, i: areas[i], within=pyo.NonNegativeReals)

    def dist_init(m, i, j):
        if 0 <= i < n_plots and 0 <= j < n_plots:
            return distances[i, j]
        return 99999

    model.distance = pyo.Param(model.I, model.I, initialize=dist_init, within=pyo.NonNegativeReals, default=99999)

    # --- Policy Parameter Extraction ---
    subsidy_policy = policy_params.get('subsidy', {})
    adj_hab_factor_margin = subsidy_policy.get('adj_hab_factor_margin', 0.0)
    adj_hab_factor_habitat = subsidy_policy.get('adj_hab_factor_habitat', 0.0)

    payment_policy = policy_params.get('payment', {})
    habitat_payment_per_ha = payment_policy.get('hab_per_ha', 0.0)

    mandate_policy = policy_params.get('mandate', {})
    min_total_hab_area = mandate_policy.get('min_total_hab_area', 0.0)
    min_margin_adj_hab = mandate_policy.get('min_margin_frac_adj_hab', 0.0)

    # --- NPV Expression (incorporating policies) ---
    @model.Expression(model.I)
    def NPV(m, i):
        p_type = plot_types[i]
        A = m.area[i]  # Area in Ha

        if p_type == 'ag_plot':
            c_label = plot_labels[i]
            if c_label not in params['crops']:
                print(f"Warning: Crop label '{c_label}' for plot {i} not found in params. Using 'Unknown'.")
                c_label = "Unknown"
            crop_def = params['crops'].get(c_label)

            # --- Determine policy-adjusted costs for this plot ---
            current_cost_margin_impl = cost_margin_impl_base
            current_cost_habitat_impl = cost_habitat_impl_base

            # Apply subsidy if plot is adjacent to existing habitat
            if i in is_adjacent_to_habitat and is_adjacent_to_habitat[i]:
                current_cost_margin_impl *= (1.0 - adj_hab_factor_margin)
                current_cost_habitat_impl *= (1.0 - adj_hab_factor_habitat)

            # --- Implementation Cost (up-front) ---
            impl_cost = A * (
                    m.margin[i] * current_cost_margin_impl
                    + m.habitat[i] * current_cost_habitat_impl
            )

            # --- Annual calculations ---
            margin_maint = m.margin[i] * cost_margin_maint * A
            habitat_maint = m.habitat[i] * cost_habitat_maint * A
            ag_maint = (1 - m.habitat[i]) * cost_ag_maint * A
            total_maint = margin_maint + habitat_maint + ag_maint

            # Annual habitat payment (policy)
            annual_habitat_payment = habitat_payment_per_ha * A * m.habitat[i]

            base_yield = m.yield_[i]
            # Use potentially policy-adjusted price from params
            p_c = crop_def.get('p_c', 0.0)

            yield_loss_by_habitat = base_yield * p_c * A * m.habitat[i]

            # Crop parameters for yield effects (same extraction as baseline)
            margin_params = crop_def.get('margin', {})
            alpha = margin_params.get('alpha', 0.0)
            beta = margin_params.get('beta', 1.0)
            gamma = margin_params.get('gamma', 0.1)
            delta_ = margin_params.get('delta', 0.0)
            epsilon_ = margin_params.get('epsilon', 1.0)
            zeta_ = margin_params.get('zeta', 0.1)
            hab_params = crop_def.get('habitat', {})
            hab_alpha = hab_params.get('alpha', 0.0)
            hab_beta = hab_params.get('beta', 1.0)
            hab_gamma = hab_params.get('gamma', 0.1)
            hab_delta = hab_params.get('delta', 0.0)
            hab_epsilon = hab_params.get('epsilon', 1.0)
            hab_zeta = hab_params.get('zeta', 0.1)

            margin_time_factors_gamma = time_factor_cache.get(gamma, np.zeros_like(discount_factors_array))
            margin_time_factors_zeta = time_factor_cache.get(zeta_, np.zeros_like(discount_factors_array))
            hab_time_factors_gamma = time_factor_cache.get(hab_gamma, np.zeros_like(discount_factors_array))
            hab_time_factors_zeta = time_factor_cache.get(hab_zeta, np.zeros_like(discount_factors_array))

            T_len = len(discount_factors_array)
            npv_val = -impl_cost

            # Loop over time steps for discounted cash flows
            for t_idx in range(T_len):
                # Pollination/Pest calculations (identical to baseline model)
                # =========== 1) Compute total pollination at this time step ============
                pollination_t = 0.0
                dist_ii = m.distance[i, i]  # Should be 0
                if alpha > 0:
                    pollination_t += alpha * m.margin[i] * pyo.exp(-beta * dist_ii) * margin_time_factors_gamma[t_idx]
                if i in neighbors:
                    for j in neighbors[i]:
                        if not (0 <= j < n_plots): continue
                        dist_ij = m.distance[i, j]
                        neighbor_type = plot_types[j]
                        if neighbor_type == 'ag_plot':
                            if alpha > 0: pollination_t += alpha * m.margin[j] * pyo.exp(-beta * dist_ij) * \
                                                           margin_time_factors_gamma[t_idx]
                            if hab_alpha > 0: pollination_t += m.habitat[j] * hab_alpha * pyo.exp(-hab_beta * dist_ij) * \
                                                               hab_time_factors_gamma[t_idx]
                        elif neighbor_type == 'hab_plots':
                            if hab_alpha > 0: pollination_t += hab_alpha * pyo.exp(-hab_beta * dist_ij) * \
                                                               hab_time_factors_gamma[t_idx]

                # =========== 2) Compute total pest control at this time step ============
                pest_t = 0.0
                if delta_ > 0:
                    pest_t += delta_ * m.margin[i] * pyo.exp(-epsilon_ * dist_ii) * margin_time_factors_zeta[t_idx]
                if i in neighbors:
                    for j in neighbors[i]:
                        if not (0 <= j < n_plots): continue
                        dist_ij = m.distance[i, j]
                        neighbor_type = plot_types[j]
                        if neighbor_type == 'ag_plot':
                            if delta_ > 0: pest_t += delta_ * m.margin[j] * pyo.exp(-epsilon_ * dist_ij) * \
                                                     margin_time_factors_zeta[t_idx]
                            if hab_delta > 0: pest_t += m.habitat[j] * hab_delta * pyo.exp(-hab_epsilon * dist_ij) * \
                                                        hab_time_factors_zeta[t_idx]
                        elif neighbor_type == 'hab_plots':
                            if hab_delta > 0: pest_t += hab_delta * pyo.exp(-hab_epsilon * dist_ij) * \
                                                        hab_time_factors_zeta[t_idx]

                # =========== 3) Combine yield, Revenue, Cash Flow ============
                combined_yield_t = base_yield * (1.0 + pollination_t + pest_t)
                revenue_t = combined_yield_t * p_c * A * (1 - m.habitat[i])

                # Include annual habitat payment (policy) in cash flow
                yearly_cf = revenue_t - total_maint - yield_loss_by_habitat + annual_habitat_payment
                df = discount_factors_array[t_idx]
                npv_val += yearly_cf * df
            return npv_val

        elif p_type == 'hab_plots':
            # Existing habitat NPV (unchanged by these policies)
            npv_val = 0.0
            for t_idx, df in enumerate(discount_factors_array):
                npv_val += -cost_existing_hab * A * df
            return npv_val
        else:
            return 0.0  # Other plot types

    # --- Penalty Term (same as baseline) ---
    @model.Expression()
    def penalty(m):
        return sum(m.margin[i] ** 2 + m.habitat[i] ** 2 for i in m.I)

    # --- Objective Function ---
    def total_npv_rule(m):
        return pyo.summation(m.NPV) - penalty_coef * m.penalty

    model.Obj = pyo.Objective(rule=total_npv_rule, sense=pyo.maximize)

    # --- Policy Constraints ---
    model.PolicyConstraints = pyo.ConstraintList()

    # Minimum total habitat area mandate
    if min_total_hab_area > 0:
        print(f"Applying policy constraint: Minimum total habitat area = {min_total_hab_area} Ha")
        # Ensure AgPlots is not empty before adding constraint
        if model.AgPlots:
            model.PolicyConstraints.add(
                sum(model.habitat[i] * model.area[i] for i in model.AgPlots) >= min_total_hab_area
            )
        else:
            print("Warning: No agricultural plots found; cannot apply total habitat area mandate.")

    # Minimum margin fraction for plots adjacent to habitat mandate
    if min_margin_adj_hab > 0:
        adj_hab_ag_plots = [i for i in model.AgPlots if i in is_adjacent_to_habitat and is_adjacent_to_habitat[i]]
        if adj_hab_ag_plots:  # Only add constraint if such plots exist
            print(
                f"Applying policy constraint: Min margin fraction {min_margin_adj_hab} for {len(adj_hab_ag_plots)} plots adjacent to habitat")
            for i in adj_hab_ag_plots:
                model.PolicyConstraints.add(model.margin[i] >= min_margin_adj_hab)
        else:
            print("Info: No agricultural plots found adjacent to habitat; min margin mandate not applied.")

    # --- Solve ---
    solver = pyo.SolverFactory("ipopt")
    solver.options['acceptable_tol'] = exit_tol
    # solver.options['max_iter'] = 500
    try:
        result = solver.solve(model, tee=False)  # Set tee=True for solver logs
        model.solutions.load_from(result)
    except Exception as e:
        print(f"Error during Pyomo solve: {e}")
        return model  # Return unsolved model on error

    return model


def assign_pyomo_solution_to_gdf(model, farm_gdf):
    # Assume model.I uses 0-based indices corresponding to farm_gdf rows
    plot_indices = list(model.I)  # Should be 0, 1, 2,...

    margin_vals = []
    habitat_vals = []

    for i in plot_indices:
        try:
            # Check if variables exist on the model before accessing value
            if hasattr(model, 'margin') and i in model.margin:
                margin_val = pyo.value(model.margin[i])
                margin_vals.append(margin_val if margin_val is not None else 0.0)
            else:
                margin_vals.append(0.0)  # Default if variable doesn't exist for index

            if hasattr(model, 'habitat') and i in model.habitat:
                habitat_val = pyo.value(model.habitat[i])
                habitat_vals.append(habitat_val if habitat_val is not None else 0.0)
            else:
                habitat_vals.append(0.0)

        except Exception as e:
            print(f"Error retrieving variable value for index {i}: {e}")
            margin_vals.append(0.0)  # Append default on error
            habitat_vals.append(0.0)

    # Ensure the length matches the GeoDataFrame
    if len(margin_vals) != len(farm_gdf):
        print(
            f"Warning: Mismatch between number of solution values ({len(margin_vals)}) and GDF rows ({len(farm_gdf)}). Padding or truncating.")
        # Adjust lengths - simplistic padding/truncating
        target_len = len(farm_gdf)
        margin_vals = (margin_vals + [0.0] * target_len)[:target_len]
        habitat_vals = (habitat_vals + [0.0] * target_len)[:target_len]

    farm_gdf['margin_intervention'] = margin_vals
    farm_gdf['habitat_conversion'] = habitat_vals
    return farm_gdf


def apply_threshold_and_save(farm_gdf, image_path, output_json, threshold=0.01):
    gdf_processed = farm_gdf.copy()  # Work on a copy

    # Clip interventions below threshold for ag_plots
    if 'type' in gdf_processed.columns:
        ag_mask = gdf_processed['type'] == 'ag_plot'
        gdf_processed['margin_intervention'] = np.where(
            ag_mask & (gdf_processed['margin_intervention'] >= threshold),
            gdf_processed['margin_intervention'],
            0.0
        )
        gdf_processed['habitat_conversion'] = np.where(
            ag_mask & (gdf_processed['habitat_conversion'] >= threshold),
            gdf_processed['habitat_conversion'],
            0.0
        )
    else:
        print("Warning: 'type' column not found in GeoDataFrame. Cannot apply threshold selectively.")
        # Apply threshold universally if type is missing
        gdf_processed['margin_intervention'] = np.where(gdf_processed['margin_intervention'] >= threshold,
                                                        gdf_processed['margin_intervention'], 0.0)
        gdf_processed['habitat_conversion'] = np.where(gdf_processed['habitat_conversion'] >= threshold,
                                                       gdf_processed['habitat_conversion'], 0.0)

    # --- Visualization Call (Commented Out) ---
    # print("Visualization requires 'utils' functions (get_margins_hab_fractions) - Skipping visualization.")
    # margin_lines_gdf, converted_polys_gdf = get_margins_hab_fractions(gdf_processed) # Requires utils
    # if image_path:
    #     visualize_optimized_farm_refactored(gdf_processed, margin_lines_gdf, converted_polys_gdf, image_path) # Requires utils

    gdf_processed.to_file(output_json, driver="GeoJSON")
    return gdf_processed  # Return the GDF with thresholds applied


# Visualization function (Commented out as it depends on utils)
# def visualize_optimized_farm_refactored(farm_gdf, margin_lines_gdf, converted_polys_gdf, image_path):
#     print("Visualization requires utils - Skipping.")
#     pass # Placeholder


def main_run_pyomo(cfg, geojson_path, image_path, output_json, neighbor_dist, exit_tol, penalty_coef):
    farm_gdf = gpd.read_file(geojson_path)
    if farm_gdf.empty:
        print("Error: Input GeoJSON is empty.")
        return None
    params = cfg.params

    # Precompute
    farm_data = precompute_inputs(farm_gdf, params, neighbor_dist=neighbor_dist)

    # Solve Baseline
    start = time.process_time()
    model = build_and_solve_pyomo_model(farm_data, params, penalty_coef=penalty_coef, exit_tol=exit_tol)
    elapsed = time.process_time() - start
    print(f"Baseline Solve time: {elapsed:.2f} seconds")

    # Assign solution
    farm_gdf_solved = assign_pyomo_solution_to_gdf(model, farm_gdf.copy())  # Use copy

    # Save final result (with thresholding)
    # Note: Visualization within apply_threshold_and_save is commented out
    farm_gdf_processed = apply_threshold_and_save(farm_gdf_solved, image_path, output_json, threshold=0.01)

    # Calculate and store baseline NPV
    baseline_npv = pyo.value(pyo.summation(model.NPV))
    farm_data['baseline_npv'] = baseline_npv

    return farm_gdf_processed, farm_data


def calculate_policy_cost_for_farm(model, farm_data, base_params, policy_params):
    costs = {
        'implementation_subsidy': 0.0,
        'maintenance_subsidy_npv': 0.0,
        'habitat_payment_npv': 0.0,
        'premium_guarantee_cost_npv': 0.0,
        'total_policy_cost_npv': 0.0
    }
    if model is None or farm_data is None:
        print("Warning: Model or farm_data missing in calculate_policy_cost_for_farm.")
        return costs # Return zero costs if model/data is missing

    # --- Extract necessary data ---
    areas = farm_data['areas'] # Ha
    plot_ids = farm_data['plot_ids'] # 0-based indices
    plot_types = farm_data['plot_types']
    plot_labels = farm_data['plot_labels']
    base_yields = farm_data['yields'] # Using base yield as approximation
    is_adjacent_to_habitat = farm_data.get('is_adjacent_to_habitat', {})
    discount_factors_array = farm_data['discount_factors']
    T = len(discount_factors_array)
    sum_discount_factors = np.sum(discount_factors_array) # For NPV of annual costs

    # --- Get Base Costs and Prices (ensure they exist in base_params) ---
    try:
        cost_margin_impl_base = base_params['costs']['margin']['implementation']
        cost_habitat_impl_base = base_params['costs']['habitat']['implementation']
        cost_margin_maint_base = base_params['costs']['margin']['maintenance']
        cost_habitat_maint_base = base_params['costs']['habitat']['maintenance']
        # We also need base crop prices from base_params
        base_crop_prices = {crop: details.get('p_c', 0)
                           for crop, details in base_params.get('crops', {}).items()
                           if isinstance(details, dict)}
    except KeyError as e:
        print(f"Warning: Missing base cost/price parameter in config: {e}. Cannot calculate policy cost accurately.")
        # Decide if we should return zero costs or continue with partial calculation
        return costs # Return zero if essential base costs/prices are missing

    # --- Get Policy Parameters ---
    subsidy_policy = policy_params.get('subsidy', {})
    adj_hab_factor_margin = subsidy_policy.get('adj_hab_factor_margin', 0.0)
    adj_hab_factor_habitat = subsidy_policy.get('adj_hab_factor_habitat', 0.0)
    maint_factor_margin = subsidy_policy.get('maint_factor_margin', 0.0)
    maint_factor_habitat = subsidy_policy.get('maint_factor_habitat', 0.0)

    payment_policy = policy_params.get('payment', {})
    habitat_payment_per_ha = payment_policy.get('hab_per_ha', 0.0)

    # --- Get Eco-Premium Factors ---
    eco_premium_policy = policy_params.get('eco_premium', {})
    eco_factors = eco_premium_policy.get('crop_factors', {}) # Dict {crop_label: factor}

    # --- Calculate Costs based on Solved Variables ---
    total_impl_subsidy = 0.0
    annual_maint_subsidy = 0.0
    annual_habitat_payment = 0.0
    annual_premium_guarantee_cost = 0.0

    for i in plot_ids:
        try:
            # Get solved variable values
            margin_i = pyo.value(model.margin[i]) if hasattr(model, 'margin') and i in model.margin else 0.0
            habitat_i = pyo.value(model.habitat[i]) if hasattr(model, 'habitat') and i in model.habitat else 0.0
            area_i = areas[i]
            plot_type_i = plot_types[i]

            # --- Standard Subsidies and Payments ---
            # 1. Implementation Subsidy (One-time cost, only if adjacent)
            if is_adjacent_to_habitat.get(i, False):
                total_impl_subsidy += area_i * margin_i * cost_margin_impl_base * adj_hab_factor_margin
                total_impl_subsidy += area_i * habitat_i * cost_habitat_impl_base * adj_hab_factor_habitat

            # 2. Maintenance Subsidy (Annual cost)
            annual_maint_subsidy += area_i * margin_i * cost_margin_maint_base * maint_factor_margin
            annual_maint_subsidy += area_i * habitat_i * cost_habitat_maint_base * maint_factor_habitat

            # 3. Habitat Payment (Annual cost, only for new habitat on ag plots)
            if plot_type_i == 'ag_plot':
                 annual_habitat_payment += habitat_payment_per_ha * area_i * habitat_i

            # --- NEW: Eco-Premium Guarantee Cost (Annual cost) ---
            # Only applies to agricultural plots with a premium factor > 1
            if plot_type_i == 'ag_plot':
                crop_label_i = plot_labels[i]
                premium_factor = eco_factors.get(crop_label_i, 1.0) # Get factor for this crop

                if premium_factor > 1.0:
                    base_price = base_crop_prices.get(crop_label_i)
                    if base_price is None or base_price <= 0:
                        # print(f"Warning: Base price missing or zero for crop {crop_label_i}, cannot calculate premium cost for plot {i}.")
                        continue # Skip premium cost for this plot if base price invalid

                    premium_price = base_price * premium_factor
                    price_difference = premium_price - base_price
                    base_yield_i = base_yields[i]

                    # Calculate approximate annual cost based on base yield and non-habitat area
                    # Note: habitat_i is the fraction converted, (1-habitat_i) is cropped fraction
                    cost_for_this_plot_annual = price_difference * base_yield_i * area_i * (1.0 - habitat_i)

                    # Ensure cost is not negative (e.g., if habitat_i > 1 somehow)
                    annual_premium_guarantee_cost += max(0.0, cost_for_this_plot_annual)

        except Exception as e:
            print(f"Warning: Error accessing model variable or calculating cost for plot {i}: {e}")
            continue # Skip this plot if there's an error

    # --- Calculate NPV of annual costs ---
    npv_maint_subsidy = annual_maint_subsidy * sum_discount_factors
    npv_habitat_payment = annual_habitat_payment * sum_discount_factors
    npv_premium_guarantee = annual_premium_guarantee_cost * sum_discount_factors # <<< NPV of premium cost

    # --- Store and sum total costs ---
    costs['implementation_subsidy'] = total_impl_subsidy
    costs['maintenance_subsidy_npv'] = npv_maint_subsidy
    costs['habitat_payment_npv'] = npv_habitat_payment
    costs['premium_guarantee_cost_npv'] = npv_premium_guarantee
    costs['total_policy_cost_npv'] = (total_impl_subsidy +
                                      npv_maint_subsidy +
                                      npv_habitat_payment +
                                      npv_premium_guarantee)
    return costs


def run_single_optimization_policy(cfg, geojson_path, nd, exit_tol, pc, policy_params):
    print(f"Running policy EI for: {geojson_path} with params: {policy_params}")
    farm_gdf = None
    farm_data = None
    model = None
    policy_costs = None # Initialize policy costs

    try:
        farm_gdf = gpd.read_file(geojson_path)
        if farm_gdf.empty:
            print("Error: Input GeoJSON is empty.")
            return None, None, None, None # Added None for policy_costs
        base_params = cfg.params  # Get base economic parameters

        # --- Adjust params based on policy if needed (eco-premium etc.) ---
        # This part remains the same - modifies cfg.params['crops'][crop]['p_c'] etc.
        # before passing to precompute/solve.
        # Make sure to use a deepcopy if you modify params here!
        params = copy.deepcopy(base_params) # Work on a copy
        # --- Apply eco-premium to the 'params' copy if present in policy_params ---
        if 'eco_premium' in policy_params and 'crop_factors' in policy_params['eco_premium']:
            if isinstance(params.get('crops'), dict):
                base_prices = {}
                # Store original prices first
                for crop, factor in policy_params['eco_premium']['crop_factors'].items():
                    if crop in params['crops'] and isinstance(params['crops'][crop], dict):
                         base_prices[crop] = base_params['crops'][crop].get('p_c', 0) # Use original base price
                # Apply the factor
                for crop, factor in policy_params['eco_premium']['crop_factors'].items():
                    if crop in base_prices:
                        params['crops'][crop]['p_c'] = base_prices[crop] * factor # Modify the 'params' copy


        # --- Precompute and Solve ---
        farm_data = precompute_inputs(farm_gdf, params, neighbor_dist=nd) # Pass potentially modified params
        model = build_and_solve_pyomo_model_policy(farm_data, params, penalty_coef=pc, exit_tol=exit_tol,
                                                   policy_params=policy_params) # Pass policy_params here

        # --- Process Results ---
        farm_gdf_solved = assign_pyomo_solution_to_gdf(model, farm_gdf.copy())

        # Apply threshold (remains the same)
        threshold = 0.01
        # ... (thresholding code remains the same) ...
        if 'type' in farm_gdf_solved.columns:
            ag_mask = farm_gdf_solved['type'] == 'ag_plot'
            farm_gdf_solved['margin_intervention'] = np.where(
                ag_mask & (farm_gdf_solved['margin_intervention'] >= threshold),
                farm_gdf_solved['margin_intervention'], 0.0
            )
            farm_gdf_solved['habitat_conversion'] = np.where(
                ag_mask & (farm_gdf_solved['habitat_conversion'] >= threshold),
                farm_gdf_solved['habitat_conversion'], 0.0
            )
        else:
            farm_gdf_solved['margin_intervention'] = np.where(farm_gdf_solved['margin_intervention'] >= threshold,
                                                              farm_gdf_solved['margin_intervention'], 0.0)
            farm_gdf_solved['habitat_conversion'] = np.where(farm_gdf_solved['habitat_conversion'] >= threshold,
                                                             farm_gdf_solved['habitat_conversion'], 0.0)


        # --- Calculate Policy Model NPV (excluding penalty) ---
        try:
            if hasattr(model, 'NPV') and model.NPV:
                total_npv_value = pyo.value(pyo.summation(model.NPV))
            else:
                print("Warning: NPV expression not found on the solved model.")
                total_npv_value = None
        except Exception as e:
            print(f"Could not calculate total NPV from policy model: {e}")
            total_npv_value = None

        # Add calculated NPV to farm_data (or return separately)
        if farm_data: farm_data['policy_npv'] = total_npv_value

        # --- Calculate Policy Costs for this farm ---
        # Use the original base_params for cost calculation, not the potentially modified 'params'
        policy_costs = calculate_policy_cost_for_farm(model, farm_data, base_params, policy_params)
        if farm_data: farm_data['policy_costs'] = policy_costs # Store costs in farm_data


        # Return GDF, farm_data (now containing costs), model, and policy_costs dict
        return farm_gdf_solved, farm_data, model, policy_costs

    except Exception as e:
        print(f"Error in run_single_optimization_policy for {geojson_path}: {e}")
        # Ensure we return the correct number of values, even on error
        return farm_gdf, farm_data, model, policy_costs


def parse_geojson(geojson_path):
    """ Reads the interventions GeoJSON file (output from EI) for EC repositioning. """
    plots = []
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict) or data.get('type') != 'FeatureCollection':
            print(f"Error: Invalid GeoJSON structure in {geojson_path}. Expected FeatureCollection.")
            return plots  # Return empty list

        for feature in data.get('features', []):
            if not isinstance(feature, dict) or 'properties' not in feature or 'geometry' not in feature:
                print("Warning: Skipping invalid feature in GeoJSON.")
                continue

            props = feature['properties']
            if not props: props = {}  # Handle null properties

            try:
                geom = shape(feature['geometry']) if feature['geometry'] else None  # Handle null geometry
                if geom is None or geom.is_empty:
                    print(f"Warning: Skipping feature with null or empty geometry (ID: {props.get('id', 'N/A')}).")
                    continue
                if not geom.is_valid:
                    print(f"Warning: Invalid geometry found (ID: {props.get('id', 'N/A')}). Attempting buffer(0).")
                    geom = geom.buffer(0)
                    if not geom.is_valid or geom.is_empty:
                        print("Error: Geometry still invalid/empty after buffer(0). Skipping.")
                        continue

            except Exception as e:
                print(f"Error parsing geometry for feature (ID: {props.get('id', 'N/A')}): {e}")
                continue

            # Build record for EC repositioning
            plot_record = {
                'farm_id': props.get('farm_id'),  # Included via save_gdf_for_ec
                'plot_id': props.get('id'),  # Original plot ID
                'plot_type': props.get('type'),  # 'ag_plot', 'hab_plots', etc.
                'label': props.get('label', ""),  # Crop label or other identifier
                'yield': props.get('yield', 0.0),
                'geometry': geom,
                # Fractions determined by the EI run:
                'margin_frac': props.get("margin_intervention", 0.0),
                'habitat_frac': props.get("habitat_conversion", 0.0)
            }
            plots.append(plot_record)

    except FileNotFoundError:
        print(f"Error: GeoJSON file not found at {geojson_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {geojson_path}")
    except Exception as e:
        print(f"An unexpected error occurred during GeoJSON parsing: {e}")

    if not plots:
        print("Warning: No valid plot data parsed from GeoJSON.")

    return plots



def build_connectivity_graph_from_chosen_pieces(chosen_pieces, adjacency_dist=0.0):
    """ Builds a NetworkX graph from the list of selected pieces. """
    G = nx.Graph()
    if not chosen_pieces:
        return G

    node_map = {}  # Map piece index in chosen_pieces to graph node ID
    for i, piece in enumerate(chosen_pieces):
        node_id = i  # Use index as node ID
        node_map[i] = node_id
        G.add_node(node_id,
                   # Store relevant attributes for metric calculation
                   geometry=piece['geom'],  # Keep geometry for potential future use
                   node_type=piece['type'],
                   area=piece.get('area', 0.0),  # Ha
                   length=piece.get('length', 0.0))  # km

    # Add edges based on adjacency distance
    for i, j in itertools.combinations(range(len(chosen_pieces)), 2):
        pi = chosen_pieces[i]
        pj = chosen_pieces[j]

        # Robust distance check
        try:
            geom_i = pi['geom'].representative_point() if pi['geom'] else Point(0, 0)
            geom_j = pj['geom'].representative_point() if pj['geom'] else Point(0, 0)
            dist = geom_i.distance(geom_j)  # Distance in meters
        except Exception:
            dist = float('inf')  # Skip if distance fails

        if dist <= adjacency_dist:
            # Add edge between corresponding nodes in the graph
            node_i = node_map[i]
            node_j = node_map[j]
            G.add_edge(node_i, node_j, distance_m=dist)  # Store distance if needed

    return G


def compute_connectivity_metric(G, connectivity_metric, **kwargs):
    """ Computes a connectivity metric on the graph G. """
    if not G or G.number_of_nodes() == 0:
        return 0.0

    metric_value = 0.0

    if connectivity_metric == 'IIC':
        # Simplified IIC-like calculation based on original EC code snippet objective
        # Assumes node attributes 'area' (Ha) and 'length' (km) exist
        # This calculation sums pairwise interactions within connected components.
        # It's NOT the standard IIC formula (which involves shortest paths).
        al_factor = kwargs.get('al_factor', 1.0)  # Get factor from kwargs
        total_sum = 0.0

        # Iterate through connected components
        for component_nodes in nx.connected_components(G):
            component_sum = 0.0
            nodes_in_comp = list(component_nodes)

            # Sum pairwise interactions (including self-interaction) within the component
            for i in range(len(nodes_in_comp)):
                for j in range(len(nodes_in_comp)):  # Include i==j (self-interaction)
                    node_i_id = nodes_in_comp[i]
                    node_j_id = nodes_in_comp[j]

                    # Get node attributes
                    attr_i = G.nodes[node_i_id]
                    attr_j = G.nodes[node_j_id]
                    ai = attr_i.get('area', 0.0)
                    li = attr_i.get('length', 0.0)
                    aj = attr_j.get('area', 0.0)
                    lj = attr_j.get('length', 0.0)

                    # Apply interaction formula (same as in build_reposition_ilp objective)
                    interaction = (li * lj) + (ai * aj) + al_factor * (ai * lj + aj * li)
                    component_sum += max(0, interaction)  # Ensure non-negative

            total_sum += component_sum
        metric_value = total_sum  # The metric is the total sum of these interactions

    elif connectivity_metric == 'num_components':
        # Simple metric: Number of connected components (lower is better)
        metric_value = nx.number_connected_components(G)

    elif connectivity_metric == 'largest_component_size':
        # Size of the largest connected component (nodes)
        if nx.number_connected_components(G) > 0:
            largest_comp = max(nx.connected_components(G), key=len)
            metric_value = len(largest_comp)
        else:
            metric_value = 0

    # Add other metrics here (e.g., LCC, PC - these require more complex path calculations)
    # elif connectivity_metric == 'PC':
    #     # Requires calculation involving probabilities based on distances
    #     print("PC metric calculation not fully implemented in this example.")
    #     metric_value = 0.0 # Placeholder

    else:
        print(f"Warning: Unknown connectivity metric '{connectivity_metric}'. Returning 0.0.")
        metric_value = 0.0

    return metric_value


def save_gdf_for_ec(farm_gdf, output_path):
    """Saves the GeoDataFrame from EI in a format readable by ec_model.parse_geojson."""
    print(f"Preparing to save GDF for EC to {output_path}...")
    # Ensure essential columns for EC are present
    required_cols = ['id', 'type', 'label', 'yield', 'geometry', 'margin_intervention', 'habitat_conversion', 'farm_id']
    gdf_to_save = farm_gdf.copy()

    # --- Data Cleaning and Preparation ---
    # 1. Ensure 'id' column exists and is suitable
    if 'id' not in gdf_to_save.columns:
        print("Warning: 'id' column missing. Adding sequential IDs.")
        gdf_to_save['id'] = range(1, len(gdf_to_save) + 1)
    # Ensure IDs are integers if possible (some tools expect this)
    try:
        gdf_to_save['id'] = gdf_to_save['id'].astype(int)
    except ValueError:
        print("Warning: Could not convert 'id' column to integer.")

    # 2. Add 'farm_id' if missing
    if 'farm_id' not in gdf_to_save.columns:
        try:
            # Attempt to infer from path (crude example)
            parent_dir_name = os.path.basename(os.path.dirname(os.path.dirname(output_path)))
            farm_id_match = [int(s) for s in parent_dir_name.split('_') if s.isdigit()]
            if farm_id_match:
                gdf_to_save['farm_id'] = farm_id_match[0]
                print(f"Inferred farm_id {farm_id_match[0]} for saving.")
            else:
                raise ValueError("Could not infer farm_id")
        except Exception:
            gdf_to_save['farm_id'] = 0  # Default farm_id
            print("Warning: Could not infer farm_id, using default 0.")
    # Ensure farm_id is integer
    try:
        gdf_to_save['farm_id'] = gdf_to_save['farm_id'].fillna(0).astype(int)
    except ValueError:
        print("Warning: Could not convert 'farm_id' column to integer.")

    # 3. Ensure 'type' column exists
    if 'type' not in gdf_to_save.columns:
        print("Warning: 'type' column missing. Defaulting to 'ag_plot'.")
        gdf_to_save['type'] = 'ag_plot'
    gdf_to_save['type'] = gdf_to_save['type'].fillna('Unknown')  # Fill NaNs

    # 4. Ensure 'label' column exists
    if 'label' not in gdf_to_save.columns: gdf_to_save['label'] = "Unknown"
    gdf_to_save['label'] = gdf_to_save['label'].fillna("Unknown")

    # 5. Ensure 'yield' column exists
    if 'yield' not in gdf_to_save.columns: gdf_to_save['yield'] = 0.0
    gdf_to_save['yield'] = gdf_to_save['yield'].fillna(0.0)
    # Ensure yield is numeric
    gdf_to_save['yield'] = pd.to_numeric(gdf_to_save['yield'], errors='coerce').fillna(0.0)

    # 6. Ensure intervention columns exist and are numeric
    if 'margin_intervention' not in gdf_to_save.columns: gdf_to_save['margin_intervention'] = 0.0
    gdf_to_save['margin_intervention'] = pd.to_numeric(gdf_to_save['margin_intervention'], errors='coerce').fillna(0.0)

    if 'habitat_conversion' not in gdf_to_save.columns: gdf_to_save['habitat_conversion'] = 0.0
    gdf_to_save['habitat_conversion'] = pd.to_numeric(gdf_to_save['habitat_conversion'], errors='coerce').fillna(0.0)

    # 7. Ensure geometry column is named 'geometry' and clean geometries
    if 'geometry' not in gdf_to_save.columns:
        geom_col = gdf_to_save.geometry.name
        if geom_col != 'geometry':
            print(f"Renaming geometry column '{geom_col}' to 'geometry'.")
            gdf_to_save = gdf_to_save.rename_geometry('geometry')

    # Clean invalid geometries
    initial_len = len(gdf_to_save)
    gdf_to_save = gdf_to_save[gdf_to_save.geometry.is_valid & ~gdf_to_save.geometry.is_empty]
    if len(gdf_to_save) < initial_len:
        print(f"Warning: Removed {initial_len - len(gdf_to_save)} rows with invalid/empty geometries.")

    if gdf_to_save.empty:
        print("Error: GeoDataFrame is empty after cleaning. Cannot save.")
        # Create an empty file to avoid downstream errors
        empty_geojson = {"type": "FeatureCollection", "features": []}
        with open(output_path, 'w') as f:
            json.dump(empty_geojson, f)
        return

    # --- Select and Save ---
    # Keep only necessary columns
    cols_to_keep = [col for col in required_cols if col in gdf_to_save.columns]
    gdf_to_save_filtered = gdf_to_save[cols_to_keep]

    try:
        # Save to GeoJSON
        gdf_to_save_filtered.to_file(output_path, driver="GeoJSON")
        print(f"Saved GeoJSON for EC repositioning to {output_path} ({len(gdf_to_save_filtered)} features)")
    except Exception as e:
        print(f"Error saving GeoJSON for EC: {e}")
        print(f"Columns available: {gdf_to_save_filtered.columns}")
        print(f"CRS: {gdf_to_save_filtered.crs}")
        print(f"Data sample:\n{gdf_to_save_filtered.head()}")


def evaluate_policy_scenario(cfg, geojson_path, scenario, ec_params, base_results=None):
    """
    Runs EI with policy, then EC repositioning, and compares to baseline.
    """
    policy_params = scenario['params']
    policy_id = scenario['policy_id']
    print(f"\n--- Evaluating Policy Scenario ---")
    print(f"GeoJSON: {geojson_path}")
    print(f"Policy Params: {policy_params}")

    results = {'policy': policy_params}  # Store input policy

    # --- Run Modified EI Model ---
    start_ei = time.time()
    policy_farm_gdf, policy_farm_data, policy_model = None, None, None  # Initialize
    try:
        policy_farm_gdf, policy_farm_data, policy_model = run_single_optimization_policy(
            cfg,
            geojson_path,
            neighbor_dist,
            exit_tol=exit_tol,
            pc=penalty_coef,
            policy_params=policy_params
        )

        if policy_farm_gdf is None or policy_farm_data is None:
            raise ValueError("Policy EI run failed to return valid results.")

        results['ei_runtime'] = time.time() - start_ei
        results['policy_npv'] = policy_farm_data.get('policy_npv', None)  # Get calculated NPV (pre-penalty)
        print(f"EI Policy Run Time: {results['ei_runtime']:.2f}s")
        print(f"Policy EI NPV (pre-penalty): {results['policy_npv']}")

        # Create a temporary path for the intermediate GeoJSON
        temp_geojson_path = os.path.join(farm_results_dir, f"policy_output_{policy_id}.geojson")

        save_gdf_for_ec(policy_farm_gdf, temp_geojson_path)  # Save results for EC

    except Exception as e:
        print(f"Error running policy EI model: {e}")
        results['ei_error'] = str(e)
        results['ei_runtime'] = time.time() - start_ei
        return results  # Cannot proceed to EC if EI failed

    # --- Run EC Repositioning ---
    results['ec_reposition_runtime'] = None
    results['policy_connectivity_score'] = None
    results['policy_reposition_optim_value'] = None
    results['ec_error'] = None

    if os.path.exists(temp_geojson_path):
        try:
            start_ec = time.time()
            # Parse the output of the policy-driven EI run
            ec_plots = parse_geojson(temp_geojson_path)
            if not ec_plots:
                raise ValueError("EC parsing resulted in no plots.")

            # Run ONLY the repositioning part of EC
            chosen_pieces_policy, optim_val_policy, conn_val_policy, _ = solve_reposition_ilp(
                ec_plots,
                adjacency_dist=ec_params['adjacency_dist'],
                boundary_seg_count=ec_params['boundary_seg_count'],
                interior_cell_count=ec_params['interior_cell_count'],
                connectivity_metric=ec_params['connectivity_metric'],
                al_factor=ec_params['al_factor']
            )
            results['ec_reposition_runtime'] = time.time() - start_ec
            results['policy_connectivity_score'] = conn_val_policy
            results['policy_reposition_optim_value'] = optim_val_policy  # Objective value from repositioning ILP

            print(f"EC Repositioning Runtime: {results['ec_reposition_runtime']:.2f}s")
            print(f"Resulting Connectivity Score ({ec_params['connectivity_metric']}): {conn_val_policy:.4f}")

        except Exception as e:
            print(f"Error running EC repositioning: {e}")
            results['ec_error'] = str(e)
            if 'ec_reposition_runtime' not in results or results['ec_reposition_runtime'] is None:
                results['ec_reposition_runtime'] = time.time() - start_ec  # Record time even if error occurred
    else:
        print(f"Error: Intermediate GeoJSON for EC not found at {temp_geojson_path}")
        results['ec_error'] = "Intermediate GeoJSON missing"

    # --- Comparison Output (Optional) ---
    if base_results:
        print("\n--- Comparison ---")
        print(f"Metric                       | Baseline        | Policy")
        print(f"-----------------------------|-----------------|----------------")
        npv_base_str = f"{base_results.get('baseline_npv', 'N/A'):<15.2f}" if isinstance(
            base_results.get('baseline_npv'), (int, float)) else f"{'N/A':<15}"
        npv_policy_str = f"{results.get('policy_npv', 'N/A'):<15.2f}" if isinstance(results.get('policy_npv'),
                                                                                    (int, float)) else f"{'N/A':<15}"
        conn_base_str = f"{base_results.get('baseline_connectivity_score', 'N/A'):<15.4f}" if isinstance(
            base_results.get('baseline_connectivity_score'), (int, float)) else f"{'N/A':<15}"
        conn_policy_str = f"{results.get('policy_connectivity_score', 'N/A'):<15.4f}" if isinstance(
            results.get('policy_connectivity_score'), (int, float)) else f"{'N/A':<15}"

        print(f"NPV (pre-penalty)          | {npv_base_str} | {npv_policy_str}")
        print(f"Connectivity ({ec_params['connectivity_metric']:<4}) | {conn_base_str} | {conn_policy_str}")

    return results


def visualize_policy_comparison(summary_df, output_dir):
    df_copy = summary_df.copy()
    num_cols = ['total_npv', 'connectivity_score', 'repositioning_score']
    for col in num_cols:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    baseline_agg = df_copy[df_copy['policy_id'] == 'baseline'][num_cols].agg(np.nanmean)
    base_npv_avg = baseline_agg['total_npv']
    base_conn_avg = baseline_agg['connectivity_score']
    base_repos_avg = baseline_agg['repositioning_score']

    policy_ids = df_copy[df_copy['policy_id'] != 'baseline']['policy_id'].unique()
    policy_agg_results = {}
    for policy_id in policy_ids:
        policy_data = df_copy[df_copy['policy_id'] == policy_id]
        if not policy_data.empty:
            # Aggregate using nanmean
            policy_agg = policy_data[num_cols].agg(np.nanmean)
            policy_agg_results[policy_id] = {
                'avg_npv': policy_agg['total_npv'],
                'avg_conn': policy_agg['connectivity_score']
            }

    plt.figure(figsize=(12, 8))

    if pd.notna(base_npv_avg) and pd.notna(base_conn_avg):
        plt.scatter(base_npv_avg, base_conn_avg, color='green', marker='*', s=250, label=f'Baseline - Optimized', zorder=10)
        #plt.text(base_npv_avg * 1.01, base_conn_avg * 1.01,
        #         f'Avg. Baseline\nNPV={base_npv_avg:,.0f}\nConn={base_conn_avg:.4f}', fontsize=10, color='red',
        #         va='bottom', zorder=11)
        # Optional: Reference lines for average baseline
        plt.axhline(base_conn_avg, color='grey', linestyle='--', linewidth=0.8, zorder=1)
        plt.axvline(base_npv_avg, color='grey', linestyle='--', linewidth=0.8, zorder=1)

        plt.scatter(base_npv_avg, base_repos_avg, color='blue', marker='*', s=250, label=f'Baseline - Repositioned', zorder=10)
        plt.axhline(base_repos_avg, color='grey', linestyle='--', linewidth=0.8, zorder=1)
        plt.axvline(base_npv_avg, color='grey', linestyle='--', linewidth=0.8, zorder=1)
    else:
        print("Warning: Could not plot average baseline due to missing aggregated values.")

    # Plot aggregated policy points
    plotted_policies = False
    cmap = plt.get_cmap('tab10')  # Colormap for policies
    policy_labels_handles = []  # To store handles for creating the legend

    policy_id_list = sorted(list(policy_agg_results.keys()))  # Sort for consistent color assignment

    for i, policy_id in enumerate(policy_id_list):
        res = policy_agg_results[policy_id]
        avg_npv = res['avg_npv']
        avg_conn = res['avg_conn']

        if pd.notna(avg_npv) and pd.notna(avg_conn):
            color = cmap(i % 10)  # Cycle through colors
            # Plot the scatter point for the policy average
            scatter = plt.scatter(avg_npv, avg_conn, s=100, alpha=0.9, color=color, zorder=5)
            # Add policy ID text label near the point
            #plt.text(avg_npv, avg_conn, f'  {policy_id}', fontsize=9, va='center', zorder=6)
            # Store a patch handle for this policy for the legend
            policy_labels_handles.append(mpatches.Patch(color=color, label=f'{policy_id}'))
            plotted_policies = True
        else:
            print(
                f"Warning: Skipping aggregate plot point for policy '{policy_id}' due to missing aggregated data (Avg NPV={avg_npv}, Avg Conn={avg_conn}).")

    # Check if any points were plotted at all
    if not plotted_policies and (pd.isna(base_npv_avg) or pd.isna(base_conn_avg)):
        print("Skipping aggregate plot generation - no valid aggregated data points.")
        plt.close()  # Close the empty figure
        return

    # --- Configure Plot Appearance ---
    plt.xlabel("Average Total Aggregated NPV")
    plt.ylabel("Average Aggregated Connectivity Score (IIC)")

    # Create legend from collected handles
    legend_handles = policy_labels_handles
    if pd.notna(base_npv_avg) and pd.notna(base_conn_avg):
        baseline_optim_marker = plt.Line2D([0], [0], marker='*', color='w', label='Baseline - Optimized',
                                     markerfacecolor='green', markersize=15)
        baseline_repos_marker = plt.Line2D([0], [0], marker='*', color='w', label='Baseline - Repositioned',
                                     markerfacecolor='blue', markersize=15)
        legend_handles = [baseline_optim_marker, baseline_repos_marker] + legend_handles

    # Place legend outside the plot area to the right
    plt.legend(handles=legend_handles, title="Policy Scenarios", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Right boundary adjusted to 0.8
    plot_filename = os.path.join(output_dir, "aggregate_policy_comparison_across_configs.svg")
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()

    unique_configs = summary_df['config_id'].unique()
    for config_id in unique_configs:
        config_path = os.path.join(base_farm_dir, f"config_{config_id}")
        config_results_dir = os.path.join(config_path, "policy")

        config_data = summary_df[summary_df['config_id'] == config_id].copy()

        baseline_data = config_data[config_data['policy_id'] == 'baseline']
        policy_data = config_data[config_data['policy_id'] != 'baseline']

        if baseline_data.empty:
            print(f"Warning: No baseline data found for config {config_id}. Skipping plot.")
            continue

        # Use .iloc[0] safely after checking not empty, handle potential NaNs
        base_npv = baseline_data['total_npv'].iloc[0] if pd.notna(baseline_data['total_npv'].iloc[0]) else None
        base_conn = baseline_data['connectivity_score'].iloc[0] if pd.notna(baseline_data['connectivity_score'].iloc[0]) else None
        base_repos = baseline_data['repositioning_score'].iloc[0] if pd.notna(
            baseline_data['repositioning_score'].iloc[0]) else None

        plt.figure(figsize=(12, 8)) # Increased figure size

        # Plot baseline point
        if base_npv is not None and base_conn is not None:
            plt.scatter(base_npv, base_conn, color='green', marker='*', s=250, label=f'Baseline - Optimized', zorder=10)
            #plt.text(base_npv * 1.01, base_conn * 1.01, f'Baseline\nNPV={base_npv:,.0f}\nConn={base_conn:.4f}', fontsize=10, color='red', va='bottom')
            # Add reference lines
            plt.axhline(base_conn, color='grey', linestyle='--', linewidth=0.8, zorder=1)
            plt.axvline(base_npv, color='grey', linestyle='--', linewidth=0.8, zorder=1)

            plt.scatter(base_npv, base_repos, color='blue', marker='*', s=250, label=f'Baseline - Repositioned', zorder=10)
            # plt.text(base_npv * 1.01, base_conn * 1.01, f'Baseline\nNPV={base_npv:,.0f}\nConn={base_conn:.4f}', fontsize=10, color='red', va='bottom')
            # Add reference lines
            plt.axhline(base_conn, color='grey', linestyle='--', linewidth=0.8, zorder=1)
            plt.axvline(base_repos, color='grey', linestyle='--', linewidth=0.8, zorder=1)
        else:
             print(f"Warning: Baseline point for Config {config_id} has missing values (NPV={base_npv}, Conn={base_conn})")


        # Plot policy points
        plotted_policies = False
        policy_labels = [] # For unified legend
        for index, row in policy_data.iterrows():
            policy_id = row['policy_id']
            npv = row['total_npv']
            # Use connectivity from repositioning if available, else None
            conn = row['connectivity_score'] # This now comes from repositioning score

            if pd.notna(npv) and pd.notna(conn):
                # Use a colormap for different policies
                cmap = plt.get_cmap('tab10')
                policy_index = list(policy_data['policy_id'].unique()).index(policy_id) % 10 # Cycle through 10 colors
                color = cmap(policy_index)

                scatter = plt.scatter(npv, conn, s=100, label=f'{policy_id}', alpha=0.9, color=color, zorder=5)
                #plt.text(npv, conn, f'  {policy_id}', fontsize=9, va='center') # Annotate points nearby
                policy_labels.append(mpatches.Patch(color=color, label=f'{policy_id}'))
                plotted_policies = True
            else:
                 print(f"Warning: Skipping plot point for policy '{policy_id}' in config {config_id} due to missing data (NPV={npv}, Conn={conn}).")

        if not plotted_policies and (base_npv is None or base_conn is None) :
             print(f"Skipping plot generation for config {config_id} - no valid data points.")
             plt.close() # Close the empty figure
             continue

        plt.xlabel("Total Aggregated NPV")
        plt.ylabel("Aggregated Connectivity Score (IIC)")
        #plt.title(f"Policy Evaluation: NPV vs. Connectivity Score (Config {config_id})")

        # Create combined legend including Baseline marker if exists
        handles, labels = plt.gca().get_legend_handles_labels()
        # Filter unique labels/handles if policies plotted multiple times (shouldn't happen with iterrows)
        by_label = dict(zip(labels, handles))
        if base_npv is not None and base_conn is not None:
            # Manually add baseline marker to legend handles/labels if needed, or rely on its label in scatter
             pass # Already labeled in scatter

        # Place legend outside plot area
        plt.legend(handles=by_label.values(), labels=by_label.keys(), title="Policy Scenarios", loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout(rect=[0, 0, 0.8, 1]) # Adjust layout to make space for legend (right side)

        # Save the plot
        plot_filename = os.path.join(config_results_dir, f"config_{config_id}_policy_comparison.png")
        try:
            plt.savefig(plot_filename, bbox_inches='tight') # Use bbox_inches='tight'
            print(f"Saved comparison plot: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        plt.close()


if __name__ == "__main__":
    cfg = Config()  # Load base configuration

    exit_tol = 1e-6
    neighbor_dist = 1500
    penalty_coef = 1e5

    syn_farm_dir = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms")
    base_farm_dir = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms", "mc")
    num_configs = np.arange(1, 501)

    # --- Define Policy Scenarios ---
    policy_scenarios = [
        {'policy_id': 'hab_subsidy_adj_50pct', 'params': {'subsidy': {'adj_hab_factor_habitat': 0.5}}},
        {'policy_id': 'margin_sub40_hab_pay75',
         'params': {'subsidy': {'adj_hab_factor_margin': 0.4}, 'payment': {'hab_per_ha': 75}}},
        {'policy_id': 'min_hab_area_5ha', 'params': {'mandate': {'min_total_hab_area': 5.0}}},
        {'policy_id': 'mandate3ha_and_pay50',
         'params': {'mandate': {'min_total_hab_area': 3.0}, 'payment': {'hab_per_ha': 50}}},
        {'policy_id': 'eco_premium_10pct', 'params': {'eco_premium': {'price_increase_factor': 1.1}}},
        {'policy_id': 'strong_subsidy_m60_h80',
         'params': {'subsidy': {'adj_hab_factor_margin': 0.6, 'adj_hab_factor_habitat': 0.8}}},
        {'policy_id': 'adj_margin_mandate_10pct', 'params': {'mandate': {'min_margin_frac_adj_hab': 0.1}}},
    ]

    all_configs_aggregated_results = {}
    output_dir = os.path.join(syn_farm_dir, "plots", "policy")
    os.makedirs(output_dir, exist_ok=True)
    for config_id in num_configs:
        print(f"Running Configuration: {config_id}")

        config_path = os.path.join(base_farm_dir, f"config_{config_id}")
        config_results_dir = os.path.join(config_path, "policy")
        os.makedirs(config_results_dir, exist_ok=True)

        num_farms = sum(1 for item in os.listdir(config_path)
                        if os.path.isdir(os.path.join(config_path, item)) and item.startswith("farm_"))

        current_config_summary = {'baseline': {}, 'policies': {}}
        baseline_gdfs_config = []
        policy_gdfs_config = {scenario['policy_id']: [] for scenario in policy_scenarios}
        aggregate_baseline_npv = 0.0
        policy_aggregate_npvs = {scenario['policy_id']: 0.0 for scenario in policy_scenarios}

        for farm_id in range(1, num_farms + 1):
            farm_path = os.path.join(config_path, f"farm_{farm_id}")
            geojson_path = os.path.join(farm_path, "input.geojson")

            farm_results_dir = os.path.join(farm_path, "policy")  # For final outputs
            os.makedirs(farm_results_dir, exist_ok=True)

            # --- Baseline Run (Standard EI + EC Repositioning) ---
            baseline_results = {}
            baseline_farm_data = None
            print("Running Baseline EI...")
            baseline_farm_json_path = os.path.join(farm_results_dir, f"farm_{farm_id}_baseline_EI_interventions.geojson")

            try:
                base_farm_gdf_processed, baseline_farm_data = main_run_pyomo(
                    cfg, geojson_path,
                    image_path=None,
                    output_json=baseline_farm_json_path,
                    neighbor_dist=neighbor_dist,
                    exit_tol=exit_tol,
                    penalty_coef=penalty_coef
                )

                if 'farm_id' not in base_farm_gdf_processed.columns:
                    base_farm_gdf_processed['farm_id'] = farm_id

                baseline_gdfs_config.append(base_farm_gdf_processed)
                farm_npv = baseline_farm_data.get('baseline_npv', 0.0)
                aggregate_baseline_npv += farm_npv if farm_npv is not None else 0.0
            except Exception as e:
                continue

        # --- 2. Combine Baseline GDFs and run EC Connectivity ---
        print(f"\n--- Combining Baseline Results for Config {config_id} and Running EC Connectivity ---")
        combined_baseline_gdf = gpd.pd.concat(baseline_gdfs_config, ignore_index=True)
        combined_baseline_geojson_path = os.path.join(config_results_dir,
                                                      f"config_{config_id}_combined_baseline.geojson")
        save_gdf_for_ec(combined_baseline_gdf, combined_baseline_geojson_path)
        ec_params_repos = {
            'adjacency_dist': 0.0,
            'boundary_seg_count': 10,
            'interior_cell_count': 10,
            'connectivity_metric': 'IIC',
            'al_factor': 1e-9
        }
        ec_params_baseline = {
            'al_factor': 1e-9,
            'neib_dist': 1500,
            'exit_tol': 1e-6,
            'params': cfg.params,
            'connectivity_metric': 'IIC',
            'max_loss_ratio': 0.1,
            'adjacency_dist': 0.0,
            'boundary_seg_count': 10,
            'interior_cell_count': 10,
            'margin_weight': 500
        }
        base_ec_plots = parse_geojson(combined_baseline_geojson_path)
        try:
            _, _, conn_val_repos = solve_reposition_ilp(base_ec_plots, **ec_params_repos)
            _, _, conn_val_optim, _, _ = solve_connectivity_ilp(base_ec_plots, **ec_params_baseline)

            current_config_summary['baseline']['connectivity_score'] = conn_val_optim
            current_config_summary['baseline']['repositioning_score'] = conn_val_repos
            current_config_summary['baseline']['total_npv'] = aggregate_baseline_npv
            print(
                f"Config {config_id} Combined Baseline Connectivity ({ec_params_baseline['connectivity_metric']}): {conn_val_optim:.4f}")
            print(f"Config {config_id} Aggregate Baseline NPV: {aggregate_baseline_npv:.2f}")
        except Exception as e:
            continue

        # --- 3. Run Policy Scenarios EI for all farms ---
        for scenario in policy_scenarios:
            policy_id = scenario['policy_id']
            policy_params = scenario['params']
            print(f"\n--- Running Policy '{policy_id}' EI for Config {config_id} ({num_farms} farms) ---")

            policy_gdfs_config[policy_id] = []  # Ensure list is fresh for this policy
            policy_aggregate_npvs[policy_id] = 0.0  # Reset NPV sum

            for farm_id in range(1, num_farms + 1):
                farm_path = os.path.join(config_path, f"farm_{farm_id}")
                geojson_path = os.path.join(farm_path, "input.geojson")
                farm_results_dir = os.path.join(farm_path, "policy")  # Per-farm results dir
                os.makedirs(farm_results_dir, exist_ok=True)

                try:
                    policy_farm_gdf, policy_farm_data, policy_model = run_single_optimization_policy(
                        cfg,
                        geojson_path,
                        neighbor_dist,
                        exit_tol=exit_tol,
                        pc=penalty_coef,
                        policy_params=policy_params
                    )

                    if policy_farm_gdf is not None and policy_farm_data is not None:
                        # Add farm_id if not present
                        if 'farm_id' not in policy_farm_gdf.columns:
                            policy_farm_gdf['farm_id'] = farm_id
                        policy_gdfs_config[policy_id].append(policy_farm_gdf)
                        farm_npv = policy_farm_data.get('policy_npv', 0.0)
                        policy_aggregate_npvs[policy_id] += farm_npv if farm_npv is not None else 0.0
                        print(f"Farm {farm_id} Policy '{policy_id}' EI NPV: {farm_npv}")
                    else:
                        print(f"Warning: Policy '{policy_id}' EI run failed for farm {farm_id}.")
                except Exception as e:
                    print(f"Error running Policy '{policy_id}' EI for farm {farm_id}: {e}")
                    continue

            # --- 4. Combine Policy GDFs and run EC Repositioning ---
            print(
                f"\n--- Combining Policy '{policy_id}' Results for Config {config_id} and Running EC Repositioning ---")
            try:
                combined_policy_gdf = gpd.pd.concat(policy_gdfs_config[policy_id], ignore_index=True)
                combined_policy_geojson_path = os.path.join(config_results_dir,
                                                            f"config_{config_id}_combined_policy_{policy_id}.geojson")
                save_gdf_for_ec(combined_policy_gdf,
                                combined_policy_geojson_path)  # Save combined policy GDF

                ec_params_policy = {
                    'adjacency_dist': 0.0,
                    'boundary_seg_count': 10,
                    'interior_cell_count': 10,
                    'connectivity_metric': 'IIC',
                    'al_factor': 1e-9
                }

                policy_ec_plots = parse_geojson(combined_policy_geojson_path)

                print(f"Running solve_reposition_ilp for policy '{policy_id}'...")
                chosen_pieces_policy, optim_val_policy, conn_val_policy = solve_reposition_ilp(
                    policy_ec_plots,
                    **ec_params_policy
                )
                current_config_summary['policies'][policy_id] = {
                    'reposition_connectivity_score': conn_val_policy,
                    'reposition_optim_value': optim_val_policy,
                    'total_npv': policy_aggregate_npvs[policy_id]
                }
                print(
                    f"Config {config_id} Policy '{policy_id}' Reposition Connectivity ({ec_params_policy['connectivity_metric']}): {conn_val_policy:.4f}")
                print(
                    f"Config {config_id} Policy '{policy_id}' Reposition Objective Value: {optim_val_policy:.4f}")
                print(
                    f"Config {config_id} Aggregate Policy '{policy_id}' NPV: {policy_aggregate_npvs[policy_id]:.2f}")
            except Exception as e:
                print(
                    f"Error during policy '{policy_id}' aggregation or EC repositioning for config {config_id}: {e}")
                current_config_summary['policies'][policy_id] = {'error': str(e)}

        # --- Store results for the current config ---
        all_configs_aggregated_results[config_id] = current_config_summary
        print(f"===== Finished Processing Configuration: {config_id} =====")

        # --- 5. Final Aggregation and Visualization/Output ---
        print("\n\n===== Aggregated Results Across All Configurations =====")


    summary_data = []
    for config_id, results in all_configs_aggregated_results.items():
        baseline_conn = results.get('baseline', {}).get('connectivity_score', None)
        baseline_repos = results.get('baseline', {}).get('repositioning_score', None)
        baseline_npv = results.get('baseline', {}).get('total_npv', None)
        summary_data.append({
            'config_id': config_id,
            'policy_id': 'baseline',
            'total_npv': baseline_npv,
            'connectivity_score': baseline_conn,
            'repositioning_score': baseline_repos
        })

        for policy_id, policy_res in results.get('policies', {}).items():
            summary_data.append({
                'config_id': config_id,
                'policy_id': policy_id,
                'total_npv': policy_res.get('total_npv', None),
                'connectivity_score': policy_res.get('reposition_connectivity_score', None),
                'repositioning_score': None
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df[['config_id', 'policy_id', 'total_npv', 'connectivity_score', 'repositioning_score']]

        # Convert numeric columns safely, coercing errors to NaN for display
        num_cols = ['total_npv', 'connectivity_score', 'repositioning_score']
        for col in num_cols:
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')

        pd.options.display.float_format = '{:,.4f}'.format
        print(summary_df.to_string(index=False))

        # Optional: Save final summary to CSV
        final_summary_path = os.path.join(output_dir, "all_configs_policy_evaluation_summary.csv")
        try:
            summary_df.to_csv(final_summary_path, index=False, float_format='%.4f')
            print(f"\nFinal aggregated summary saved to: {final_summary_path}")
        except Exception as e:
            print(f"\nError saving final summary results to CSV: {e}")
    else:
        print("No results generated across configurations.")

    # --- 6. Generate Visualizations ---
    print("\n--- Generating Comparison Visualizations ---")
    # Ensure summary_df exists and potentially filter out error rows before visualizing
    if 'summary_df' in locals() and not summary_df.empty:
        # Convert numeric columns again just before visualization in case they were modified
        for col in num_cols:
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')

        visualize_policy_comparison(summary_df, output_dir)
    else:
        print("Visualization skipped as summary DataFrame is not available.")