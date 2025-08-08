import os
import json
import time
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import Config
import pyomo.environ as pyo
from utils.utils import get_margins_hab_fractions


def precompute_inputs(farm_gdf, params, neighbor_dist=15.0):
    """
    Gathers all geometry and problem data needed to build the Pyomo model:
    - plot indices
    - distances and neighbor sets
    - yields, areas, discount factors
    - time factor cache for each gamma

    In your real code, if you need 'distance to boundary' or 'random point' logic,
    you must do that *here* and store the results as numeric parameters to feed
    to Pyomo.
    """
    plot_ids = list(farm_gdf['id'])

    id_to_idx = {pid: i for i, pid in enumerate(plot_ids)}
    idx_to_id = {i: pid for i, pid in enumerate(plot_ids)}

    plot_ids = [p-1 for p in plot_ids]
    n_plots = len(plot_ids)

    # Extract centroids
    centroids = np.array([
        row.geometry.centroid.coords[0]
        for _, row in farm_gdf.iterrows()
    ])

    # Pairwise distances
    distances = np.zeros((n_plots, n_plots), dtype=float)
    for i in range(n_plots):
        for j in range(n_plots):
            if i == j:
                distances[i, j] = 0.0
            else:
                dx = centroids[i, 0] - centroids[j, 0]
                dy = centroids[i, 1] - centroids[j, 1]
                distances[i, j] = np.sqrt(dx * dx + dy * dy)

    # Create neighbor sets based on threshold
    neighbors = {}
    for i in range(n_plots):
        neighbors[i] = [
            j for j in range(n_plots)
            if (j != i and distances[i, j] < neighbor_dist)
        ]

    # Basic columns
    yields_ = farm_gdf['yield'].fillna(0.0).values
    areas = np.array([geom.area/10000 for geom in farm_gdf.geometry])
    plot_types = farm_gdf['type'].values
    if 'label' in farm_gdf.columns:
        plot_labels = farm_gdf['label'].values
    else:
        # Fallback if no label column:
        plot_labels = np.array(["Unknown"] * n_plots)

    # Time discounting
    r = params['r']
    T = params['t']
    t_arr = np.arange(1, T + 1)
    discount_factors = (1 + r) ** (-t_arr)

    # Precompute time-factor cache for each gamma
    gamma_values = set()
    for crop_def in params['crops'].values():
        gamma_values.add(crop_def['margin']['gamma'])
        gamma_values.add(crop_def['margin']['zeta'])
        gamma_values.add(crop_def['habitat']['gamma'])
        gamma_values.add(crop_def['habitat']['zeta'])

    # Store arrays of (1 - exp(-g * t)) if that is your formula
    time_factor_cache = {}
    for g in gamma_values:
        time_factor_cache[g] = (1 - np.exp(-g * t_arr))

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
        'params': params
    }
    return farm_data


def build_and_solve_pyomo_model(farm_data, params, penalty_coef, exit_tol):
    """
    Creates and solves a Pyomo model that replicates the margin/habitat logic
    from your original code.

    Returns the solved model.
    """
    plot_ids = farm_data['plot_ids']
    distances = farm_data['distances']
    neighbors = farm_data['neighbors']
    yields_ = farm_data['yields']
    areas = farm_data['areas']
    plot_types = farm_data['plot_types']
    plot_labels = farm_data['plot_labels']
    discount_factors_array = farm_data['discount_factors']
    time_factor_cache = farm_data['time_factor_cache']

    # Costs
    cost_margin_impl = params['costs']['margin']['implementation']
    cost_margin_maint = params['costs']['margin']['maintenance']
    cost_habitat_impl = params['costs']['habitat']['implementation']
    cost_habitat_maint = params['costs']['habitat']['maintenance']
    cost_existing_hab = params['costs']['habitat']['existing_hab']
    cost_ag_maint = params['costs']['agriculture']['maintenance']

    # Build the Pyomo model
    model = pyo.ConcreteModel("FarmOptimization")

    # Sets
    model.I = pyo.Set(initialize=plot_ids)

    # Subsets for convenience
    ag_plots = [i for i in plot_ids if plot_types[i] == 'ag_plot']
    hab_plots = [i for i in plot_ids if plot_types[i] == 'hab_plots']

    # Variables: margin[i], habitat[i] in [0,1]
    model.margin = pyo.Var(model.I, bounds=(0.0, 1.0), initialize=0.0)
    model.habitat = pyo.Var(model.I, bounds=(0.0, 1.0), initialize=0.0)

    # Numeric parameters
    model.yield_ = pyo.Param(model.I, initialize=lambda m, i: yields_[i], within=pyo.NonNegativeReals)
    model.area = pyo.Param(model.I, initialize=lambda m, i: areas[i], within=pyo.NonNegativeReals)

    # Distance param (2D)
    def dist_init(m, i, j):
        return distances[i, j]

    model.distance = pyo.Param(model.I, model.I, initialize=dist_init, within=pyo.NonNegativeReals, default=99999)

    # We’ll define an Expression for NPV[i] for each plot i
    @model.Expression(model.I)
    def NPV(m, i):
        """
        Computes the NPV for plot i, including:
          1) Continuous margin-based pollination/pest with distance-based exponentials,
          2) No more summing of time factors ahead of time – we use them per time step,
          3) Summation of pollination/pest across neighbors in a list (per time step),
          4) yield_loss_by_habitat explicitly included,
          5) revenue_t uses combined_yield[t_idx] and the discount factor discount_factors_array[t_idx].
        """
        p_type = plot_types[i]
        if p_type == 'ag_plot':
            # Identify the crop label to get alpha/beta/gamma, etc.
            c_label = plot_labels[i]
            crop_def = params['crops'][c_label]

            # Margin parameters
            alpha = crop_def['margin']['alpha']
            beta = crop_def['margin']['beta']
            gamma = crop_def['margin']['gamma']
            delta_ = crop_def['margin']['delta']
            epsilon_ = crop_def['margin']['epsilon']
            zeta_ = crop_def['margin']['zeta']

            # Habitat parameters
            hab_alpha = crop_def['habitat']['alpha']
            hab_beta = crop_def['habitat']['beta']
            hab_gamma = crop_def['habitat']['gamma']
            hab_delta = crop_def['habitat']['delta']
            hab_epsilon = crop_def['habitat']['epsilon']
            hab_zeta = crop_def['habitat']['zeta']

            # Extract the time-factor arrays (one entry per t) so we can use them per time step
            # e.g. time_factor_cache[gamma] is an array of length T
            margin_time_factors_gamma = time_factor_cache[gamma]
            margin_time_factors_zeta = time_factor_cache[zeta_]
            hab_time_factors_gamma = time_factor_cache[hab_gamma]
            hab_time_factors_zeta = time_factor_cache[hab_zeta]

            base_yield = m.yield_[i]
            p_c = crop_def['p_c']
            A = m.area[i]

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

            # Added explicitly: yield loss if we convert to habitat
            yield_loss_by_habitat = base_yield * p_c * A * m.habitat[i]

            # Build pollination_list[t_idx] and pest_list[t_idx], for each time step
            # We'll accumulate margin + habitat contributions from i itself and neighbors.
            T_len = len(discount_factors_array)  # or len(margin_time_factors_gamma), etc.

            # For final NPV
            npv_val = -impl_cost  # up-front cost (not discounted)

            # Loop over each time step t_idx
            for t_idx in range(T_len):
                # =========== 1) Compute total pollination at this time step ============
                pollination_t = 0.0
                # i's own margin effect (assuming distance[i,i] ~ 0 => exp(0)=1)
                pollination_t += alpha * m.margin[i] * pyo.exp(-beta * m.distance[i, i]) \
                                 * margin_time_factors_gamma[t_idx]

                # Also accumulate from neighbors for margin pollination & habitat effect
                # (We interpret "js that are margins" as j having m.margin[j] > 0, but in code
                #  we simply multiply by m.margin[j], which is zero if no margin is chosen.)
                for j in neighbors[i]:
                    dist_ij = m.distance[i, j]
                    # Margin-based neighbor (pollination)
                    if plot_types[j] == 'ag_plot':
                        # neighbor's margin effect
                        pollination_t += alpha * m.margin[j] * pyo.exp(-beta * dist_ij) \
                                         * margin_time_factors_gamma[t_idx]
                        pollination_t += (m.habitat[j] * hab_alpha
                                          * pyo.exp(-hab_beta * dist_ij)
                                          * hab_time_factors_gamma[t_idx])
                    elif plot_types[j] == 'hab_plots':
                        # existing habitat
                        pollination_t += hab_alpha * pyo.exp(-hab_beta * dist_ij) \
                                         * hab_time_factors_gamma[t_idx]

                # =========== 2) Compute total pest control at this time step ============
                pest_t = 0.0
                # i's own margin effect
                pest_t += delta_ * m.margin[i] * pyo.exp(-epsilon_ * m.distance[i, i]) \
                          * margin_time_factors_zeta[t_idx]

                for j in neighbors[i]:
                    dist_ij = m.distance[i, j]
                    # neighbor's margin-based pest
                    if plot_types[j] == 'ag_plot':
                        pest_t += delta_ * m.margin[j] * pyo.exp(-epsilon_ * dist_ij) \
                                  * margin_time_factors_zeta[t_idx]
                        pest_t += (m.habitat[j] * hab_delta
                                   * pyo.exp(-hab_epsilon * dist_ij)
                                   * hab_time_factors_zeta[t_idx])
                    elif plot_types[j] == 'hab_plots':
                        pest_t += hab_delta * pyo.exp(-hab_epsilon * dist_ij) \
                                  * hab_time_factors_zeta[t_idx]

                # =========== 3) Combine yield at time t_idx ============
                # base_yield * (1 + pollination_t + pest_t)
                combined_yield_t = base_yield * (1.0 + pollination_t + pest_t)

                # =========== 4) Compute revenue at time t_idx ============
                # revenue_t = combined_yield[t_idx] * p_c * A * (1 - m.habitat[i])
                revenue_t = combined_yield_t * p_c * A * (1 - m.habitat[i])

                # =========== 5) Subtract maintenance + yield_loss_by_habitat,
                #     and discount this year's flow ============
                yearly_cf = revenue_t - total_maint - yield_loss_by_habitat
                df = discount_factors_array[t_idx]
                npv_val += yearly_cf * df
            return npv_val

        elif p_type == 'hab_plots':
            # "existing habitat" – negative cost each year
            npv_val = 0.0
            A = m.area[i]
            for t_idx, df in enumerate(discount_factors_array):
                npv_val += -cost_existing_hab * A * df
            return npv_val

        else:
            # If there's a third type, set NPV=0
            return 0.0

    @model.Expression()
    def penalty(m):
        return sum((m.margin[i]) ** 2 + (m.habitat[i]) ** 2 for i in m.I)

    # Objective: maximize sum of NPV
    def total_npv_rule(m):
        return pyo.summation(m.NPV) - penalty_coef * m.penalty

    model.Obj = pyo.Objective(rule=total_npv_rule, sense=pyo.maximize)

    # Solve
    solver = pyo.SolverFactory("ipopt")
    solver.options['acceptable_tol'] = exit_tol
    result = solver.solve(model, tee=False)
    model.solutions.load_from(result)
    return model


def assign_pyomo_solution_to_gdf(model, farm_gdf):
    """
    Extracts the Pyomo decision variables (margin, habitat) and saves them
    into the GeoDataFrame as new columns.
    """
    margin_vals = []
    habitat_vals = []
    plot_ids = list(farm_gdf['id'])
    plot_ids = [p - 1 for p in plot_ids]
    for i in plot_ids:
        margin_vals.append(pyo.value(model.margin[i]))
        habitat_vals.append(pyo.value(model.habitat[i]))

    farm_gdf['margin_intervention'] = margin_vals
    farm_gdf['habitat_conversion'] = habitat_vals
    return farm_gdf


def apply_threshold_and_save(farm_gdf, image_path, output_json, threshold=0.01):
    """
    Applies a threshold to 'margin_intervention' and 'habitat_conversion',
    then saves a final GeoJSON with only the "active" interventions.
    Also returns that filtered GDF.
    """
    # Clip any fractional interventions below threshold
    farm_gdf['margin_intervention'] = np.where(
        (farm_gdf['type'] == 'ag_plot') & (farm_gdf['margin_intervention'] >= threshold),
        farm_gdf['margin_intervention'],
        0.0
    )
    farm_gdf['habitat_conversion'] = np.where(
        (farm_gdf['type'] == 'ag_plot') & (farm_gdf['habitat_conversion'] >= threshold),
        farm_gdf['habitat_conversion'],
        0.0
    )

    margin_lines_gdf, converted_polys_gdf = get_margins_hab_fractions(farm_gdf)
    visualize_optimized_farm_refactored(farm_gdf, margin_lines_gdf, converted_polys_gdf, image_path)


    # If you only want to save those that actually had an intervention:
    filtered = farm_gdf.loc[
        (farm_gdf['type'] == "ag_plot") &
        ((farm_gdf['margin_intervention'] > 0.0) | (farm_gdf['habitat_conversion'] > 0.0))
        ].copy()

    filtered.to_file(output_json, driver="GeoJSON")

    # Optionally remove geometry from the final JSON
    with open(output_json) as f:
        data_json = json.load(f)
    for feat in data_json["features"]:
        if "geometry" in feat:
            del feat["geometry"]
    with open(output_json, 'w') as f:
        json.dump(data_json, f)

    return farm_gdf


def visualize_optimized_farm_refactored(farm_gdf, margin_lines_gdf, converted_polys_gdf, image_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')

    farm_gdf.boundary.plot(ax=ax, color='grey', linewidth=0.8, aspect=1)
    farm_gdf = farm_gdf.rename_geometry("farm_geom")
    combined_gdf = farm_gdf.copy()

    if margin_lines_gdf is not None:
        margin_lines_gdf.plot(ax=ax, color='#e41a1c', linewidth=2, aspect=1)  # ColorBrewer Set1 Red
        margin_lines_gdf["margin_wkt"] = margin_lines_gdf.geometry.to_wkt()
        margin_lines_gdf = margin_lines_gdf.drop(columns="geometry")
        common_cols = set(farm_gdf.columns).intersection(margin_lines_gdf.columns) - {'id', 'margin_wkt'}
        margin_lines_gdf = margin_lines_gdf.drop(columns=common_cols)
        combined_gdf = combined_gdf.merge(margin_lines_gdf[['id', 'margin_wkt']], on='id', how='left')

    if converted_polys_gdf is not None:
        converted_polys_gdf.plot(ax=ax, color='#4daf4a', alpha=0.65, edgecolor='darkgreen',
                                 linewidth=0.5, aspect=1)  # ColorBrewer Set1 Green
        converted_polys_gdf["converted_wkt"] = converted_polys_gdf.geometry.to_wkt()
        converted_polys_gdf = converted_polys_gdf.drop(columns="geometry")
        common_cols = set(farm_gdf.columns).intersection(converted_polys_gdf.columns) - {'id', 'converted_wkt'}
        converted_polys_gdf = converted_polys_gdf.drop(columns=common_cols)
        combined_gdf = combined_gdf.merge(converted_polys_gdf[['id', 'converted_wkt']], on='id', how='left')


    if 'type' in farm_gdf.columns:
        hab_plots_gdf = farm_gdf[farm_gdf['type'] == 'hab_plots']
        if not hab_plots_gdf.empty:
            hab_plots_gdf.plot(ax=ax, color='#377eb8', alpha=0.65, edgecolor='darkblue',
                               linewidth=0.5, aspect=1)  # ColorBrewer Set1 Blue

    # Add legend using the updated colors
    patches = [
        mpatches.Patch(color='#e41a1c', label='Margin Interventions') if margin_lines_gdf is not None else None,
        mpatches.Patch(color='#4daf4a', label='Habitat Conversions') if converted_polys_gdf is not None else None,
        mpatches.Patch(color='#377eb8', label='Existing Habitats') if 'type' in farm_gdf.columns and not farm_gdf[
            farm_gdf['type'] == 'hab_plots'].empty else None,
        mpatches.Patch(color='darkgrey', label='Field Boundaries')
    ]
    # Filter out None patches before creating the legend
    valid_patches = [p for p in patches if p is not None]
    if valid_patches:
        plt.legend(handles=valid_patches, loc='best', fontsize='small', frameon=True, framealpha=0.8)

    # Clean up axis appearance for a map-like look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.tight_layout()

    # Save with higher resolution and tight bounding box
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory


def main_run_pyomo(cfg, geojson_path, image_path, output_json, neighbor_dist, exit_tol, penalty_coef):
    """
    Reads the farm file, precomputes geometry-based parameters, builds and
    solves the Pyomo model, then saves the results.
    """
    # Read farm data
    farm_gdf = gpd.read_file(geojson_path)
    params = cfg.params

    # Precompute
    farm_data = precompute_inputs(farm_gdf, params, neighbor_dist=neighbor_dist)

    # Solve
    start = time.process_time()
    model = build_and_solve_pyomo_model(farm_data, params, penalty_coef=penalty_coef, exit_tol=exit_tol)
    elapsed = time.process_time() - start
    print(f"Solve time: {elapsed:.2f} seconds")

    # Assign solution
    farm_gdf = assign_pyomo_solution_to_gdf(model, farm_gdf)

    # Save final result
    farm_gdf_processed = apply_threshold_and_save(farm_gdf, image_path, output_json, threshold=0.01)
    return farm_gdf_processed


if __name__ == "__main__":
    cfg = Config()
    exit_tol = 1e-6
    neighbor_dist = 1500
    penalty_coef = 1e5

    type_ = "syn_farms"
    farm_dir = os.path.join(cfg.data_dir, "crop_inventory", type_)
    farm_ids = np.arange(6, 11)
    results_list = []
    for farm_id in farm_ids:
        print(f"Running farm: {farm_id}")
        farm_path = os.path.join(farm_dir, f"farm_{farm_id}")
        if not os.path.exists(farm_path):
            continue
        #geojson_path = os.path.join(farm_path, f"farm_{farm_id}_filled.geojson")
        geojson_path = os.path.join(farm_path, "input.geojson")

        image_path = os.path.join(farm_dir, "plots", "ei", f"farm_{farm_id}_output_gt.svg")
        output_json = os.path.join(farm_dir, "plots", "ei", f"farm_{farm_id}_output_gt.geojson")

        result_gdf = main_run_pyomo(cfg, geojson_path, image_path, output_json, neighbor_dist, exit_tol, penalty_coef)
        if result_gdf is not None:
            results_list.append(result_gdf)
