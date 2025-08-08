import os.path
from config import Config
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import io
from utils.utils import plot_farm, get_margins_hab_fractions, plot_combined
import numpy as np
import pandas as pd
import seaborn as sns
from math import pi
from collections import defaultdict

FONT_SIZE = 25
plt.rcParams['axes.labelsize'] = FONT_SIZE  # For x and y labels
plt.rcParams['axes.titlesize'] = FONT_SIZE  # For the subplot title
plt.rcParams['xtick.labelsize'] = FONT_SIZE  # For x-axis tick labels
plt.rcParams['ytick.labelsize'] = FONT_SIZE  # For y-axis tick labels
plt.rcParams['legend.fontsize'] = FONT_SIZE  # For the legend
plt.rcParams['figure.titlesize'] = FONT_SIZE

def plot_farms(config_id):
    gdf = gpd.read_file(os.path.join(base_farm_dir, f"config_{config_id}", "farms.geojson"))
    gdf['id'] = gdf['id'].astype('category')
    plt.style.use('seaborn-v0_8-ticks')
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    gdf.plot(column='id', cmap='viridis',
             ax=ax,
             edgecolor='black',
             linewidth=0.8,
             legend=True,
             aspect=1,
             legend_kwds={
                 'title': "Farm ID",
                 'loc': 'upper left',
                 'bbox_to_anchor': (1.02, 1),
                 'fontsize': fontsize,
                 'title_fontsize': fontsize
             })

    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X-Coordinate', fontsize=fontsize)
    ax.set_ylabel('Y-Coordinate', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(base_farm_dir, f"config_{config_id}", "farms.svg"))
    plt.close(fig)


def plot_plots(config_id):
    gdf_farms = gpd.read_file(io.StringIO(os.path.join(base_farm_dir, f"config_{config_id}", "farms.geojson")))
    gdf_plots = gpd.read_file(io.StringIO(os.path.join(base_farm_dir, f"config_{config_id}", "all_subplots.geojson")))

    gdf_plots['label'] = gdf_plots['label'].astype('category')

    plt.style.use('seaborn-v0_8-ticks')
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    gdf_farms.plot(ax=ax,
                   facecolor='none',  # No fill color
                   edgecolor='darkgrey',  # Outline color
                   linestyle='--',  # Dashed line
                   linewidth=1.0,  # Line thickness
                   aspect=1,
                   label='_nolegend_')  # Exclude from legend

    gdf_plots.plot(column='label',
                   cmap='Paired',  # Using 'Paired' colormap - good for categories
                   ax=ax,
                   edgecolor='black',
                   linewidth=0.8,
                   legend=True,
                   aspect=1,
                   legend_kwds={
                       'title': "Plot Label",  # Legend title
                       'loc': 'upper left',
                       'bbox_to_anchor': (1.02, 1),
                       'fontsize': fontsize,
                       'title_fontsize': fontsize
                   })

    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X-Coordinate', fontsize=fontsize)
    ax.set_ylabel('Y-Coordinate', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(base_farm_dir, f"config_{config_id}", "plots.svg"))
    plt.close(fig)


def farm_combine_interventions(farm_path, mode):
    farm_gdf = gpd.read_file(os.path.join(farm_path, "input.geojson"))
    interventions = gpd.read_file(os.path.join(farm_path, "output_gt.geojson"))
    interventions = interventions.drop(["geometry"], axis=1)
    common_cols = set(farm_gdf.columns).intersection(interventions.columns) - {'id'}
    interventions_subset = interventions.drop(columns=common_cols)
    if len(interventions_subset) == 0:
        farm_gdf["margin_intervention"] = 0
        farm_gdf["habitat_conversion"] = 0
    else:
        farm_gdf = farm_gdf.merge(interventions_subset, on="id", how="left")
    farm_gdf = farm_gdf.fillna(0)
    margin_lines_gdf, converted_polys_gdf = get_margins_hab_fractions(farm_gdf)
    plot_farm(farm_path, farm_gdf, margin_lines_gdf, converted_polys_gdf, mode=mode)


def plot_all():
    mode = "og"

    for config_id in num_configs:
        if config_id != 15:
            continue
        print(f"Running config: {config_id}")
        config_path = os.path.join(base_farm_dir, f"config_{config_id}")
        farm_folders = [f for f in os.listdir(config_path)
                        if os.path.isdir(os.path.join(config_path, f)) and f.startswith("farm_")]
        farm_ids = sorted([int(f.split('_')[1]) for f in farm_folders])

        if not os.listdir(config_path):
            continue

        plot_farms(config_id)
        plot_plots(config_id)

        for farm_id in farm_ids:
            if farms == "syn_farms":
                farm_path = os.path.join(config_path, f"farm_{farm_id}")
            else:
                farm_path = os.path.join(config_path, f"farm_{farm_id}", "ei")

            farm_combine_interventions(farm_path, mode)

        gdfs = []
        for farm_id in farm_ids:
            if farms == "syn_farms":
                farm_path = os.path.join(config_path, f"farm_{farm_id}")
            else:
                farm_path = os.path.join(config_path, f"farm_{farm_id}", "ei")

            gdf = gpd.read_file(os.path.join(farm_path, f"combined_{mode}.geojson"))
            gdf["farm_id"] = farm_id
            gdfs.append(gdf)

        combined_gdf = pd.concat(gdfs, ignore_index=True)
        combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry="geometry")
        combined_gdf.to_file(os.path.join(config_path, f"all_plots_interventions_{mode}.geojson"), driver="GeoJSON")

        plot_combined(config_path, mode)


def plot_policy():
    def get_readable_name(param_name):
        """Converts internal parameter names to human-readable labels."""
        # Specific handling for eco_premium factors
        if param_name.startswith("eco_premium_factor_"):
            # Assumes crop name follows "eco_premium_factor_"
            crop_name_parts = param_name.split("eco_premium_factor_")[1].split('_')
            crop_name = " ".join(word.capitalize() for word in crop_name_parts)
            return f"Eco Premium ({crop_name})"

        # General mapping for other known parameter names
        name_map = {
            "adj_hab_factor_margin": "Adj. Habitat Factor (Margin)",
            "adj_hab_factor_habitat": "Adj. Habitat Factor (Habitat)",
            "maint_subsidy_factor_margin": "Maint. Subsidy (Margin)",
            "maint_subsidy_factor_habitat": "Maint. Subsidy (Habitat)",
            "hab_per_ha": "Payment (Habitat/ha)",
            "min_total_hab_area": "Min. Total Habitat Area",
            "min_margin_frac_adj_hab": "Min. Margin Frac. Adj. Habitat",
            # Performance metrics often used in correlations
            "avg_conn": "Avg Connectivity",
            "avg_farm_npv": "Avg Farm NPV",
            "avg_policy_cost": "Avg Policy Cost",
            "avg_conn_diff": "Avg Connectivity Diff.",
            "avg_npv_diff": "Avg NPV Diff."
        }
        if param_name in name_map:
            return name_map[param_name]

        # Default formatting if not in map (e.g., for other params_list items)
        # This part might not be hit if all names are covered above or are eco_premiums
        parts = param_name.split('_')
        return " ".join(word.capitalize() for word in parts)

    file_name = "bo_results_avg_cfg_obj.json"
    results_file_path = os.path.join(syn_farm_dir, "plots", "policy", "avg_budget_500k", file_name)
    output_plot_dir = os.path.join(syn_farm_dir, "plots", "policy", "avg_budget_500k", "revised_analysis_plots")
    os.makedirs(output_plot_dir, exist_ok=True)

    print(f"--- Loading BO Results from: {results_file_path} ---")
    try:
        with open(results_file_path, 'r') as f:
            results_data = json.load(f)
    except Exception as e:
        print(f"Error loading or parsing JSON: {e}")
        return

    top_policies_df_orig = pd.DataFrame(results_data.get('top_policies_evaluated_for_plot', []))
    if top_policies_df_orig.empty:
        print("No top policies data found.")
        return

    top_policies_df = top_policies_df_orig.copy()
    parameter_order = results_data.get('parameter_order', [])
    if not parameter_order:
        print("Parameter order not found in results data.")
        return

    if 'params_list' not in top_policies_df.columns:
        print("Column 'params_list' not found in top_policies_evaluated_for_plot.")
        return

    valid_params_lists = []
    for p_list in top_policies_df['params_list']:
        if isinstance(p_list, list) and len(p_list) == len(parameter_order):
            valid_params_lists.append(p_list)
        else:
            valid_params_lists.append([np.nan] * len(parameter_order))

    params_values_df = pd.DataFrame(valid_params_lists, columns=parameter_order, index=top_policies_df.index)

    # --- 1. Correlation Heatmap ---
    print("\n--- Generating Correlation Heatmap ---")
    metrics_to_correlate = ['avg_conn', 'avg_farm_npv', 'avg_policy_cost', 'avg_conn_diff', 'avg_npv_diff']
    existing_metrics = [col for col in metrics_to_correlate if col in top_policies_df.columns]

    if existing_metrics and not params_values_df.empty:
        analysis_df_corr = pd.concat([params_values_df, top_policies_df.loc[params_values_df.index, existing_metrics]],
                                     axis=1)
        analysis_df_corr.dropna(inplace=True)

        if not analysis_df_corr.empty:
            readable_column_names_corr = [get_readable_name(col) for col in analysis_df_corr.columns]
            analysis_df_corr_renamed = analysis_df_corr.copy()
            analysis_df_corr_renamed.columns = readable_column_names_corr

            correlation_matrix = analysis_df_corr_renamed.corr()

            plt.figure(figsize=(
            max(12, len(analysis_df_corr_renamed.columns) * 0.7), max(10, len(analysis_df_corr_renamed.columns) * 0.6)))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
            plt.title("Correlation Matrix of Policy Parameters and Performance Metrics", fontsize=14)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            plot_filename = os.path.join(output_plot_dir, "analyzed_correlation_heatmap.svg")
            try:
                plt.savefig(plot_filename, bbox_inches='tight')
                print(f"Saved: {plot_filename}")
            except Exception as e:
                print(f"Error saving correlation heatmap: {e}")
            plt.close()
        else:
            print("Skipping correlation heatmap: DataFrame empty after processing NaNs.")
    else:
        print("Skipping correlation heatmap: No performance metrics or valid parameter values.")

    # --- 2. Parameter Distribution Box Plots ---
    print("\n--- Generating Parameter Distribution Box Plots ---")
    param_groups_box = {
        "Factors (0-1 Scale)": [p for p in parameter_order if
                                ("factor" in p or "frac" in p or "subsidy" in p) and "eco_premium" not in p],
        "Eco Premiums (approx. 1-1.3 Scale)": [p for p in parameter_order if "eco_premium_factor" in p],
        "Habitat Payment (e.g., 0-150 Scale)": [p for p in parameter_order if p == "hab_per_ha"],
        "Habitat Area (e.g., 0-10 Scale)": [p for p in parameter_order if p == "min_total_hab_area"]
    }

    valid_param_groups_box = {}
    for group_name, params_in_group in param_groups_box.items():
        valid_params = [p for p in params_in_group if
                        p in params_values_df.columns and params_values_df[p].notna().any()]
        if valid_params:
            valid_param_groups_box[group_name] = valid_params

    if not valid_param_groups_box:
        print("No valid parameter groups found for box plots.")
    else:
        num_groups_box = len(valid_param_groups_box)
        fig_boxplots, axes_boxplots = plt.subplots(num_groups_box, 1, figsize=(12, 4 * num_groups_box + 2),
                                                   squeeze=False)

        for i, (group_name, params_in_group) in enumerate(valid_param_groups_box.items()):
            ax = axes_boxplots[i, 0]
            data_to_plot = params_values_df[params_in_group].copy()

            if data_to_plot.empty or data_to_plot.isnull().all().all():
                ax.text(0.5, 0.5, "No data available for this group", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(group_name, fontsize=25)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            if len(params_in_group) > 1:
                data_to_plot_melted = data_to_plot.melt(var_name='Parameter', value_name='Value')
                sns.boxplot(data=data_to_plot_melted, x='Value', y='Parameter', ax=ax, orient='h', whis=[5, 95],
                            palette="Set3")
                ax.set_yticklabels([get_readable_name(p) for p in data_to_plot.columns], fontsize=25)
            elif len(params_in_group) == 1:
                sns.boxplot(data=data_to_plot, x=params_in_group[0], ax=ax, orient='h', whis=[5, 95], palette="Set3")
                ax.set_ylabel(get_readable_name(params_in_group[0]), fontsize=25)

            ax.set_title(group_name, fontsize=25)
            ax.tick_params(axis='x', labelsize=25)
            ax.grid(axis='x', linestyle='--', alpha=0.7)

        fig_boxplots.suptitle("Parameter Distributions for Top Policies (Grouped by Scale)", fontsize=25, y=1.0)
        fig_boxplots.tight_layout(rect=[0, 0, 1, 0.97])
        plot_filename_box = os.path.join(output_plot_dir, "analyzed_parameter_boxplots_grouped.svg")
        try:
            fig_boxplots.savefig(plot_filename_box, bbox_inches='tight')
            print(f"Saved: {plot_filename_box}")
        except Exception as e:
            print(f"Error saving box plots: {e}")
        plt.close(fig_boxplots)

        # --- 3. Single Radar Plot for All Parameters (with range in labels) ---
        print("\n--- Generating Single Radar Plot for All Parameters ---")
        baseline_avg_conn = results_data.get('overall_avg_baseline_connectivity')

        if baseline_avg_conn is None:
            print("Baseline average connectivity not found. Cannot accurately identify policies for radar plot.")
        else:
            df_radar_selection = top_policies_df.copy()
            for param in parameter_order:
                if param in params_values_df.columns:
                    df_radar_selection[param] = params_values_df.loc[df_radar_selection.index, param]
            df_radar_selection.dropna(subset=parameter_order, how='any', inplace=True)

            required_cols_radar_metrics = ['avg_conn', 'avg_policy_cost', 'avg_farm_npv', 'avg_conn_diff',
                                           'avg_npv_diff']
            # Also check for 'policy_id' if it's used for saving original ID
            if 'policy_id' not in df_radar_selection.columns and 'policy_id' in top_policies_df_orig.columns:
                df_radar_selection['policy_id'] = top_policies_df_orig.loc[df_radar_selection.index, 'policy_id']

            missing_metric_cols = [col for col in required_cols_radar_metrics if col not in df_radar_selection.columns]

            if missing_metric_cols:
                print(
                    f"Missing performance columns for radar plot policy selection: {missing_metric_cols}. Skipping radar plot.")
            elif df_radar_selection.empty:
                print("No valid policies remaining after processing NaNs in parameter values for radar plot.")
            else:
                # criteria_to_index_map will store {'Criterion Label': policy_index_if_found_else_None}
                criteria_to_index_map = {}

                print(
                    "\n--- Debugging Policy Selection for Radar Plot (with +/-100 tolerance, no number prefix in labels) ---")

                # Define descriptive labels without numbering
                label_closest_conn = "Closest Conn to Baseline"
                label_closest_conn_min_cost = "Closest Conn (+-100), Min Cost"
                label_closest_conn_max_npv = "Closest Conn (+-100), Max NPV"
                label_min_conn_diff = "Min Conn Diff"
                label_min_conn_diff_min_cost = "Min Conn Diff Range, Min Cost"
                label_min_npv_diff = "Min NPV Diff"
                label_min_combined_diff = "Min Combined (Conn+NPV) Diff"

                # 1. Closest Conn to Baseline
                if 'avg_conn' in df_radar_selection and df_radar_selection['avg_conn'].notna().any():
                    df_radar_selection.loc[:, 'conn_abs_diff_from_baseline'] = (
                                df_radar_selection['avg_conn'] - baseline_avg_conn).abs()
                    if df_radar_selection['conn_abs_diff_from_baseline'].notna().any():
                        idx1 = df_radar_selection['conn_abs_diff_from_baseline'].idxmin()
                        criteria_to_index_map[label_closest_conn] = idx1
                        print(f"Criterion '{label_closest_conn}': Selected Policy Index {idx1}")

                        avg_conn_of_idx1 = df_radar_selection.loc[idx1, 'avg_conn']
                        lower_bound_conn = avg_conn_of_idx1 - 100
                        upper_bound_conn = avg_conn_of_idx1 + 100
                        candidates23 = df_radar_selection[
                            (df_radar_selection['avg_conn'] >= lower_bound_conn) &
                            (df_radar_selection['avg_conn'] <= upper_bound_conn)
                            ].copy()

                        if not candidates23.empty:
                            # 2. Closest Conn (within tolerance), Min Cost
                            if 'avg_policy_cost' in candidates23 and candidates23['avg_policy_cost'].notna().any():
                                idx2 = candidates23['avg_policy_cost'].idxmin()
                                criteria_to_index_map[label_closest_conn_min_cost] = idx2
                                print(f"Criterion '{label_closest_conn_min_cost}': Selected Policy Index {idx2}")
                            else:
                                criteria_to_index_map[label_closest_conn_min_cost] = None; print(
                                    f"Debug: Policy for criterion '{label_closest_conn_min_cost}' not found.")

                            # 3. Closest Conn (within tolerance), Max NPV
                            if 'avg_farm_npv' in candidates23 and candidates23['avg_farm_npv'].notna().any():
                                idx3 = candidates23['avg_farm_npv'].idxmax()
                                criteria_to_index_map[label_closest_conn_max_npv] = idx3
                                print(f"Criterion '{label_closest_conn_max_npv}': Selected Policy Index {idx3}")
                            else:
                                criteria_to_index_map[label_closest_conn_max_npv] = None; print(
                                    f"Debug: Policy for criterion '{label_closest_conn_max_npv}' not found.")
                        else:
                            criteria_to_index_map[label_closest_conn_min_cost] = None;
                            criteria_to_index_map[label_closest_conn_max_npv] = None
                            print(
                                f"Debug: Policies for criteria '{label_closest_conn_min_cost}' & '{label_closest_conn_max_npv}' not found due to empty candidates23.")
                    else:
                        criteria_to_index_map[label_closest_conn] = None; print(
                            f"Debug: Policy for criterion '{label_closest_conn}' not found.")
                else:
                    criteria_to_index_map[label_closest_conn] = None; print(
                        f"Debug: Policy for criterion '{label_closest_conn}' not found.")

                # 4. Min Conn Diff
                if 'avg_conn_diff' in df_radar_selection and df_radar_selection['avg_conn_diff'].notna().any():
                    idx4 = df_radar_selection['avg_conn_diff'].idxmin()
                    criteria_to_index_map[label_min_conn_diff] = idx4
                    print(f"Criterion '{label_min_conn_diff}': Selected Policy Index {idx4}")

                    min_conn_diff_val_for_p5 = df_radar_selection.loc[idx4, 'avg_conn_diff']
                    lower_bound_conn_diff_p5 = min_conn_diff_val_for_p5
                    upper_bound_conn_diff_p5 = min_conn_diff_val_for_p5 + 100
                    candidates5 = df_radar_selection[
                        (df_radar_selection['avg_conn_diff'] >= lower_bound_conn_diff_p5) &
                        (df_radar_selection['avg_conn_diff'] <= upper_bound_conn_diff_p5)
                        ].copy()

                    if not candidates5.empty:
                        # 5. Min Conn Diff (within tolerance), Min Cost
                        if 'avg_policy_cost' in candidates5 and candidates5['avg_policy_cost'].notna().any():
                            idx5 = candidates5['avg_policy_cost'].idxmin()
                            criteria_to_index_map[label_min_conn_diff_min_cost] = idx5
                            print(f"Criterion '{label_min_conn_diff_min_cost}': Selected Policy Index {idx5}")
                        else:
                            criteria_to_index_map[label_min_conn_diff_min_cost] = None; print(
                                f"Debug: Policy for criterion '{label_min_conn_diff_min_cost}' not found.")
                    else:
                        criteria_to_index_map[label_min_conn_diff_min_cost] = None; print(
                            f"Debug: Policy for criterion '{label_min_conn_diff_min_cost}' not found due to empty candidates5.")
                else:
                    criteria_to_index_map[label_min_conn_diff] = None;
                    criteria_to_index_map[label_min_conn_diff_min_cost] = None
                    print(
                        f"Debug: Policies for criteria '{label_min_conn_diff}' & '{label_min_conn_diff_min_cost}' not found.")

                # 6. Min NPV Diff
                if 'avg_npv_diff' in df_radar_selection and df_radar_selection['avg_npv_diff'].notna().any():
                    idx6 = df_radar_selection['avg_npv_diff'].idxmin()
                    criteria_to_index_map[label_min_npv_diff] = idx6
                    print(f"Criterion '{label_min_npv_diff}': Selected Policy Index {idx6}")
                else:
                    criteria_to_index_map[label_min_npv_diff] = None; print(
                        f"Debug: Policy for criterion '{label_min_npv_diff}' not found.")

                # 7. Min Combined (Conn+NPV) Diff
                if ('avg_conn_diff' in df_radar_selection and df_radar_selection['avg_conn_diff'].notna().any() and \
                        'avg_npv_diff' in df_radar_selection and df_radar_selection['avg_npv_diff'].notna().any()):
                    temp_df_for_norm = df_radar_selection[['avg_conn_diff', 'avg_npv_diff']].copy()
                    for diff_col in ['avg_conn_diff', 'avg_npv_diff']:
                        min_d, max_d = temp_df_for_norm[diff_col].min(), temp_df_for_norm[diff_col].max()
                        if max_d - min_d == 0:
                            temp_df_for_norm[f'norm_{diff_col}'] = 0.0
                        else:
                            temp_df_for_norm[f'norm_{diff_col}'] = (temp_df_for_norm[diff_col] - min_d) / (
                                        max_d - min_d)

                    temp_df_for_norm['combined_diff_score'] = temp_df_for_norm.get('norm_avg_conn_diff',
                                                                                   0) + temp_df_for_norm.get(
                        'norm_avg_npv_diff', 0)

                    if 'combined_diff_score' in temp_df_for_norm and temp_df_for_norm[
                        'combined_diff_score'].notna().any():
                        idx7 = temp_df_for_norm['combined_diff_score'].idxmin()
                        criteria_to_index_map[label_min_combined_diff] = idx7
                        print(f"Criterion '{label_min_combined_diff}': Selected Policy Index {idx7}")
                    else:
                        criteria_to_index_map[label_min_combined_diff] = None; print(
                            f"Debug: Policy for criterion '{label_min_combined_diff}' not found.")
                else:
                    criteria_to_index_map[label_min_combined_diff] = None; print(
                        f"Debug: Policy for criterion '{label_min_combined_diff}' not found.")

                # Consolidate policies for plotting and saving
                index_to_criteria_labels = defaultdict(list)
                for criterion_label, policy_idx in criteria_to_index_map.items():
                    if policy_idx is not None and policy_idx in df_radar_selection.index:
                        index_to_criteria_labels[policy_idx].append(criterion_label)

                final_policies_for_plot = {}
                radar_policies_summary_for_json = []

                for policy_idx, list_of_labels in index_to_criteria_labels.items():
                    policy_data_series = df_radar_selection.loc[policy_idx].copy()
                    # Sort descriptive labels alphabetically for consistent consolidated label
                    sorted_labels = sorted(list_of_labels)
                    consolidated_label = " & ".join(sorted_labels)
                    final_policies_for_plot[consolidated_label] = policy_data_series

                    # Prepare data for JSON saving
                    params_dict = {p: policy_data_series[p] for p in parameter_order if p in policy_data_series}
                    metrics_dict = {m: policy_data_series[m] for m in required_cols_radar_metrics if
                                    m in policy_data_series}
                    for calc_metric in ['conn_abs_diff_from_baseline', 'norm_avg_conn_diff', 'norm_avg_npv_diff',
                                        'combined_diff_score']:
                        if calc_metric in policy_data_series:  # Add these calculated metrics as well
                            metrics_dict[calc_metric] = policy_data_series[calc_metric]

                    radar_policies_summary_for_json.append({
                        "criteria_met_by_policy": consolidated_label,
                        "original_policy_id_if_available": policy_data_series.get('policy_id', f"Index_{policy_idx}"),
                        "parameters": params_dict,
                        "performance_metrics": metrics_dict
                    })

                print(f"\nTotal unique policies for radar plotting: {len(final_policies_for_plot)}")
                print(f"Consolidated labels for legend: {list(final_policies_for_plot.keys())}")

                # Save the selected policies summary to JSON
                summary_filename = os.path.join(output_plot_dir, "radar_selected_policies_summary.json")
                try:
                    # Custom default function to handle numpy types for JSON serialization
                    def np_encoder(object):
                        if isinstance(object, (np.generic, np.ndarray)):
                            return object.item() if isinstance(object, np.generic) else object.tolist()
                        raise TypeError(f"Object of type {type(object)} is not JSON serializable")

                    with open(summary_filename, 'w') as f:
                        json.dump(radar_policies_summary_for_json, f, indent=4, default=np_encoder)
                    print(f"Saved selected policies summary to: {summary_filename}")
                except Exception as e:
                    print(f"Error saving selected policies summary: {e}")

                if not final_policies_for_plot:
                    print("No specific policies were identified for the radar plot after consolidation.")
                else:
                    all_radar_params_to_plot = [p for p in parameter_order if p in df_radar_selection.columns]

                    if not all_radar_params_to_plot:
                        print("No valid parameters found for radar plot axes after filtering.")
                    elif len(all_radar_params_to_plot) < 3:
                        print(
                            f"Skipping single radar plot due to insufficient parameters ({len(all_radar_params_to_plot)}). Needs at least 3.")
                    else:
                        readable_radar_params_with_ranges = []
                        for param in all_radar_params_to_plot:
                            min_val = df_radar_selection[param].min()
                            max_val = df_radar_selection[param].max()
                            base_readable_name = get_readable_name(param)
                            if pd.notna(min_val) and pd.notna(max_val):
                                readable_radar_params_with_ranges.append(
                                    f"{base_readable_name}\n[{min_val:.2g}-{max_val:.2g}]")  # Use .2g for general format
                            else:
                                readable_radar_params_with_ranges.append(f"{base_readable_name}\n[N/A]")

                        num_vars = len(all_radar_params_to_plot)
                        normalized_params_for_radar = pd.DataFrame(index=final_policies_for_plot.keys(),
                                                                   columns=all_radar_params_to_plot)

                        for param in all_radar_params_to_plot:
                            min_val = df_radar_selection[param].min()
                            max_val = df_radar_selection[param].max()
                            range_val = max_val - min_val
                            if range_val == 0 or pd.isna(range_val):
                                normalized_params_for_radar[param] = 0.5
                            else:
                                for policy_label, policy_data_series in final_policies_for_plot.items():
                                    if param in policy_data_series and pd.notna(policy_data_series[param]):
                                        normalized_params_for_radar.loc[policy_label, param] = (policy_data_series[
                                                                                                    param] - min_val) / range_val
                                    else:
                                        normalized_params_for_radar.loc[policy_label, param] = np.nan

                        normalized_params_for_radar = normalized_params_for_radar.fillna(0.5).astype(float)

                        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
                        angles += angles[:1]

                        fig_radar, ax_radar = plt.subplots(figsize=(15, 15),
                                                           subplot_kw=dict(polar=True))  # Increased size
                        ax_radar.set_xticks(angles[:-1])
                        # Adjust label properties for readability
                        xticklabels = ax_radar.set_xticklabels(readable_radar_params_with_ranges,
                                                               fontsize=25)  # Further reduce if needed
                        for label in xticklabels:  # Rotate labels slightly if they overlap, or adjust position
                            label.set_horizontalalignment('center')

                        ax_radar.set_yticks(np.arange(0, 1.1, 0.2))
                        ax_radar.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0, 1.1, 0.2)], fontsize=25)
                        ax_radar.set_ylim(0, 1)

                        color_map_radar = plt.cm.get_cmap('tab10', len(final_policies_for_plot))

                        for i, (label, policy_row) in enumerate(normalized_params_for_radar.iterrows()):
                            data = policy_row[all_radar_params_to_plot].values.flatten().tolist()
                            data += data[:1]
                            ax_radar.plot(angles, data, linewidth=2, linestyle='solid', label=label,
                                          color=color_map_radar(i))
                            ax_radar.fill(angles, data, color=color_map_radar(i), alpha=0.25)

                        plt.title('Selected Policies: All Parameters (Normalized Values, Original Ranges in Labels)',
                                  size=16, y=1.08)  # Adjust y
                        ax_radar.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15 - (0.03 * (num_vars / 12))),
                                        ncol=min(2, len(final_policies_for_plot)),
                                        fontsize=25)  # Adjusted ncol and bbox

                        plot_filename_radar_all = os.path.join(output_plot_dir,
                                                               "analyzed_policy_radar_plot_all_params_with_ranges.svg")
                        try:
                            fig_radar.savefig(plot_filename_radar_all, bbox_inches='tight')
                            print(f"Saved: {plot_filename_radar_all}")
                        except Exception as e:
                            print(f"Error saving single radar plot: {e}")
                        plt.close(fig_radar)

        print("\n--- Revised Analysis and Plotting Complete ---")
        print(f"Plots saved in: {output_plot_dir}")


if __name__ == "__main__":
    cfg = Config()
    farms = "real_farms"  # syn_farms, real_farms

    syn_farm_dir = os.path.join(cfg.disk_dir, "crop_inventory", "syn_farms")
    if farms == "syn_farms":
        base_farm_dir = os.path.join(syn_farm_dir, "mc")
        num_configs = np.arange(1, 501)
    else:
        base_farm_dir = os.path.join(cfg.disk_dir, "crop_inventory", "farms_config_s")
        num_configs = np.arange(1, 571)
    fontsize = 25

    # plot_all()

    plot_policy()






