import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import random  # Added import
from eco_intensification import Config, main_run_pyomo
from scipy.stats import pearsonr
from config import Config
from eco_intensification import precompute_inputs, build_and_solve_pyomo_model, assign_pyomo_solution_to_gdf
from pathlib import Path  # Add pathlib
import matplotlib
import matplotlib.patches as mpatches  # Needed for manual legend
import matplotlib.cm as cm


def calculate_proximity_to_habitat(farm_gdf):
    ag_plots = farm_gdf[farm_gdf['type'] == 'ag_plot'].copy()
    hab_plots = farm_gdf[farm_gdf['type'] == 'hab_plots']

    if hab_plots.empty or ag_plots.empty:
        return pd.Series(np.inf, index=ag_plots.index)

    # Combine all habitat geometries into a single MultiPolygon or GeometryCollection
    # Using unary_union is robust but can be slow for many complex shapes
    try:
        hab_union = hab_plots.geometry.unary_union
    except Exception as e:
        print(f"Warning: Could not create unary union of habitat plots: {e}. Using individual geometries.")
        # Fallback: iterate through individual habitat plots (slower)
        hab_union = None

    distances = []
    for idx, ag_row in ag_plots.iterrows():
        ag_centroid = ag_row.geometry.centroid
        min_dist = np.inf
        if hab_union:
            try:
                # Calculate distance from ag centroid to the unified habitat boundary
                min_dist = ag_centroid.distance(hab_union)
            # Alternative: distance to the nearest point on the habitat boundary
            # nearest_geom = nearest_points(ag_centroid, hab_union)[1]
            # min_dist = ag_centroid.distance(nearest_geom)
            except Exception as e:
                print(f"Warning: Distance calculation failed for plot {ag_row.get('id', idx)}: {e}")
                min_dist = np.inf  # Assign inf if calculation fails
        else:  # Fallback if unary_union failed
            for _, hab_row in hab_plots.iterrows():
                dist = ag_centroid.distance(hab_row.geometry)
                if dist < min_dist:
                    min_dist = dist
        distances.append(min_dist)

    return pd.Series(distances, index=ag_plots.index)


def run_analysis(cfg, num_configs, base_farm_dir, neighbor_dist, exit_tol, penalty_coef):
    """
    Runs the optimization for specified farms, aggregates results,
    and performs analysis on factors influencing interventions.
    """
    all_farm_analysis_data = []
    correlations = False
    scatter = False
    crop_type = True
    run_opt = False
    load_df = True

    if mode == "syn_farms":
        analysis_plot_output_dir = os.path.join(syn_farm_dir, "plots", "ei", "corr")
    else:
        analysis_plot_output_dir = os.path.join(syn_farm_dir, "plots", "ei", "corr_real")
    os.makedirs(analysis_plot_output_dir, exist_ok=True)

    if not load_df:
        for config_id in num_configs:
            print(f"\n===== Analyzing Configuration: {config_id} =====")
            config_path = os.path.join(base_farm_dir, f"config_{config_id}")

            num_farms = sum(1 for item in os.listdir(config_path)
                            if os.path.isdir(os.path.join(config_path, item)) and item.startswith("farm_"))
            print(f"Found {num_farms} farm directories in {config_path}")
            for farm_id in range(1, num_farms + 1):
                if mode == "syn_farms":
                    farm_path = os.path.join(config_path, f"farm_{farm_id}")
                else:
                    farm_path = os.path.join(config_path, f"farm_{farm_id}", "ei")

                    geojson_path = os.path.join(farm_path, "input.geojson")
                if not os.path.exists(farm_path) or not os.path.exists(geojson_path):
                    print(f"Required paths/files not found for farm {farm_id}, skipping.")
                    continue

                # Run the optimization process to get the results GDF
                # This uses the modified main_run_pyomo which returns the GDF
                image_path = os.path.join(farm_path, "output_gt.svg")
                output_json = os.path.join(farm_path, "output_gt.geojson")
                output_json_full = os.path.join(farm_path,
                                                "output_gt_full.geojson")
                if run_opt:
                    try:
                        farm_gdf_results = main_run_pyomo(
                            cfg, geojson_path, image_path, output_json,
                            neighbor_dist, exit_tol, penalty_coef
                        )
                        farm_gdf_results.to_file(output_json_full, driver="GeoJSON")
                    except Exception as e:
                        continue
                else:
                    farm_gdf_results = gpd.read_file(output_json_full)

                farm_analysis = farm_gdf_results[farm_gdf_results['type'] == 'ag_plot'].copy()
                farm_analysis['proximity_hab'] = calculate_proximity_to_habitat(farm_gdf_results)

                # Extract other relevant factors
                farm_analysis['area_ha'] = farm_analysis["area"] / 10000
                farm_analysis['perimeter_km'] = farm_analysis["perimeter"] / 1000
                farm_analysis['farm_id'] = farm_id
                farm_analysis['config_id'] = config_id

                # Select relevant columns for analysis
                analysis_cols = [
                    'config_id', 'farm_id', 'id', 'yield', 'area_ha', 'perimeter_km', 'num_sides', 'num_neighbours',
                    'proximity_hab', 'margin_intervention', 'habitat_conversion', 'label'
                ]
                # Ensure all columns exist before selection
                cols_to_select = [col for col in analysis_cols if col in farm_analysis.columns]
                farm_analysis_data = farm_analysis[cols_to_select]

                all_farm_analysis_data.append(farm_analysis_data)

        agg_df = pd.concat(all_farm_analysis_data, ignore_index=True)
        agg_df.to_csv(os.path.join(analysis_plot_output_dir, "corr.csv"))
    else:
        agg_df = pd.read_csv(os.path.join(analysis_plot_output_dir, "corr.csv"))

    print(f"\n--- Aggregated Analysis Data (first 5 rows) ---")
    print(agg_df.head())
    print(f"Total agricultural plots analyzed: {len(agg_df)}")

    # Replace inf proximity with NaN for correlation/plotting
    agg_df['proximity_hab'].replace([np.inf, -np.inf], np.nan, inplace=True)

    sns.set_theme(style="whitegrid")

    # Define consistent font sizes
    AXIS_LABEL_FONTSIZE = 25
    TICK_LABEL_FONTSIZE = 25
    LEGEND_FONTSIZE = 25
    ANNOTATION_FONTSIZE = 25
    BAR_LABEL_FONTSIZE = 25

    # Define consistent colors
    MARGIN_COLOR = '#e41a1c'  # Red from Set1
    HABITAT_COLOR = '#377eb8'  # Blue from Set1

    factors_to_plot = {
        'yield': 'Yield',
        'area_ha': 'Plot Area (hectares)',
        'perimeter_km': 'Plot Perimeter (kms)',
        'num_sides': 'Number of Sides',
        'num_neighbours': 'Number of Neighbours',
        'proximity_hab': 'Proximity to Habitat'
    }

    if scatter:
        # Scatter Plots for Key Relationships (Margin Intervention)
        num_factors = len(factors_to_plot)
        fig_margin, axes_margin = plt.subplots(3, 2, figsize=(16, 18))
        axes_margin_flat = axes_margin.flatten()

        for i, (factor_col, factor_label) in enumerate(factors_to_plot.items()):
            ax = axes_margin_flat[i]
            if factor_col in agg_df.columns and 'margin_intervention' in agg_df.columns:
                plot_data = agg_df.dropna(subset=[factor_col, 'margin_intervention'])
                if not plot_data.empty and len(plot_data[factor_col].unique()) > 1:  # Need variance for correlation
                    # Scatter plot with regression line
                    sns.regplot(data=plot_data, x=factor_col, y='margin_intervention', ax=ax,
                                scatter_kws={'alpha': 0.3, 's': 25, 'color': MARGIN_COLOR},  # Slightly larger points
                                line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 1.5})  # Clearer line

                    # Calculate R-squared
                    r, _ = pearsonr(plot_data[factor_col], plot_data['margin_intervention'])
                    r_squared = r ** 2
                    # Annotate R-squared value on the plot
                    ax.annotate(f'$R^2 = {r_squared:.2f}$', xy=(0.05, 0.95), xycoords='axes fraction',
                                fontsize=ANNOTATION_FONTSIZE, ha='left', va='top',
                                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

                    ax.set_xlabel(factor_label, fontsize=AXIS_LABEL_FONTSIZE)
                    ax.set_ylabel('Margin Intervention Level', fontsize=AXIS_LABEL_FONTSIZE)
                    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust rect to prevent suptitle overlap
        margin_scatter_path = os.path.join(analysis_plot_output_dir, "scatter_margin.svg")
        plt.savefig(margin_scatter_path, dpi=300, bbox_inches='tight')
        plt.close(fig_margin)
        print(f"Saved enhanced margin intervention scatter plots to: {margin_scatter_path}")

        # 2. Scatter Plots: Habitat Conversion vs. Key Factors
        print("\n--- Generating Enhanced Scatter Plots (Habitat) ---")
        fig_habitat, axes_habitat = plt.subplots(3, 2, figsize=(16, 18))
        axes_habitat_flat = axes_habitat.flatten()

        for i, (factor_col, factor_label) in enumerate(factors_to_plot.items()):
            ax = axes_habitat_flat[i]
            if factor_col in agg_df.columns and 'habitat_conversion' in agg_df.columns:
                plot_data = agg_df.dropna(subset=[factor_col, 'habitat_conversion'])
                if not plot_data.empty and len(plot_data[factor_col].unique()) > 1:
                    sns.regplot(data=plot_data, x=factor_col, y='habitat_conversion', ax=ax,
                                scatter_kws={'alpha': 0.3, 's': 25, 'color': HABITAT_COLOR},
                                line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 1.5})

                    # Calculate R-squared
                    r, _ = pearsonr(plot_data[factor_col], plot_data['habitat_conversion'])
                    r_squared = r ** 2
                    ax.annotate(f'$R^2 = {r_squared:.2f}$', xy=(0.05, 0.95), xycoords='axes fraction',
                                fontsize=ANNOTATION_FONTSIZE, ha='left', va='top',
                                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

                    ax.set_xlabel(factor_label, fontsize=AXIS_LABEL_FONTSIZE)
                    ax.set_ylabel('Habitat Conversion Level', fontsize=AXIS_LABEL_FONTSIZE)
                    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust rect
        habitat_scatter_path = os.path.join(analysis_plot_output_dir, "scatter_habitat.svg")
        plt.savefig(habitat_scatter_path, dpi=300, bbox_inches='tight')
        plt.close(fig_habitat)
        print(f"Saved enhanced habitat conversion scatter plots to: {habitat_scatter_path}")

    if correlations:
        # 3. Visualize Correlations Nicely (Bar Chart)
        print("\n--- Generating Enhanced Correlation Bar Chart ---")
        valid_factors = [f for f in factors_to_plot if f in agg_df.columns]

        corr_data_list = []
        # Calculate Margin correlations
        if 'margin_intervention' in agg_df.columns and valid_factors:
            corr_margin = agg_df[valid_factors + ['margin_intervention']].corr()['margin_intervention'].drop(
                'margin_intervention').dropna()
            if not corr_margin.empty:
                corr_margin_df = corr_margin.reset_index()
                corr_margin_df.columns = ['Factor', 'Correlation']
                # Map factor technical names to readable labels for the plot
                corr_margin_df['Factor_Label'] = corr_margin_df['Factor'].map(factors_to_plot)
                corr_margin_df['Intervention'] = 'Margin'
                corr_data_list.append(corr_margin_df)

        # Calculate Habitat correlations
        if 'habitat_conversion' in agg_df.columns and valid_factors:
            corr_habitat = agg_df[valid_factors + ['habitat_conversion']].corr()['habitat_conversion'].drop(
                'habitat_conversion').dropna()
            if not corr_habitat.empty:
                corr_habitat_df = corr_habitat.reset_index()
                corr_habitat_df.columns = ['Factor', 'Correlation']
                corr_habitat_df['Factor_Label'] = corr_habitat_df['Factor'].map(factors_to_plot)
                corr_habitat_df['Intervention'] = 'Habitat'
                corr_data_list.append(corr_habitat_df)

        # Create the plot if data exists
        if corr_data_list:
            combined_corr_df = pd.concat(corr_data_list, ignore_index=True)
            # Use the mapped Factor_Label for plotting if available, otherwise fallback to Factor
            x_axis_col = 'Factor_Label' if 'Factor_Label' in combined_corr_df.columns else 'Factor'

            plt.figure(figsize=(12, 8))  # Adjusted size
            barplot = sns.barplot(
                data=combined_corr_df,
                x=x_axis_col,
                y='Correlation',
                hue='Intervention',
                palette={'Margin': MARGIN_COLOR, 'Habitat': HABITAT_COLOR},
                edgecolor='black',  # Add subtle edge color
                linewidth=0.75
            )

            barplot.yaxis.grid(False)

            # Add value labels on bars with adjusted size
            for container in barplot.containers:
                barplot.bar_label(container, fmt='%.2f', fontsize=BAR_LABEL_FONTSIZE, padding=4)  # Increased padding

            plt.axhline(0, color='grey', linewidth=1.0, linestyle='--')  # Slightly thicker line
            plt.ylabel("Correlation Coefficient (Pearson's r)", fontsize=AXIS_LABEL_FONTSIZE)
            plt.xlabel("")
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right', fontsize=TICK_LABEL_FONTSIZE)  # Keep font size consistent
            plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

            # Adjust y-limits slightly, ensure they cover a reasonable range like [-0.5, 0.5] or based on data
            min_corr = combined_corr_df['Correlation'].min()
            max_corr = combined_corr_df['Correlation'].max()
            y_lim_lower = min(min_corr - 0.1, -0.5)  # Ensure at least -0.5
            y_lim_upper = max(max_corr + 0.1, 0.5)  # Ensure at least 0.5
            plt.ylim(y_lim_lower, y_lim_upper)

            # Enhance legend
            legend = plt.legend(title='Intervention Type', loc='best', fontsize=LEGEND_FONTSIZE,
                                title_fontsize=LEGEND_FONTSIZE)
            # legend.get_frame().set_edgecolor('black') # Optional: frame around legend

            plt.tight_layout()
            correlation_plot_path = os.path.join(analysis_plot_output_dir,
                                                 "correlation_factors.svg")
            plt.savefig(correlation_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved enhanced correlation bar chart to: {correlation_plot_path}")
        else:
            print("\nSkipping correlation visualization as no valid correlation data was generated.")

    if crop_type:
        for col in ['margin_intervention', 'habitat_conversion']:
            agg_df[col] = pd.to_numeric(agg_df[col], errors='coerce')

        df_cleaned = agg_df.dropna(subset=['margin_intervention', 'habitat_conversion'])
        summary_stats = df_cleaned.groupby('label')[['margin_intervention', 'habitat_conversion']].agg(
            ['mean', 'sum', 'count']
        )

        mean_margin = summary_stats[('margin_intervention', 'mean')]
        mean_habitat = summary_stats[('habitat_conversion', 'mean')]
        labels = mean_margin.index.tolist()  # Crop labels
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        # --- Combined Plot for Mean Margin Intervention and Habitat Conversion ---
        fig, ax1 = plt.subplots(figsize=(14, 8))  # Adjusted figsize

        # Plot Margin bars on primary axis (ax1)
        rects1 = ax1.bar(x - width / 2, mean_margin, width, label='Mean Margin Intervention', color=MARGIN_COLOR)
        ax1.set_xlabel('Crop Type', fontsize=AXIS_LABEL_FONTSIZE)
        ax1.set_ylabel('Mean Margin Intervention', fontsize=AXIS_LABEL_FONTSIZE, color=MARGIN_COLOR)
        ax1.tick_params(axis='y', labelcolor=MARGIN_COLOR, labelsize=TICK_LABEL_FONTSIZE)
        ax1.tick_params(axis='x', rotation=45, labelsize=TICK_LABEL_FONTSIZE)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, ha='right')  # Set x-ticks labels with alignment
        ax1.grid(False)  # Turn off grid for ax1

        # Create secondary axis (ax2) sharing the x-axis
        ax2 = ax1.twinx()
        # Plot Habitat bars on secondary axis (ax2)
        rects2 = ax2.bar(x + width / 2, mean_habitat, width, label='Mean Habitat Conversion', color=HABITAT_COLOR)
        ax2.set_ylabel('Mean Habitat Conversion', fontsize=AXIS_LABEL_FONTSIZE, color=HABITAT_COLOR)
        ax2.tick_params(axis='y', labelcolor=HABITAT_COLOR, labelsize=TICK_LABEL_FONTSIZE)
        ax2.grid(False)  # Turn off grid for ax2

        fig.tight_layout()  # Adjust layout
        plt.savefig(os.path.join(analysis_plot_output_dir, "crop_type_mean_combined.svg"))
        plt.close(fig)


def run_single_optimization(cfg, geojson_path, nd, exit_tol, pc):
    farm_gdf = gpd.read_file(geojson_path)
    params = cfg.params

    farm_data = precompute_inputs(farm_gdf, params, neighbor_dist=nd)
    model = build_and_solve_pyomo_model(farm_data, params, penalty_coef=pc, exit_tol=exit_tol)
    farm_gdf = assign_pyomo_solution_to_gdf(model, farm_gdf.copy())
    farm_gdf['margin_intervention'] = np.where(
        (farm_gdf['type'] == 'ag_plot') & (farm_gdf['margin_intervention'] >= 0.01),
        farm_gdf['margin_intervention'],
        0.0
    )
    farm_gdf['habitat_conversion'] = np.where(
        (farm_gdf['type'] == 'ag_plot') & (farm_gdf['habitat_conversion'] >= 0.01),
        farm_gdf['habitat_conversion'],
        0.0
    )
    return farm_gdf, farm_data


def neib_penalty_sensitivity():
    neighbor_dist_values = [100, 200, 500, 1000, 1500, 2000, 2500]
    penalty_coef_values = [0, 1e1, 1e2, 1e3, 1e4, 1e5]
    if mode == "syn_farms":
        analysis_plot_output_dir = os.path.join(syn_farm_dir, "plots", "ei", "nbdist_penality")
    else:
        analysis_plot_output_dir = os.path.join(syn_farm_dir, "plots", "ei", "nbdist_penality_real")
    os.makedirs(analysis_plot_output_dir, exist_ok=True)
    load = False

    def visualize_sensitivity(results_df):
        if results_df is None or results_df.empty:
            print("No results to visualize.")
            return

        # Pivot data for heatmaps
        pivots = {
            'sum_margins': results_df.pivot(index='penalty_coef', columns='neighbor_dist', values='sum_margins'),
            'sum_habitats': results_df.pivot(index='penalty_coef', columns='neighbor_dist', values='sum_habitats'),
            'length_margins': results_df.pivot(index='penalty_coef', columns='neighbor_dist', values='length_margins'),
            'area_habitats': results_df.pivot(index='penalty_coef', columns='neighbor_dist', values='area_habitats'),
            'mean_margins': results_df.pivot(index='penalty_coef', columns='neighbor_dist', values='mean_margins'),
            'mean_habitats': results_df.pivot(index='penalty_coef', columns='neighbor_dist', values='mean_habitats'),
        }

        titles = {
            'sum_margins': 'Farmwise Sum Margin Fraction',
            'sum_habitats': 'Farmwise Sum Habitat Fraction',
            'length_margins': 'Farmwise Sum Margin Length (km)',
            'area_habitats': 'Farmwise Sum Habitat Area (ha)',
            'mean_margins': 'Farmwise Mean Margin Fraction',
            'mean_habitats': 'Farmwise Mean Habitat Fraction',
        }

        # --- Create Figure and Axes ---
        fig, axes = plt.subplots(3, 2, figsize=(20, 24))  # Adjusted figsize for 3x2 grid
        axes_flat = axes.flatten()  # Flatten axes array for easy iteration

        AXIS_LABEL_FONTSIZE = 16  # Adjusted font size for better fit
        TICK_LABEL_FONTSIZE = 14
        CBAR_LABEL_FONTSIZE = 16
        ANNOT_SIZE = 10

        # --- Iterate through pivots and create heatmaps ---
        for i, (key, pivot_table) in enumerate(pivots.items()):
            ax = axes_flat[i]
            cbar_label = titles.get(key, key.replace('_', ' ').title())  # Use predefined title or generate one

            hm = sns.heatmap(
                pivot_table,
                annot=False,  # Keep annotations off for clarity with many cells
                fmt=".2f",
                cmap="viridis",
                ax=ax,
                annot_kws={"size": ANNOT_SIZE},
                cbar_kws={'label': cbar_label}
            )
            ax.set_title(cbar_label, fontsize=AXIS_LABEL_FONTSIZE + 2)  # Add title to each subplot
            ax.set_xlabel('Neighbor Distance Threshold (m)', fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel('Penalty Coefficient', fontsize=AXIS_LABEL_FONTSIZE)

            # Set tick label sizes and rotation
            ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE, rotation=45)  # Rotate x-ticks, removed ha='right'
            ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)

            # Format y-axis labels for penalty coefficient (scientific notation)
            ax.set_yticklabels([f"{x:.1e}" for x in pivot_table.index], size=TICK_LABEL_FONTSIZE)

            # Set color bar label size explicitly
            cbar = hm.collections[0].colorbar
            cbar.ax.set_ylabel(cbar_label, fontsize=CBAR_LABEL_FONTSIZE)
            cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE - 2)  # Adjust cbar tick label size if needed

        # --- Adjust Layout and Save ---
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Adjust rect for main title if needed
        save_path = os.path.join(analysis_plot_output_dir, "nbdist_penalty_sensitivity_heatmap.svg")
        plt.savefig(save_path, dpi=300)
        print(f"Saved sensitivity heatmap to: {save_path}")
        plt.close(fig)  # Close the figure

    def run_sensitivity():
        results = []
        print("\nStarting Sensitivity Analysis...")
        for nd in neighbor_dist_values:
            for pc in penalty_coef_values:
                print(f"ND: {nd}, PC: {pc}")

                length_margins = []
                area_habitats = []
                sum_margins = []
                mean_margins = []
                sum_habitats = []
                mean_habitats = []

                for config_id in num_configs:
                    print(f"Running Configuration: {config_id}")

                    config_path = os.path.join(base_farm_dir, f"config_{config_id}")
                    num_farms = sum(1 for item in os.listdir(config_path)
                                    if os.path.isdir(os.path.join(config_path, item)) and item.startswith("farm_"))
                    for farm_id in range(1, num_farms + 1):
                        if mode == "syn_farms":
                            farm_path = os.path.join(config_path, f"farm_{farm_id}")
                        else:
                            farm_path = os.path.join(config_path, f"farm_{farm_id}", "ei")
                        geojson_path = os.path.join(farm_path, "input.geojson")

                        # Run the optimization for this parameter set
                        try:
                            result_gdf, _ = run_single_optimization(cfg, geojson_path, nd, exit_tol, pc)
                        except Exception as e:
                            sum_margin_raw, sum_habitat_raw = 0.0, 0.0
                            peri_weighted_margin, area_weighted_habitat = 0.0, 0.0
                            result_gdf = None

                        if result_gdf is not None:
                            # Calculate total intervention fractions for agricultural plots
                            ag_plots_results = result_gdf[result_gdf['type'] == 'ag_plot']

                            if not ag_plots_results.empty:
                                # Sum of raw intervention values (0 to 1 per plot)
                                sum_margin_raw = ag_plots_results['margin_intervention'].sum()
                                sum_habitat_raw = ag_plots_results['habitat_conversion'].sum()

                                mean_margin_raw = ag_plots_results['margin_intervention'].mean()
                                mean_habitat_raw = ag_plots_results['habitat_conversion'].mean()

                                # Area-weighted sums
                                peri_weighted_margin = (ag_plots_results['margin_intervention'] * ag_plots_results[
                                    "perimeter"] / 1000).sum()
                                area_weighted_habitat = (ag_plots_results['habitat_conversion'] * ag_plots_results[
                                    "area"] / 10000).sum()

                            else:
                                # Handle cases with no agricultural plots
                                sum_margin_raw, sum_habitat_raw = 0.0, 0.0
                                peri_weighted_margin, area_weighted_habitat = 0.0, 0.0
                                mean_margin_raw, mean_habitat_raw = 0.0, 0.0

                            length_margins.append(peri_weighted_margin)
                            area_habitats.append(area_weighted_habitat)
                            sum_margins.append(sum_margin_raw)
                            sum_habitats.append(sum_habitat_raw)
                            mean_margins.append(mean_margin_raw)
                            mean_habitats.append(mean_habitat_raw)
                        else:
                            length_margins.append(np.nan)
                            area_habitats.append(np.nan)
                            sum_margins.append(np.nan)
                            sum_habitats.append(np.nan)
                            mean_margins.append(np.nan)
                            mean_habitats.append(np.nan)

                results.append({
                    'neighbor_dist': nd,
                    'penalty_coef': pc,
                    'length_margins': np.nanmean(length_margins),
                    'area_habitats': np.nanmean(area_habitats),
                    'sum_margins': np.nanmean(sum_margins),
                    'sum_habitats': np.nanmean(sum_habitats),
                    'mean_margins': np.nanmean(mean_margins),
                    'mean_habitats': np.nanmean(mean_habitats),
                })

        print("\nSensitivity Analysis Complete.")
        return pd.DataFrame(results)

    if not load:
        sensitivity_results_df = run_sensitivity()
        sensitivity_results_df.to_csv(os.path.join(analysis_plot_output_dir, "nd_pc.csv"))
    else:
        sensitivity_results_df = pd.read_csv(os.path.join(analysis_plot_output_dir, "nd_pc.csv"))
    visualize_sensitivity(sensitivity_results_df)


def run_alpha_delta_analysis():
    parameters_to_test = {
        'margin': ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta'],
        'habitat': ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
    }

    def calculate_summary_metrics(result_gdf):
        """Calculates summary metrics for a single farm optimization result."""
        if result_gdf is None or result_gdf.empty:
            return {
                'total_margin_fraction': 0.0, 'total_habitat_fraction': 0.0,
                'mean_margin_fraction': 0.0, 'mean_habitat_fraction': 0.0,
            }

        ag_plots = result_gdf[result_gdf['type'] == 'ag_plot']
        if ag_plots.empty:
            return {
                'total_margin_fraction': 0.0, 'total_habitat_fraction': 0.0,
                'mean_margin_fraction': 0.0, 'mean_habitat_fraction': 0.0,
            }

        # Sum of raw intervention values (0 to 1 per plot)
        total_margin_fraction = ag_plots['margin_intervention'].sum()
        total_habitat_fraction = ag_plots['habitat_conversion'].sum()

        # Mean intervention fractions
        mean_margin_fraction = ag_plots['margin_intervention'].mean()
        mean_habitat_fraction = ag_plots['habitat_conversion'].mean()

        return {
            'total_margin_fraction': total_margin_fraction,
            'total_habitat_fraction': total_habitat_fraction,
            'mean_margin_fraction': mean_margin_fraction,
            'mean_habitat_fraction': mean_habitat_fraction,
        }

    def format_factor_name(factor_str):
        if not isinstance(factor_str, str) or not factor_str.startswith('factor_'):
            return factor_str  # Return original if format is unexpected

        parts = factor_str.replace('factor_', '').split('_')
        # Capitalize first letter of each part (Title Case style)
        formatted_parts = [part.capitalize() for part in parts]
        return ' '.join(formatted_parts)

    def plot_multi_factor_sensitivity_results(results_df, output_dir="sensitivity_multi_factor"):
        """Plots sensitivity results from multi-factor sampling (per crop-parameter) using correlation analysis."""
        if results_df is None or results_df.empty:
            print("No multi-factor results to plot.")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Dynamically identify factor columns (starting with 'factor_')
        factor_cols = [col for col in results_df.columns if col.startswith('factor_')]

        metric_cols = [
            'mean_margin_fraction', 'mean_habitat_fraction',
            'total_margin_fraction', 'total_habitat_fraction'
        ]
        # Filter out metric columns that might not exist or are all NaN
        metric_cols = [m for m in metric_cols if m in results_df.columns and results_df[m].notna().any()]

        if not factor_cols:
            print("Error: No factor columns found in results DataFrame.")
            return
        if not metric_cols:
            print("Error: No valid metric columns found in results DataFrame.")
            return

        # Separate baseline (factor=1) and sampled runs
        sampled_runs = results_df[results_df['parameter_name'] != 'baseline'].copy()
        sampled_runs.dropna(subset=factor_cols + metric_cols,
                            inplace=True)  # Drop rows where any factor or metric is NaN

        if sampled_runs.empty:
            print("No valid sampled runs found after dropping NaNs.")
            return

        print(f"Calculating correlations for {len(sampled_runs)} valid simulation runs.")

        # Calculate Pearson correlations
        correlations = sampled_runs[factor_cols + metric_cols].corr(method='pearson')

        # Extract correlations between factors and metrics
        factor_metric_corr = correlations.loc[factor_cols, metric_cols]

        # --- Bar Plot of Correlations (Optional, alternative view) ---
        corr_long = factor_metric_corr.reset_index().melt(id_vars='index', var_name='Metric', value_name='Correlation')
        corr_long.rename(columns={'index': 'Factor'}, inplace=True)
        corr_long['Formatted Factor'] = corr_long['Factor'].apply(format_factor_name)

        print("\nGenerating individual correlation bar plots per metric...")
        for metric in metric_cols:  # Iterate through valid metrics
            metric_data = corr_long[corr_long['Metric'] == metric].sort_values('Correlation', ascending=False)

            fig_height = max(4, len(metric_data) * 0.15)
            fig, ax = plt.subplots(figsize=(10, fig_height))

            barplot = sns.barplot(
                data=metric_data,
                y='Formatted Factor',
                x='Correlation',
                ax=ax,  # Use the axes for this specific figure
                palette='viridis',
                orient='h'
            )

            ax.bar_label(barplot.containers[0], fmt='%.2f', padding=3, fontsize=9)
            ax.set_xlabel("Pearson Correlation", fontsize=12)
            ax.set_ylabel("Factor", fontsize=12)
            ax.tick_params(axis='y', labelsize=10)
            ax.tick_params(axis='x', labelsize=10)
            min_corr = metric_data['Correlation'].min()
            max_corr = metric_data['Correlation'].max()
            ax.set_xlim(left=min(-1.0, min_corr - 0.1),
                        right=max(1.0, max_corr + 0.1))
            ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)

            safe_metric_name = metric.replace(' ', '_')
            filename = f"correlation_bar_{safe_metric_name}.svg"
            filepath = os.path.join(output_dir, filename)

            # Apply tight layout before saving
            plt.tight_layout()
            plt.savefig(filepath, dpi=300)
            print(f"  Saved correlation bar plot to: {filepath}")

            plt.close(fig)

    # --- Main Analysis Loop ---
    results_list = []
    cfg = Config()
    base_params = cfg.params  # Get the baseline parameters
    factor_range = (0.5, 2)  # Range for uniform sampling
    num_simulations_per_farm = 5  # Number of runs where all crop-params are varied together

    # Get list of all crops from base_params
    all_crops = list(base_params.get('crops', {}).keys())
    if not all_crops:
        print("Error: No crops found in base parameters. Exiting.")
        return

    print("\nStarting Multi-Factor Sensitivity Analysis (Per Crop-Parameter)...")
    # Iterate through configurations and farms
    for config_id in num_configs:
        print(f"\n===== Processing Configuration: {config_id} =====")
        config_path = os.path.join(base_farm_dir, f"config_{config_id}")
        if not os.path.isdir(config_path):
            print(f"Skipping config {config_id}: Directory not found.")
            continue

        num_farms = sum(1 for item in os.listdir(config_path)
                        if os.path.isdir(os.path.join(config_path, item)) and item.startswith("farm_"))
        print(f"Found {num_farms} farms.")

        for farm_id in range(1, num_farms + 1):
            if mode == "syn_farms":
                farm_path = os.path.join(config_path, f"farm_{farm_id}")
            else:
                farm_path = os.path.join(config_path, f"farm_{farm_id}", "ei")
            geojson_path = os.path.join(farm_path, "input.geojson")

            if not os.path.exists(geojson_path):
                print(f"Skipping farm {config_id}-{farm_id}: GeoJSON not found at {geojson_path}")
                continue

            print(f"--- Processing Farm: {config_id}-{farm_id} ---")

            # --- Baseline Run --- 
            try:
                print("  Running Baseline...")
                temp_cfg_baseline = Config()
                temp_cfg_baseline.params = copy.deepcopy(base_params)
                baseline_gdf, _ = run_single_optimization(temp_cfg_baseline, geojson_path, neighbor_dist, exit_tol,
                                                          penalty_coef)
                baseline_metrics = calculate_summary_metrics(baseline_gdf)
                baseline_metrics.update({
                    'config_id': config_id,
                    'farm_id': farm_id,
                    'parameter_category': 'baseline',
                    'parameter_name': 'baseline',
                    'factor': 1.0
                })
                # Add placeholder factor columns for baseline
                for crop in all_crops:
                    for category, names in parameters_to_test.items():
                        for name in names:
                            factor_key = f"factor_{crop}_{category}_{name}"
                            baseline_metrics[factor_key] = 1.0
                results_list.append(baseline_metrics)
            except Exception as e:
                print(f"  ERROR running baseline for farm {config_id}-{farm_id}: {e}")
                continue  # Skip farm if baseline fails

            # --- Multi-Factor Parameter Variation Runs (Per Crop-Parameter) ---
            print(f"  Running {num_simulations_per_farm} multi-factor (crop-parameter) simulations...")
            for i in range(num_simulations_per_farm):

                # 1. Create a fresh copy of params for this run
                current_run_params = copy.deepcopy(base_params)
                sampled_factors_for_run = {}  # Store factors used in this run
                modification_failed = False

                # 2. Iterate through ALL CROPS and ALL PARAMETERS, sample unique factor for EACH combination
                for crop in all_crops:
                    if modification_failed: break  # Stop modifying if an error occurred

                    for param_category, param_names in parameters_to_test.items():
                        if modification_failed: break  # Stop modifying if an error occurred

                        for param_name in param_names:
                            # Define the unique key for this factor
                            factor_key = f"factor_{crop}_{param_category}_{param_name}"

                            # Sample a factor for this specific crop-parameter combination
                            factor = np.random.uniform(factor_range[0], factor_range[1])
                            sampled_factors_for_run[factor_key] = factor

                            # Modify the parameter directly in the copied dictionary
                            try:
                                # Check structure exists before trying to access/modify
                                if crop in current_run_params.get('crops', {}) and \
                                        param_category in current_run_params['crops'][crop] and \
                                        param_name in current_run_params['crops'][crop][param_category]:

                                    original_value = current_run_params['crops'][crop][param_category][param_name]
                                    current_run_params['crops'][crop][param_category][
                                        param_name] = original_value * factor
                                else:
                                    # Parameter doesn't exist for this crop, store NaN factor and report warning
                                    print(
                                        f"    Warning: Parameter '{param_category}.{param_name}' not found for crop '{crop}'. Storing NaN factor.")
                                    sampled_factors_for_run[factor_key] = np.nan
                                    # Decide if this is a fatal error for the run
                                    # modification_failed = True # Option: Treat missing param as failure

                            except Exception as e:
                                print(
                                    f"    ERROR modifying param {crop}-{param_category}-{param_name} for run {i + 1}: {e}")
                                sampled_factors_for_run[factor_key] = np.nan
                                modification_failed = True  # Mark run as failed
                                break  # Stop modifying parameters for this run

                # Check if any parameter modification failed during the loops
                if modification_failed:
                    print(f"    Skipping simulation {i + 1} due to parameter modification error.")
                    metrics = {m: np.nan for m in calculate_summary_metrics(None)}  # Store NaN metrics
                else:
                    # 3. Use a temporary config object with ALL parameters modified per crop
                    temp_cfg_multifactor = Config()
                    temp_cfg_multifactor.params = current_run_params

                    # 4. Run the simulation
                    try:
                        # print(f"    Running multi-factor simulation {i+1}...")
                        result_gdf, _ = run_single_optimization(temp_cfg_multifactor, geojson_path, neighbor_dist,
                                                                exit_tol, penalty_coef)
                        metrics = calculate_summary_metrics(result_gdf)
                    except Exception as e:
                        print(f"    ERROR running multi-factor simulation {i + 1}: {e}")
                        metrics = {m: np.nan for m in calculate_summary_metrics(None)}

                # 5. Store results (metrics + all crop-specific factors used)
                run_result = {
                    'config_id': config_id,
                    'farm_id': farm_id,
                    'parameter_category': 'multi_crop',  # Indicate detailed multi-factor run
                    'parameter_name': 'multi_crop',
                    'factor': np.nan
                }
                run_result.update(metrics)
                run_result.update(sampled_factors_for_run)
                results_list.append(run_result)

            # --- End of multi-factor simulations for this farm ---

        print(f"--- Finished Configuration {config_id} ---")

    print("\nMulti-Factor Sensitivity Analysis (Per Crop-Parameter) Complete.")

    results_df = pd.DataFrame(results_list)

    if mode == "syn_farms":
        output_dir = os.path.join(syn_farm_dir, "plots", "ei", "alpha_delta")
    else:
        output_dir = os.path.join(syn_farm_dir, "plots", "ei", "alpha_delta_real")
    os.makedirs(output_dir, exist_ok=True)
    results_csv_path = os.path.join(output_dir, "sensitivity_results_multi_factor_crop.csv")  # New name
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved multi-factor (crop-specific) results to: {results_csv_path}")

    # Plot the results using the modified multi-factor plotting function
    plot_multi_factor_sensitivity_results(results_df, output_dir=output_dir)


def run_economic_sensitivity_analysis(
        config_dir_pattern="config_{config_id}",  # Pattern for config subdirs
        farm_subdir_pattern="farm_{farm_id}",  # Pattern for farm subdirs
        input_geojson_name="input.geojson"):  # Subdir for results

    def calculate_summary_metrics(result_gdf):
        """Calculates summary metrics for a single farm optimization result."""
        # Now takes NPVs as arguments since they come from run_single_optimization return
        if result_gdf is None:
            # Ensure 'areas' column is present for calculations
            return {
                'mean_margin_fraction': 0.0, 'mean_habitat_fraction': 0.0,
            }

        ag_plots = result_gdf[result_gdf['type'] == 'ag_plot']
        if ag_plots.empty:
            return {
                'mean_margin_fraction': 0.0, 'mean_habitat_fraction': 0.0,
            }

        # Check if intervention columns exist, default to 0 if not
        margin_col = 'margin_intervention' if 'margin_intervention' in ag_plots.columns else None
        habitat_col = 'habitat_conversion' if 'habitat_conversion' in ag_plots.columns else None

        mean_margin_fraction = ag_plots[margin_col].mean() if margin_col else 0.0
        mean_habitat_fraction = ag_plots[habitat_col].mean() if habitat_col else 0.0

        return {
            'mean_margin_fraction': mean_margin_fraction,
            'mean_habitat_fraction': mean_habitat_fraction,
        }

    # format_factor_name remains the same
    def format_factor_name(factor_str):
        """Formats factor_<type>_<name>... into readable labels."""
        if not isinstance(factor_str, str) or not factor_str.startswith('factor_'):
            return factor_str
        parts = factor_str.replace('factor_', '').split('_')
        if parts[0] == 'price':
            formatted_parts = [parts[0].capitalize()] + parts[1:]
        elif parts[0] == 'cost':
            formatted_parts = [parts[0].capitalize()] + [p.capitalize() for p in parts[1:]]
        else:
            formatted_parts = [part.capitalize() for part in parts]
        return ' '.join(formatted_parts)

    # plot_multi_factor_sensitivity_results remains the same
    def plot_multi_factor_sensitivity_results(results_df, output_dir="sensitivity_economic"):
        """Plots sensitivity results from multi-factor sampling using correlation analysis."""
        if results_df is None or results_df.empty:
            print("No multi-factor results to plot.")
            return

        os.makedirs(output_dir, exist_ok=True)
        matplotlib.use('Agg')  # Use Agg backend for saving plots without display

        factor_cols = [col for col in results_df.columns if col.startswith('factor_')]
        metric_cols = [
            'mean_margin_fraction', 'mean_habitat_fraction',
            'total_margin_area', 'total_habitat_area',
            'total_npv_objective', 'total_npv_sum',
        ]
        metric_cols = [m for m in metric_cols if
                       m in results_df.columns and pd.to_numeric(results_df[m], errors='coerce').notna().any()]

        if not factor_cols: print("Error: No factor columns found."); return
        if not metric_cols: print("Error: No valid metric columns found."); return

        sampled_runs = results_df[results_df['parameter_name'] != 'baseline'].copy()
        for col in factor_cols + metric_cols:
            sampled_runs[col] = pd.to_numeric(sampled_runs[col], errors='coerce')
        sampled_runs.dropna(subset=factor_cols + metric_cols, inplace=True)

        if sampled_runs.empty: print("No valid sampled runs found after dropping NaNs."); return

        print(f"Calculating correlations for {len(sampled_runs)} valid simulation runs.")
        correlations = sampled_runs[factor_cols + metric_cols].corr(method='pearson')
        factor_metric_corr = correlations.loc[factor_cols, metric_cols]

        corr_long = factor_metric_corr.reset_index().melt(id_vars='index', var_name='Metric', value_name='Correlation')
        corr_long.rename(columns={'index': 'Factor'}, inplace=True)
        corr_long['Formatted Factor'] = corr_long['Factor'].apply(format_factor_name)

        print("\nGenerating individual correlation bar plots per metric...")
        for metric in metric_cols:
            metric_data = corr_long[corr_long['Metric'] == metric].sort_values('Correlation', ascending=False)
            if metric_data['Correlation'].isnull().all():
                print(f"  Skipping plot for metric '{metric}': All correlations are NaN.")
                continue

            num_factors = len(metric_data)
            fig_height = max(4, 3 + num_factors * 0.15)
            fig, ax = plt.subplots(figsize=(15, fig_height+5))
            barplot = sns.barplot(data=metric_data, y='Formatted Factor', x='Correlation', ax=ax, palette='viridis',
                                  orient='h')

            try:  # Robust bar labeling
                for container in barplot.containers:
                    labels = [f'{x:.2f}' if pd.notna(x) else 'NaN' for x in container.datavalues]
                    ax.bar_label(container, labels=labels, fmt='%.2f', padding=3, fontsize=25)
            except Exception as e:
                print(f"  Warning: Could not add bar labels for {metric}: {e}")

            ax.set_xlabel("Pearson Correlation", fontsize=25);
            ax.set_ylabel("Economic Factor", fontsize=25)
            ax.tick_params(axis='both', labelsize=25)
            valid_corrs = metric_data['Correlation'].dropna()
            if not valid_corrs.empty:
                min_corr, max_corr = valid_corrs.min(), valid_corrs.max()
                ax.set_xlim(left=min(-1.05, min_corr - 0.1), right=max(1.05, max_corr + 0.1))
            else:
                ax.set_xlim(-1.05, 1.05)
            ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
            safe_metric_name = metric.replace(' ', '_').replace('/', '_')
            filename = f"correlation_bar_{safe_metric_name}.svg"
            filepath = os.path.join(output_dir, filename)
            # plt.title(f"Sensitivity of '{metric}' to Economic Factors", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.savefig(filepath, dpi=300);
            print(f"  Saved correlation bar plot to: {filepath}");
            plt.close(fig)

    def plot_ratio_sensitivity(df, base_params, output_plot_dir):
        numeric_cols = ['mean_margin_fraction', 'mean_habitat_fraction']
        factor_cols = [col for col in df.columns if col.startswith('factor_')]
        for col in numeric_cols + factor_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['mean_margin_fraction', 'mean_habitat_fraction'], inplace=True)

        df_baseline = df[df['parameter_name'] == 'baseline'].copy()
        df_simulations = df[df['parameter_name'] != 'baseline'].copy()

        abs_param_mapping = {}
        baseline_values = {}  # Store baseline values for reference

        # 1. Crop Prices
        all_crops = list(base_params.get('crops', {}).keys())
        print(f"Processing crops: {all_crops}")
        for crop in all_crops:
            factor_col = f"factor_price_{crop}"
            abs_col = f"abs_price_{crop}"
            path = ['crops', crop, 'p_c']
            if factor_col in df_simulations.columns:  # Check if factor exists in CSV
                try:
                    # Get baseline value using path
                    base_val = base_params
                    for key in path:
                        base_val = base_val[key]
                    abs_param_mapping[abs_col] = {'factor_col': factor_col, 'path': path}
                    baseline_values[abs_col] = base_val
                    print(f"  Mapping {factor_col} -> {abs_col} (Baseline: {base_val})")
                except KeyError:
                    print(f"  Warning: Baseline parameter path {path} not found for factor {factor_col}. Skipping.")
                except Exception as e:
                    print(f"  Warning: Error getting baseline for {factor_col}: {e}. Skipping.")
            else:
                print(f"  Warning: Factor column {factor_col} not found in CSV. Skipping.")

        cost_paths_dict = {
            "abs_cost_margin_implementation": {'factor_col': "factor_cost_margin_implementation",
                                               'path': ['costs', 'margin', 'implementation']},
            "abs_cost_margin_maintenance": {'factor_col': "factor_cost_margin_maintenance",
                                            'path': ['costs', 'margin', 'maintenance']},
            "abs_cost_habitat_implementation": {'factor_col': "factor_cost_habitat_implementation",
                                                'path': ['costs', 'habitat', 'implementation']},
            "abs_cost_habitat_maintenance": {'factor_col': "factor_cost_habitat_maintenance",
                                             'path': ['costs', 'habitat', 'maintenance']},
            "abs_cost_habitat_existing_hab": {'factor_col': "factor_cost_habitat_existing_hab",
                                              'path': ['costs', 'habitat', 'existing_hab']},
            "abs_cost_agriculture_maintenance": {'factor_col': "factor_cost_agriculture_maintenance",
                                                 'path': ['costs', 'agriculture', 'maintenance']},
        }
        print(f"Processing costs...")
        for abs_col, info in cost_paths_dict.items():
            factor_col = info['factor_col']
            path = info['path']
            if factor_col in df_simulations.columns:
                try:
                    base_val = base_params
                    for key in path:
                        base_val = base_val[key]
                    abs_param_mapping[abs_col] = info
                    baseline_values[abs_col] = base_val
                    print(f"  Mapping {factor_col} -> {abs_col} (Baseline: {base_val})")
                except KeyError:
                    print(f"  Warning: Baseline parameter path {path} not found for factor {factor_col}. Skipping.")
                except Exception as e:
                    print(f"  Warning: Error getting baseline for {factor_col}: {e}. Skipping.")
            else:
                print(f"  Warning: Factor column {factor_col} not found in CSV. Skipping.")

        # --- Calculate Absolute Values in DataFrame ---
        print("\nCalculating absolute parameter values for each simulation run...")
        for abs_col, info in abs_param_mapping.items():
            factor_col = info['factor_col']
            base_val = baseline_values[abs_col]
            # Vectorized calculation: base_value * factor_value_for_run
            df_simulations[abs_col] = base_val * df_simulations[factor_col]
        print("Absolute value calculation complete.")

        def classify_intervention(row, threshold):
            margin = row['mean_margin_fraction']
            habitat = row['mean_habitat_fraction']
            margin_active = margin >= threshold
            habitat_active = habitat >= threshold

            if margin_active and habitat_active:
                return 'Mixed'
            elif margin_active:
                return 'Margin'
            elif habitat_active:
                return 'Habitat'
            else:
                return 'None'

        df_simulations['intervention_state'] = df_simulations.apply(
            classify_intervention, axis=1, threshold=0.05
        )

        abs_price_cols = [col for col in abs_param_mapping.keys() if col.startswith('abs_price_')]
        abs_cost_cols = [col for col in abs_param_mapping.keys() if col.startswith('abs_cost_')]

        ratio_cols = []
        key_costs_for_price_ratio = [
            'abs_cost_margin_implementation', 'abs_cost_margin_maintenance',
            'abs_cost_habitat_implementation', 'abs_cost_habitat_maintenance',
            'abs_cost_agriculture_maintenance'
        ]
        for price_col in abs_price_cols:
            for cost_col in key_costs_for_price_ratio:
                if cost_col in df_simulations.columns:  # Ensure cost column exists
                    ratio_col_name = f"ratio_{price_col.replace('abs_', '')}_div_{cost_col.replace('abs_', '')}"
                    # Calculate ratio, adding epsilon to denominator
                    with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings locally
                        df_simulations[ratio_col_name] = df_simulations[price_col] / (
                                df_simulations[cost_col] + 1e-9)
                    # Replace potential inf/-inf resulting from near-zero division with NaN
                    df_simulations[ratio_col_name].replace([np.inf, -np.inf], np.nan, inplace=True)
                    ratio_cols.append(ratio_col_name)
                    print(f"  Calculated: {ratio_col_name}")
                else:
                    print(f"  Skipping ratio involving missing cost column: {cost_col}")

        # 2. Cost / Cost Ratios (Example pairs)
        cost_ratio_pairs = [
            ('abs_cost_margin_implementation', 'abs_cost_habitat_implementation'),
            ('abs_cost_margin_maintenance', 'abs_cost_habitat_maintenance'),
            ('abs_cost_agriculture_maintenance', 'abs_cost_margin_maintenance'),
            ('abs_cost_agriculture_maintenance', 'abs_cost_habitat_maintenance'),
            ('abs_cost_margin_implementation', 'abs_cost_margin_maintenance'),  # Impl vs Maint
            ('abs_cost_habitat_implementation', 'abs_cost_habitat_maintenance'),  # Impl vs Maint
        ]
        for cost1_col, cost2_col in cost_ratio_pairs:
            if cost1_col in df_simulations.columns and cost2_col in df_simulations.columns:
                ratio_col_name = f"ratio_{cost1_col.replace('abs_', '')}_div_{cost2_col.replace('abs_', '')}"
                with np.errstate(divide='ignore', invalid='ignore'):
                    df_simulations[ratio_col_name] = df_simulations[cost1_col] / (df_simulations[cost2_col] + 1e-9)
                df_simulations[ratio_col_name].replace([np.inf, -np.inf], np.nan, inplace=True)
                ratio_cols.append(ratio_col_name)
                print(f"  Calculated: {ratio_col_name}")
            else:
                print(f"  Skipping cost ratio involving missing columns: {cost1_col} or {cost2_col}")

        print(f"Calculated {len(ratio_cols)} ratio columns.")

        state_colors = {
            'None': '#fee090', 'Margin': '#e41a1c', 'Habitat': '#4daf4a', 'Mixed': '#984ea3'
        }

        def create_boxplot(data, x_col, y_col, title, xlabel, ylabel, order, palette, output_dir, filename,
                           baseline_val=None):
            print(f"  Generating box plot: {filename}")
            # Filter out NaN values in the y_col for this specific plot
            plot_data = data.dropna(subset=[y_col])
            if plot_data.empty:
                print(f"    Skipping plot {filename}: No valid data after dropping NaNs for {y_col}.")
                return

            plt.figure(figsize=(8, 8))
            try:
                sns.boxplot(
                    data=plot_data,
                    x=x_col,
                    y=y_col,
                    order=order,
                    palette=palette
                )
                # plt.title(title, fontsize=14)
                plt.xlabel(xlabel, fontsize=25)
                plt.ylabel(ylabel, fontsize=25)
                plt.xticks(fontsize=25)
                plt.yticks(fontsize=25)

                # Add baseline value line if provided
                if baseline_val is not None and pd.notna(baseline_val):
                    plt.axhline(baseline_val, color='black', linestyle='--', linewidth=1.2, alpha=0.9,
                                label=f'Baseline ({baseline_val:.2f})')
                    plt.legend()

                plt.grid(False)
                plt.tight_layout()
                plot_path = os.path.join(output_dir, filename)
                plt.savefig(plot_path, dpi=300)  # PNG format suitable for potentially many plots
                plt.close()
            except Exception as e:
                print(f"    ERROR generating box plot {filename}: {e}")
                plt.close()  # Ensure plot is closed even on error

        # 1. Box Plots for Individual Absolute Values
        print("\nGenerating box plots for absolute parameter values...")
        abs_value_cols_to_plot = abs_price_cols + abs_cost_cols
        for abs_col in abs_value_cols_to_plot:
            col_name_formatted = abs_col.replace('abs_', '').replace('_', ' ').title()
            baseline_val = baseline_values.get(abs_col)  # Get baseline value if exists
            fname = f"boxplot_abs_{abs_col.replace('abs_', '')}.svg"
            create_boxplot(df_simulations, 'intervention_state', abs_col,
                           f'Distribution of {col_name_formatted} by State', 'Intervention State',
                           f'{col_name_formatted} (Absolute Value)',
                           ['None', 'Margin', 'Habitat', 'Mixed'], state_colors,
                           output_plot_dir, fname, baseline_val)

        # 2. Box Plots for Ratios
        print("\nGenerating box plots for calculated ratios...")
        for ratio_col in ratio_cols:
            # Format names for plot titles/labels
            try:
                num_str, den_str = ratio_col.replace('ratio_', '').split('_div_')
                num_name = num_str.replace('_', ' ').title()
                den_name = den_str.replace('_', ' ').title()
                plot_title = f'Distribution of {num_name} / \n {den_name}\nRatio by State'
                plot_ylabel = f'{num_name} / \n {den_name} Ratio'
                fname = f"boxplot_{ratio_col}.svg"

                create_boxplot(df_simulations, 'intervention_state', ratio_col,
                               plot_title, 'Intervention State', plot_ylabel,
                               ['None', 'Margin', 'Habitat', 'Mixed'], state_colors,
                               output_plot_dir, fname)  # No baseline line for ratios generally
            except ValueError:
                print(f"  Warning: Could not parse ratio column name '{ratio_col}' for plotting. Skipping.")
            except Exception as e:
                print(f"  Error processing boxplot for ratio {ratio_col}: {e}")

    load = True
    factor_range = (0.5, 2.0)
    num_simulations_per_farm = 5
    results_list = []
    base_cfg_obj = Config()  # Load base config once to get baseline parameters structure
    base_params = copy.deepcopy(base_cfg_obj.params)  # Deep copy to avoid modifying original

    # --- Define the Economic Parameters to Test (Same as before) ---
    economic_parameters = {}
    all_crops = list(base_params.get('crops', {}).keys())
    for crop in all_crops:
        key = f"price_{crop}"
        economic_parameters[key] = ['crops', crop, 'p_c']
    cost_paths = {
        "cost_margin_implementation": ['costs', 'margin', 'implementation'],
        "cost_margin_maintenance": ['costs', 'margin', 'maintenance'],
        "cost_habitat_implementation": ['costs', 'habitat', 'implementation'],
        "cost_habitat_maintenance": ['costs', 'habitat', 'maintenance'],
        "cost_habitat_existing_hab": ['costs', 'habitat', 'existing_hab'],
        "cost_agriculture_maintenance": ['costs', 'agriculture', 'maintenance'],
    }
    economic_parameters.update(cost_paths)
    factor_keys = [f"factor_{key}" for key in economic_parameters.keys()]
    # --- End Parameter Definition ---

    print("\nStarting Multi-Factor Economic Sensitivity Analysis...")

    if not load:
        # --- Loop through Configurations ---
        for config_id in num_configs:
            print(f"\n===== Processing Configuration: {config_id} =====")
            config_subdir = config_dir_pattern.format(config_id=config_id)
            config_path = os.path.join(base_farm_dir, config_subdir)

            if not os.path.isdir(config_path):
                print(f"Skipping config {config_id}: Directory not found at {config_path}")
                continue

            # --- Detect number of farms in this configuration ---
            farm_dirs = [item for item in os.listdir(config_path)
                         if os.path.isdir(os.path.join(config_path, item)) and item.startswith("farm_")]
            num_farms_in_config = len(farm_dirs)
            farm_ids_in_config = sorted([int(d.split('_')[-1]) for d in farm_dirs])
            print(f"Found {num_farms_in_config} farms in config {config_id}. IDs: {farm_ids_in_config}")
            # ---

            # --- Loop through Farms in Configuration ---
            for farm_id in farm_ids_in_config:
                print(f"--- Processing Farm: Config {config_id} - Farm {farm_id} ---")
                farm_subdir = farm_subdir_pattern.format(farm_id=farm_id)

                if mode == "syn_farms":
                    farm_path = os.path.join(config_path, farm_subdir)
                else:
                    farm_path = os.path.join(config_path, farm_subdir, "ei")
                geojson_path = os.path.join(farm_path, input_geojson_name)

                if not os.path.exists(geojson_path):
                    print(f"Skipping farm {config_id}-{farm_id}: GeoJSON not found at {geojson_path}")
                    continue

                # --- Baseline Run for this Config/Farm ---
                print("    Running Baseline...")
                # Create a fresh Config object for baseline to ensure clean params
                baseline_cfg = Config()
                baseline_cfg.params = copy.deepcopy(base_params)  # Use the standard base params

                # Call the optimization function
                baseline_gdf, _ = run_single_optimization(baseline_cfg, geojson_path, neighbor_dist, exit_tol,
                                                          penalty_coef)

                baseline_metrics = calculate_summary_metrics(baseline_gdf)

                # Store baseline result
                baseline_result = {
                    'config_id': config_id, 'farm_id': farm_id,
                    'parameter_name': 'baseline', 'factor': 1.0
                }
                baseline_result.update(baseline_metrics)
                for f_key in factor_keys: baseline_result[f_key] = 1.0  # Add factor placeholders
                results_list.append(baseline_result)

                # --- Multi-Factor Parameter Variation Runs for this Config/Farm ---
                print(f"    Running {num_simulations_per_farm} multi-factor economic simulations...")
                for i in range(num_simulations_per_farm):
                    print(f"      Simulation {i + 1}/{num_simulations_per_farm}")
                    current_run_params = copy.deepcopy(base_params)
                    sampled_factors_for_run = {}
                    modification_failed = False

                    # Modify parameters (same logic as before)
                    for param_key, path_list in economic_parameters.items():
                        factor_key = f"factor_{param_key}"
                        factor = np.random.uniform(factor_range[0], factor_range[1])
                        sampled_factors_for_run[factor_key] = factor
                        try:
                            target_dict = current_run_params
                            for key in path_list[:-1]: target_dict = target_dict[key]
                            param_to_modify = path_list[-1]
                            if param_to_modify in target_dict:
                                target_dict[param_to_modify] *= factor
                            else:
                                print(f"      Warning: Param path {path_list} not found. Storing NaN.")
                                sampled_factors_for_run[factor_key] = np.nan
                        except Exception as e:
                            print(f"      ERROR modifying {param_key}: {e}. Storing NaN.")
                            sampled_factors_for_run[factor_key] = np.nan
                            modification_failed = True;
                            break

                    # Create a temporary Config object for this run
                    current_cfg = Config()  # Assumes Config() constructor is simple
                    current_cfg.params = current_run_params

                    # Run simulation if parameter modification succeeded
                    if modification_failed:
                        print(f"      Skipping simulation {i + 1} due to param modification error.")
                        metrics = {m: np.nan for m in calculate_summary_metrics(None)}
                        sim_npv_obj, sim_npv_sum = np.nan, np.nan
                        sim_status, sim_time = 'modification_error', 0
                    else:
                        # Call the optimization function with the modified cfg
                        sim_gdf, _ = run_single_optimization(current_cfg, geojson_path, neighbor_dist, exit_tol,
                                                             penalty_coef)
                        metrics = calculate_summary_metrics(sim_gdf)  # Handles None GDF

                    # Store results
                    run_result = {
                        'config_id': config_id, 'farm_id': farm_id,
                        'parameter_name': 'multi_economic', 'factor': np.nan
                    }
                    run_result.update(metrics)
                    run_result.update(sampled_factors_for_run)
                    results_list.append(run_result)
                    # --- End of simulation loop ---
                print(f"--- Finished Farm {config_id}-{farm_id} ---")
                # --- End of farm loop ---
            print(f"===== Finished Configuration {config_id} =====")
            # --- End of config loop ---

        results_df = pd.DataFrame(results_list)

        # Define output directory path (now relative to base_farm_dir)
        if mode == "syn_farms":
            output_subdir = "sensitivity_economic"
        else:
            output_subdir = "sensitivity_economic_real"
        output_path = os.path.join(syn_farm_dir, "plots", "ei", output_subdir)
        os.makedirs(output_path, exist_ok=True)

        results_csv_path = os.path.join(output_path,
                                        "sensitivity_results_economic_multi_config.csv")  # Updated filename
        results_df.to_csv(results_csv_path, index=False)
    else:
        if mode == "syn_farms":
            output_subdir = "sensitivity_economic"
        else:
            output_subdir = "sensitivity_economic_real"
        output_path = os.path.join(syn_farm_dir, "plots", "ei", output_subdir)
        results_df = pd.read_csv(os.path.join(output_path, "sensitivity_results_economic_multi_config.csv"))

    plot_multi_factor_sensitivity_results(results_df, output_dir=output_path)
    plot_ratio_sensitivity(results_df, base_params, output_plot_dir=output_path)


def analyze_temporal_npv():
    def run(plot_id_to_analyze, farm_data, farm_gdf):
        plot_idx = farm_data['id_to_idx'][plot_id_to_analyze]
        plot_info_rows = farm_gdf[farm_gdf['id'] == plot_id_to_analyze]
        plot_info = plot_info_rows.iloc[0]

        if plot_info.get('type', '') != 'ag_plot': return None
        margin_i = plot_info.get('margin_intervention', 0.0)
        habitat_i = plot_info.get('habitat_conversion', 0.0)
        if margin_i == 0.0 and habitat_i == 0.0: return None

        params = farm_data['params']
        id_to_idx = farm_data['id_to_idx']
        idx_to_id = farm_data['idx_to_id']
        neighbors = farm_data['neighbors']
        plot_types = farm_data['plot_types']
        time_factor_cache = farm_data['time_factor_cache']
        discount_factors_array = farm_data['discount_factors']
        T_len = len(discount_factors_array)

        # Costs...
        cost_margin_impl = params['costs']['margin']['implementation']
        cost_margin_maint = params['costs']['margin']['maintenance']
        cost_habitat_impl = params['costs']['habitat']['implementation']
        cost_habitat_maint = params['costs']['habitat']['maintenance']
        cost_ag_maint = params['costs']['agriculture']['maintenance']

        # Plot specific params...
        c_label = plot_info.get('label', None)
        if not c_label or c_label not in params['crops']:
            # print(f"Warning: Crop label '{c_label}' invalid for plot {plot_id_to_analyze}. Skipping.")
            return None
        crop_def = params['crops'][c_label]
        base_yield = plot_info.get('yield', 0.0)

        A = plot_info.geometry.area/10000

        p_c = crop_def['p_c']
        # Margin parameters...
        alpha = crop_def['margin']['alpha']
        beta = crop_def['margin']['beta']
        gamma = crop_def['margin']['gamma']
        delta_ = crop_def['margin']['delta']
        epsilon_ = crop_def['margin']['epsilon']
        zeta_ = crop_def['margin']['zeta']
        # Habitat parameters...
        hab_alpha = crop_def['habitat']['alpha']
        hab_beta = crop_def['habitat']['beta']
        hab_gamma = crop_def['habitat']['gamma']
        hab_delta = crop_def['habitat']['delta']
        hab_epsilon = crop_def['habitat']['epsilon']
        hab_zeta = crop_def['habitat']['zeta']
        # Time-factor arrays...
        margin_time_factors_gamma = time_factor_cache.get(gamma, np.zeros(T_len))
        margin_time_factors_zeta = time_factor_cache.get(zeta_, np.zeros(T_len))
        hab_time_factors_gamma = time_factor_cache.get(hab_gamma, np.zeros(T_len))
        hab_time_factors_zeta = time_factor_cache.get(hab_zeta, np.zeros(T_len))

        impl_cost = A * (margin_i * cost_margin_impl + habitat_i * cost_habitat_impl)
        margin_maint_cost = margin_i * cost_margin_maint * A
        habitat_maint_cost = habitat_i * cost_habitat_maint * A
        ag_maint_cost = (1 - habitat_i) * cost_ag_maint * A
        total_maint = margin_maint_cost + habitat_maint_cost + ag_maint_cost
        yield_loss_by_habitat = base_yield * p_c * A * habitat_i

        # Calculate annual cash flows...
        annual_discounted_cf = np.zeros(T_len)
        # Get neighbor intervention values from the *passed GDF*
        neighbor_margins = {}
        neighbor_habitats = {}
        i_neighbors = neighbors.get(plot_id_to_analyze, [])
        gdf_id_set = set(farm_gdf['id'])  #

        for j_idx in i_neighbors:
            j_id = idx_to_id[j_idx]
            if j_id in gdf_id_set:
                # Use .loc and iloc[0] assuming IDs are unique within the GDF for this farm run
                neighbor_info = farm_gdf.loc[farm_gdf['id'] == j_id].iloc[0]
                neighbor_margins[j_id] = neighbor_info.get('margin_intervention', 0.0)
                neighbor_habitats[j_id] = neighbor_info.get('habitat_conversion', 0.0)
            else:  # Neighbor exists in precompute data but somehow missing from GDF? Assign 0.
                neighbor_margins[j_id] = 0.0
                neighbor_habitats[j_id] = 0.0

        for t_idx in range(T_len):
            # Pollination_t calculation... (using margin_i, habitat_i, neighbor_margins, neighbor_habitats)
            pollination_t = 0.0
            pollination_t += alpha * margin_i * np.exp(-beta * 0.0) * margin_time_factors_gamma[t_idx]  # Own margin

            for j_idx in i_neighbors:
                j_id = idx_to_id[j_idx]
                dist_ij = farm_data['distances'][plot_idx, j_idx]  # Use precomputed distance array
                j_plot_type = plot_types[j_idx]

                if j_plot_type == 'ag_plot':
                    pollination_t += alpha * neighbor_margins.get(j_id, 0.0) * np.exp(-beta * dist_ij) * \
                                     margin_time_factors_gamma[t_idx]
                    pollination_t += neighbor_habitats.get(j_id, 0.0) * hab_alpha * np.exp(-hab_beta * dist_ij) * \
                                     hab_time_factors_gamma[t_idx]
                elif j_plot_type == 'hab_plots':
                    pollination_t += hab_alpha * np.exp(-hab_beta * dist_ij) * hab_time_factors_gamma[
                        t_idx]  # Existing habitat

            # Pest_t calculation... (using margin_i, habitat_i, neighbor_margins, neighbor_habitats)
            pest_t = 0.0
            pest_t += delta_ * margin_i * np.exp(-epsilon_ * 0.0) * margin_time_factors_zeta[t_idx]  # Own margin

            for j_idx in i_neighbors:
                j_id = idx_to_id[j_idx]
                dist_ij = farm_data['distances'][plot_idx, j_idx]
                j_plot_type = plot_types[j_idx]

                if j_plot_type == 'ag_plot':
                    pest_t += delta_ * neighbor_margins.get(j_id, 0.0) * np.exp(-epsilon_ * dist_ij) * \
                              margin_time_factors_zeta[t_idx]
                    pest_t += neighbor_habitats.get(j_id, 0.0) * hab_delta * np.exp(-hab_epsilon * dist_ij) * \
                              hab_time_factors_zeta[t_idx]
                elif j_plot_type == 'hab_plots':
                    pest_t += hab_delta * np.exp(-hab_epsilon * dist_ij) * hab_time_factors_zeta[
                        t_idx]  # Existing habitat

            # Combined Yield, Revenue, CF, Discounting...
            combined_yield_t = base_yield * (1.0 + pollination_t + pest_t)
            revenue_t = combined_yield_t * p_c * A * (1 - habitat_i)
            yearly_cf = revenue_t - total_maint - yield_loss_by_habitat
            discounted_cf_t = yearly_cf * discount_factors_array[t_idx]
            annual_discounted_cf[t_idx] = discounted_cf_t

        cumulative_npv = np.cumsum(annual_discounted_cf) - impl_cost
        cumulative_npv_plot = np.insert(cumulative_npv, 0, -impl_cost)  # For year 0

        breakeven_year = -1
        positive_npv_indices = np.where(cumulative_npv_plot >= 0)[0]
        if len(positive_npv_indices) > 0:
            breakeven_year = positive_npv_indices[0]  # Year = index in this array

        total_npv_calc = cumulative_npv_plot[-1]

        # Return results dictionary (NO PLOTTING HERE)
        analysis_results = {
            "plot_id": plot_id_to_analyze,
            "annual_discounted_cf": annual_discounted_cf.tolist(),  # Convert to list for easier aggregation
            "cumulative_npv": cumulative_npv_plot.tolist(),  # Includes year 0
            "breakeven_year": breakeven_year,
            "total_npv_calculated": total_npv_calc
        }
        return analysis_results

    def plot_average_temporal_npv(avg_annual_cf, avg_cumulative_npv, T_horizon, output_dir):
        if avg_annual_cf is None or avg_cumulative_npv is None:
            print("  No average data to plot.")
            return

        years = np.arange(1, T_horizon + 1)
        plot_years = np.arange(0, T_horizon + 1)  # For cumulative plot (includes year 0)

        # Ensure average arrays have correct lengths
        if len(avg_annual_cf) != T_horizon or len(avg_cumulative_npv) != T_horizon + 1:
            print(f"Error: Average array lengths mismatch T_horizon ({T_horizon}). Cannot plot.")
            # print(f"Lengths - Annual: {len(avg_annual_cf)}, Cumulative: {len(avg_cumulative_npv)}")
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        #fig.suptitle(f'Average Temporal NPV Analysis for Config: {config_id}', fontsize=14)

        # Plot 1: Average Annual Discounted Cash Flow
        axes[0].bar(years, avg_annual_cf, color='skyblue', edgecolor='black')
        axes[0].set_ylabel('Avg. Annual Discounted CF ($)')
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        axes[0].set_title('Average Contribution of Each Year to NPV')

        # Plot 2: Average Cumulative Discounted NPV
        avg_breakeven_year = -1
        positive_avg_indices = np.where(avg_cumulative_npv >= 0)[0]
        if len(positive_avg_indices) > 0:
            avg_breakeven_year = plot_years[positive_avg_indices[0]]

        axes[1].plot(plot_years, avg_cumulative_npv, marker='o', linestyle='-', color='purple', label='Avg. Cumulative NPV')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Avg. Cumulative Discounted NPV ($)')
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        axes[1].legend(loc='best')  # Use 'best' location for average plot
        axes[1].set_title('Average Cumulative NPV Over Time')
        axes[1].set_xticks(np.arange(0, T_horizon + 1, max(1, T_horizon // 10)))  # Auto ticks

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the plot
        plot_filename = "average_temporal_npv.svg"
        analysis_plot_save_path = os.path.join(output_dir, plot_filename)
        os.makedirs(output_dir, exist_ok=True)  # Ensure dir exists
        plt.savefig(analysis_plot_save_path, dpi=200)
        plt.close(fig)

    T_horizon = cfg.params.get('t', 20)

    # Aggregate lists for this configuration
    config_annual_cfs = []
    config_cumulative_npvs = []
    config_breakeven_years = []
    for config_id in num_configs:
        print(f"Running config: {config_id}")

        config_path = os.path.join(base_farm_dir, f"config_{config_id}")

        processed_farms_count = 0

        farm_folders = [f for f in os.listdir(config_path)
                        if os.path.isdir(os.path.join(config_path, f)) and f.startswith("farm_")]

        farm_ids = sorted([int(f.split('_')[1]) for f in farm_folders])
        for farm_id in farm_ids:
            farm_id_str = f"farm_{farm_id}" if isinstance(farm_id, int) else farm_id
            if mode == "syn_farms":
                farm_path = os.path.join(config_path, farm_id_str)
            else:
                farm_path = os.path.join(config_path, farm_id_str, "ei")
            geojson_path = os.path.join(farm_path, "input.geojson")
            farm_gdf, farm_data = run_single_optimization(
                cfg, geojson_path, neighbor_dist, exit_tol, penalty_coef
            )
            processed_farms_count += 1

            for plot_id in farm_gdf['id']:
                # Check type and intervention level directly from GDF row
                row = farm_gdf[farm_gdf['id'] == plot_id].iloc[0]
                if row.get('type', '') == 'ag_plot' and \
                        (row.get('margin_intervention', 0.0) > 0.0 or row.get('habitat_conversion', 0.0) > 0.0):

                    analysis_result = run(plot_id, farm_data, farm_gdf)

                    if analysis_result:
                        # Ensure arrays have the expected length before appending
                        if len(analysis_result['annual_discounted_cf']) == T_horizon and \
                                len(analysis_result['cumulative_npv']) == T_horizon + 1:
                            config_annual_cfs.append(analysis_result['annual_discounted_cf'])
                            config_cumulative_npvs.append(analysis_result['cumulative_npv'])
                            config_breakeven_years.append(analysis_result['breakeven_year'])



    # Convert lists of lists/arrays to 2D NumPy arrays for averaging
    annual_cfs_array = np.array(config_annual_cfs)
    cumulative_npvs_array = np.array(config_cumulative_npvs)

    # Calculate mean across plots/farms (axis=0)
    avg_annual_cf = np.mean(annual_cfs_array, axis=0)
    avg_cumulative_npv = np.mean(cumulative_npvs_array, axis=0)

    # Optionally calculate median breakeven year (handling -1 for never)
    valid_breakevens = [yr for yr in config_breakeven_years if yr != -1]
    median_breakeven = np.median(valid_breakevens) if valid_breakevens else -1
    print(
        f"  Median Breakeven Year (for plots that break even): {median_breakeven if median_breakeven != -1 else 'N/A'}")

    output_base = os.path.join(syn_farm_dir, "plots", "ei")
    if mode == "syn_farms":
        output_analysis_dir = os.path.join(output_base, "temporal_analysis")
    else:
        output_analysis_dir = os.path.join(output_base, "temporal_analysis_real")

    os.makedirs(output_analysis_dir, exist_ok=True)
    np.save(os.path.join(output_analysis_dir, "avg_annual_cf.npy"), avg_annual_cf)
    np.save(os.path.join(output_analysis_dir, "avg_cumulative_npv.npy"), avg_cumulative_npv)

    # Plot the averages
    plot_average_temporal_npv(
        avg_annual_cf,
        avg_cumulative_npv,
        T_horizon,
        output_analysis_dir  # Save plot in config-specific analysis dir
    )


def yield_factor_plot():
    def model_function(t_arr, dist, fraction, alpha, beta, gamma):
        """Calculates the function value."""
        # Ensure t_arr is treated as an array for element-wise operations, even if scalar t_val is passed
        t_term = (1 - np.exp(-gamma * np.array(t_arr)))
        return fraction * alpha * np.exp(-beta * dist) * t_term

    # Default parameters
    default_params = {
        'fraction': 0.5,
        'alpha': 0.10,
        'beta': 0.01,
        'gamma': 0.2,
        't': 20
    }

    # Ranges for variables
    t_arr = np.arange(1, default_params['t'] + 1)
    dist_arr = np.linspace(0, 200, 100)  # Distance range from 0 to 200

    # Values to test for each parameter ablation
    param_variations = {
        'alpha': [0.05, 0.10, 0.20],
        'fraction': [0.25, 0.5, 0.75],
        'beta': [0.005, 0.01, 0.02],
        'gamma': [0.1, 0.2, 0.4]
    }

    # Fixed values used in simplified plots
    fixed_dist_for_time_plot = 50
    fixed_time_for_dist_plot = 25

    # --- Create Combined Panel Plot ---

    plt.style.use('seaborn-v0_8-whitegrid')  # Starting with this style, but removing grid below

    # Create a figure with a 4x2 grid of subplots
    fig, axes = plt.subplots(len(param_variations), 2, figsize=(16, 22), sharey='row')  # Share Y axis, Adjusted figsize
    #fig.suptitle('Function Ablation Analysis: Output vs. Time and Distance', fontsize=18, y=0.99)

    # Iterate through each parameter to vary
    param_names = list(param_variations.keys())
    for row_idx, param_name in enumerate(param_names):
        param_values = param_variations[param_name]

        # Build string of default parameters *not* being varied in this row
        other_defaults_list = []
        for dp_name, dp_value in default_params.items():
            if dp_name != param_name and dp_name != 't':  # Exclude the varied param and 't'
                other_defaults_list.append(f'{dp_name}={dp_value}')
        other_defaults_str = ", ".join(other_defaults_list)

        # --- Plot 1: Output vs. Time (Left Column) ---
        ax_time = axes[row_idx, 0]
        variation_colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))

        for i, p_val in enumerate(param_values):
            current_params = default_params.copy()
            current_params[param_name] = p_val  # Set the varying parameter value

            output = model_function(
                t_arr, fixed_dist_for_time_plot,
                current_params['fraction'], current_params['alpha'],
                current_params['beta'], current_params['gamma']
            )
            ax_time.plot(t_arr, output, label=f'{param_name}={p_val}', color=variation_colors[i])

        #ax_time.set_title(f'{param_name} vs time @ dist={fixed_dist_for_time_plot}', fontsize=16)
        ax_time.set_xlabel('Time (t)', fontsize=14)
        ax_time.set_ylabel('Yield Factor', fontsize=14)
        ax_time.legend(fontsize=16)
        ax_time.grid(False)  # Remove grid
        ax_time.tick_params(axis='x', labelsize=14)
        ax_time.tick_params(axis='y', labelsize=14)

        # --- Plot 2: Output vs. Distance (Right Column) ---
        ax_dist = axes[row_idx, 1]

        for i, p_val in enumerate(param_values):
            current_params = default_params.copy()
            current_params[param_name] = p_val  # Set the varying parameter value

            output = model_function(
                fixed_time_for_dist_plot, dist_arr,
                current_params['fraction'], current_params['alpha'],
                current_params['beta'], current_params['gamma']
            )
            ax_dist.plot(dist_arr, output, label=f'{param_name}={p_val}', color=variation_colors[i])

        #ax_dist.set_title(f'{param_name} vs distance @ time={fixed_time_for_dist_plot}', fontsize=16)
        ax_dist.set_xlabel('Distance (dist)', fontsize=14)
        # ax_dist.set_ylabel('Function Output') # Y-label is shared
        ax_dist.legend(fontsize=16)
        ax_dist.grid(False)  # Remove grid
        ax_dist.tick_params(axis='x', labelsize=14)
        ax_dist.tick_params(axis='y', labelsize=14)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
    output_file = os.path.join(syn_farm_dir, "plots", "ei", "data", "yield_factor.svg")
    plt.savefig(output_file)
    plt.close()


def poll_pest_plot():
    def pollination_effect(t_arr, dist, fraction, alpha, beta, gamma):
        """Calculates the pollination effect value."""
        t_term = (1 - np.exp(-gamma * np.array(t_arr)))
        return fraction * alpha * np.exp(-beta * dist) * t_term

    def pest_control_effect(t_arr, dist, fraction, delta, epsilon, zeta):
        """Calculates the pest control effect value."""
        t_term = (1 - np.exp(-zeta * np.array(t_arr)))
        return fraction * delta * np.exp(-epsilon * dist) * t_term

    # Provided parameters
    params = cfg.params

    # Global settings
    group_A_crops = ['Spring wheat', 'Barley', 'Corn', 'Oats']
    group_A_label = "Wheat/Barley/Corn/Oats"
    group_A_rep = 'Spring wheat'  # Representative crop for params

    group_B_crops = ['Canola/rapeseed']
    group_B_label = "Canola/rapeseed"
    group_B_rep = 'Canola/rapeseed'

    group_C_crops = ['Soybeans']
    group_C_label = "Soybeans"
    group_C_rep = 'Soybeans'

    # Group for Pest Control overlap
    group_BC_pest_label = "Canola/Soybeans"
    group_BC_pest_rep = 'Canola/rapeseed'  # Use Canola's params for pest control (identical to Soybeans)

    # Global settings
    fraction = 0.5
    t_max = 50
    t_arr = np.arange(1, t_max + 1)
    dist_arr = np.linspace(0, 200, 100)  # Distance range

    # Assign colors to the effective groups/lines
    # Need 3 distinct colors for Pollination, reuse for Pest Control
    colors = cm.tab10(np.linspace(0, 1, 3))
    color_map = {
        group_A_label: colors[0],
        group_B_label: colors[1],
        group_C_label: colors[2],
        group_BC_pest_label: colors[1]  # Reuse Canola's color for combined pest group
    }

    # Font sizes
    tick_label_fontsize = 14
    axis_label_fontsize = 14
    title_fontsize = 16
    legend_fontsize = 16
    legend_title_fontsize = 16  # User request

    # --- Create Combined Panel Plot (2x4 layout) ---

    fig, axes = plt.subplots(4, 2, figsize=(14, 18),
                             sharex='col')  # 4 rows, 2 cols. Share X axis within each column. Adjusted figsize.
    #fig.suptitle('Pollination and Pest Control Effects (Grouped Crops)', fontsize=18, y=0.99)

    # Define mapping for the new 4x2 layout
    plot_configs = [
        # Row 0: Margin, Pollination
        {'row': 0, 'col': 0, 'effect': 'pollination', 'condition': 'margin', 'x_axis': 'time'},
        {'row': 0, 'col': 1, 'effect': 'pollination', 'condition': 'margin', 'x_axis': 'distance'},
        # Row 1: Margin, Pest Control
        {'row': 1, 'col': 0, 'effect': 'pest_control', 'condition': 'margin', 'x_axis': 'time'},
        {'row': 1, 'col': 1, 'effect': 'pest_control', 'condition': 'margin', 'x_axis': 'distance'},
        # Row 2: Habitat, Pollination
        {'row': 2, 'col': 0, 'effect': 'pollination', 'condition': 'habitat', 'x_axis': 'time'},
        {'row': 2, 'col': 1, 'effect': 'pollination', 'condition': 'habitat', 'x_axis': 'distance'},
        # Row 3: Habitat, Pest Control
        {'row': 3, 'col': 0, 'effect': 'pest_control', 'condition': 'habitat', 'x_axis': 'time'},
        {'row': 3, 'col': 1, 'effect': 'pest_control', 'condition': 'habitat', 'x_axis': 'distance'},
    ]

    for config in plot_configs:
        ax = axes[config['row'], config['col']]
        condition = config['condition']
        effect_type = config['effect']

        # Determine which groups to plot based on effect type
        if effect_type == 'pollination':
            groups_to_plot = [
                {'label': group_A_label, 'rep': group_A_rep},
                {'label': group_B_label, 'rep': group_B_rep},
                {'label': group_C_label, 'rep': group_C_rep},
            ]
        else:  # pest_control
            groups_to_plot = [
                {'label': group_A_label, 'rep': group_A_rep},
                {'label': group_BC_pest_label, 'rep': group_BC_pest_rep},
            ]

        # Plot data for each group
        for group in groups_to_plot:
            label = group['label']
            rep_crop = group['rep']
            crop_params = params['crops'][rep_crop][condition]
            color = color_map[label]

            if effect_type == 'pollination':
                alpha = crop_params['alpha']
                beta = crop_params['beta']
                gamma = crop_params['gamma']
                if config['x_axis'] == 'time':
                    fixed_dist = 0
                    y_values = pollination_effect(t_arr, fixed_dist, fraction, alpha, beta, gamma)
                    ax.plot(t_arr, y_values, label=label, color=color)
                else:
                    fixed_time = t_max
                    y_values = pollination_effect(fixed_time, dist_arr, fraction, alpha, beta, gamma)
                    ax.plot(dist_arr, y_values, label=label, color=color)
            else:  # pest_control
                delta = crop_params['delta']
                epsilon = crop_params['epsilon']
                zeta = crop_params['zeta']
                if config['x_axis'] == 'time':
                    fixed_dist = 0
                    y_values = pest_control_effect(t_arr, fixed_dist, fraction, delta, epsilon, zeta)
                    ax.plot(t_arr, y_values, label=label, color=color)
                else:
                    fixed_time = t_max
                    y_values = pest_control_effect(fixed_time, dist_arr, fraction, delta, epsilon, zeta)
                    ax.plot(dist_arr, y_values, label=label, color=color)

        # --- Subplot Formatting ---
        effect_label = effect_type.replace('_', ' ').title()
        condition_label = condition.title()
        x_axis_label = config['x_axis'].title()
        fixed_val_label = f"Dist=0" if config['x_axis'] == 'time' else f"Time={t_max}"

        # Set titles - simplified slightly for the new layout
        ax.set_title(f'{condition_label}: {effect_label} (@ {fixed_val_label})', fontsize=title_fontsize)

        # Set X labels only for the bottom row plots
        if config['row'] == 3:
            if config['x_axis'] == 'time':
                ax.set_xlabel('Time (t)', fontsize=axis_label_fontsize)
            else:
                ax.set_xlabel('Distance (dist)', fontsize=axis_label_fontsize)

        # Set Y labels only for the first column plots
        if config['col'] == 0:
            ax.set_ylabel(f'Effect Level', fontsize=axis_label_fontsize)  # Generalized Y label

        ax.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax.tick_params(axis='y', labelsize=tick_label_fontsize)
        ax.grid(False)

        # Add legend INSIDE each subplot
        ax.legend(title="Crops", title_fontsize=legend_title_fontsize, fontsize=legend_fontsize)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.01, 1, 0.96])  # Adjust rect to prevent main title overlap
    plt.subplots_adjust(hspace=0.5, wspace=0.2)# Adjust layout to prevent title overlap and make space for legend
    output_file = os.path.join(syn_farm_dir, "plots", "ei", "data", "poll_pest_mh_crops.svg")
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":
    cfg = Config()
    exit_tol = 1e-6
    neighbor_dist = 1000 #1500
    penalty_coef = 1e3
    mode = "real_farms" #syn_farms, real_farms

    syn_farm_dir = os.path.join(cfg.disk_dir, "crop_inventory", "syn_farms")
    if mode == "syn_farms":
        base_farm_dir = os.path.join(syn_farm_dir, "mc")
        num_configs = np.arange(1, 501)
    else:
        base_farm_dir = os.path.join(cfg.disk_dir, "crop_inventory", "farms_config_s")
        num_configs = np.arange(1, 571)


    # run_analysis(cfg, num_configs, base_farm_dir, neighbor_dist, exit_tol, penalty_coef)
    # neib_penalty_sensitivity()
    # run_alpha_delta_analysis()
    # run_economic_sensitivity_analysis()
    # analyze_temporal_npv()
    # yield_factor_plot()
    # poll_pest_plot()


