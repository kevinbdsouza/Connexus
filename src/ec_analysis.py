from pyomo.core.expr.sympy_tools import sympy
import math
from config import Config
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import networkx as nx
from graph_connectivity import optimize_landscape_connectivity, build_connectivity_graph_from_chosen_pieces, plot_farms, \
    compute_connectivity_metric, parse_geojson
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon as ShapelyPolygon, LineString as ShapelyLineString
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
import itertools
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from shapely import wkt as shapely_wkt
import geopandas as gpd
from matplotlib.patches import Polygon as MplPolygon
import warnings
from matplotlib.collections import PatchCollection

FONT_SIZE = 25
plt.rcParams['axes.labelsize'] = FONT_SIZE  # For x and y labels
plt.rcParams['axes.titlesize'] = FONT_SIZE  # For the subplot title
plt.rcParams['xtick.labelsize'] = FONT_SIZE  # For x-axis tick labels
plt.rcParams['ytick.labelsize'] = FONT_SIZE  # For y-axis tick labels
plt.rcParams['legend.fontsize'] = FONT_SIZE  # For the legend
plt.rcParams['figure.titlesize'] = FONT_SIZE

def run_all_configs():
    for config_id in num_configs:
        print(f"Running config: {config_id}")
        config_path = os.path.join(base_farm_dir, f"config_{config_id}")
        all_plots_geojson = os.path.join(config_path, f"all_plots_interventions_og.geojson")
        try:
            _, _, conn_val_final, conn_val_repos, _, _, _ = optimize_landscape_connectivity(all_plots_geojson, boundary_seg_count,
                                                                                      interior_cell_count, adjacency_dist,
                                                                                      connectivity_metric, al_factor,
                                                                                      max_loss_ratio,
                                                                                      neib_dist, exit_tol, reposition,
                                                                                      params, config_path, margin_weight,
                                                                                      mode, plot)
        except Exception as e:
            continue


def analyze_repositioning_results():
    def create_gdf_from_plots(plots_list):
        if not plots_list:
            return None
        try:
            # Ensure all geometries are valid Shapely objects
            valid_plots = []
            for p in plots_list:
                geom = p.get('geometry')
                if geom and not geom.is_empty:
                    # Attempt to fix invalid geometries
                    if not geom.is_valid:
                        geom_fixed = geom.buffer(0)
                        if geom_fixed and not geom_fixed.is_empty and geom_fixed.is_valid:
                            p['geometry'] = geom_fixed
                            valid_plots.append(p)
                        else:
                            warnings.warn(
                                f"Could not fix invalid geometry for plot {p.get('plot_id', 'N/A')}. Skipping.")
                    else:
                        valid_plots.append(p)
                else:
                    warnings.warn(f"Plot {p.get('plot_id', 'N/A')} has empty or missing geometry. Skipping.")

            if not valid_plots:
                print("Error: No valid plot geometries found to create GeoDataFrame.")
                return None

            gdf = gpd.GeoDataFrame(valid_plots, geometry='geometry')

            # Ensure geometries are still valid after potential CRS operations if any happened implicitly
            gdf['geometry'] = gdf.geometry.buffer(0)
            gdf = gdf[gdf.is_valid & ~gdf.is_empty]
            if gdf.empty:
                print("Error: GeoDataFrame became empty after validity checks.")
                return None
            return gdf

        except Exception as e:
            print(f"Error creating GeoDataFrame from plots list: {e}")
            return None

    # --- Modified plot_farms to handle gap patches and legend ---
    def plot_farms(ax, plots, chosen_pieces=None, gap_patches=None, title="Farm Plots"):
        # 1. Plot Gaps first (if provided) so they are in the background
        gap_collection = None
        if gap_patches:
            gap_collection = PatchCollection(gap_patches, facecolor='forestgreen',
                                             edgecolor='none', alpha=0.6)
            ax.add_collection(gap_collection)

        # 2. Plot Base Farm Polygons
        base_patches = []
        plot_types_present = set()
        for p in plots:
            geom = p['geometry']
            plot_type = p.get('plot_type', 'unknown')
            plot_types_present.add(plot_type)
            if geom.is_empty:
                continue

            face_color = 'lightgray'  # Default for ag_plot
            if plot_type == 'hab_plots':
                face_color = 'forestgreen'
            # Add more conditions if other base plot types exist

            if geom.geom_type == 'Polygon':
                coords = list(geom.exterior.coords)
                poly_patch = MplPolygon(coords, closed=True, facecolor=face_color, edgecolor='black', alpha=1.0)
                base_patches.append(poly_patch)
            elif geom.geom_type == 'MultiPolygon':
                for subg in geom.geoms:
                    if subg.is_empty: continue
                    coords = list(subg.exterior.coords)
                    poly_patch = MplPolygon(coords, closed=True, facecolor=face_color, edgecolor='black', alpha=1.0)
                    base_patches.append(poly_patch)

        base_pc = PatchCollection(base_patches, match_original=True)
        ax.add_collection(base_pc)

        # 3. Highlight chosen pieces (on top)
        margin_pieces_present = False
        habitat_pieces_present = False
        if chosen_pieces:
            for c in chosen_pieces:
                g = c['geom']
                piece_type = c['type']
                if piece_type == 'margin':
                    margin_pieces_present = True
                    if hasattr(g, 'xy'):  # LineString
                        x, y = g.xy
                        ax.plot(x, y, color='red', linewidth=2)
                elif piece_type == 'habitat_patch':
                    habitat_pieces_present = True
                    patch_color = 'lime'
                    edge_color = 'green'
                    if g.geom_type == 'Polygon':
                        coords = list(g.exterior.coords)
                        poly_patch = MplPolygon(coords, closed=True, facecolor=patch_color, edgecolor=edge_color,
                                                alpha=0.6)
                        ax.add_patch(poly_patch)
                    elif g.geom_type == 'MultiPolygon':
                        for subp in g.geoms:
                            if subp.is_empty: continue
                            coords = list(subp.exterior.coords)
                            poly_patch = MplPolygon(coords, closed=True, facecolor=patch_color, edgecolor=edge_color,
                                                    alpha=0.6)
                            ax.add_patch(poly_patch)
                elif piece_type == 'full_habitat':
                    # These are already plotted by the base plot loop if hab_plots type exists
                    pass

        #ax.set_title(title)
        ax.set_aspect('equal', 'box')
        ax.autoscale_view()
        ax.set_xlabel("Easting (m)",  fontsize=25)
        ax.set_ylabel("Northing (m)",  fontsize=25)
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')

        # 4. Build Legend Handles Dynamically
        legend_handles = []
        legend_handles.append(
            mpatches.Patch(facecolor='lightgray', edgecolor='black', alpha=1.0, label='Agricultural Plots'))
        legend_handles.append(
            mpatches.Patch(facecolor='forestgreen', edgecolor='black', alpha=1.0, label='Existing Habitats'))
        legend_handles.append(
            mpatches.Patch(facecolor='forestgreen', alpha=0.6, label='Existing Habitats (Gap Filled)'))
        legend_handles.append(mlines.Line2D([], [], color='red', linewidth=2, label='Margin Interventions'))
        legend_handles.append(
            mpatches.Patch(facecolor='lime', edgecolor='green', alpha=0.6, label='Habitat Conversions'))

        if legend_handles:
            ax.legend(handles=legend_handles, loc='best', fontsize=25)

    load = False
    all_plots = True
    if farms == "syn_farms":
        analysis_output_dir = os.path.join(syn_farm_dir, "plots", "ec", "repositioning")
    else:
        analysis_output_dir = os.path.join(syn_farm_dir, "plots", "ec", "repositioning_real")

    repositioning_connectivity_scores = []
    final_connectivity_scores = []
    failed_configs = []
    all_repositioning_pieces = []
    all_final_pieces = []

    # Ensure analysis output directory exists
    if not os.path.exists(analysis_output_dir):
        os.makedirs(analysis_output_dir)

    if not load:
        print(f"\n--- Starting Analysis for {len(list(num_configs))} Configurations ---")
        print(f"Repositioning Enabled: {reposition}")
        print(f"Connectivity Metric: {connectivity_metric}")

        all_plots_list = []
        for config_id in num_configs:
            print(f"\nProcessing config: {config_id}")
            config_path = os.path.join(base_farm_dir, f"config_{config_id}")
            # Define the specific path where optimization outputs for this config will go
            all_plots_geojson = os.path.join(config_path, f"all_plots_interventions_{mode}.geojson")
            if not os.path.exists(all_plots_geojson):
                failed_configs.append(config_id)
                continue

            plots_in_config = parse_geojson(all_plots_geojson)
            for p in plots_in_config:
                all_plots_list.append(p)

            try:
                chosen_final, _, final_conn, repo_conn, _, _, _, chosen_repos = optimize_landscape_connectivity(
                    all_plots_geojson, boundary_seg_count,
                    interior_cell_count, adjacency_dist, connectivity_metric, al_factor,
                    max_loss_ratio, neib_dist, exit_tol, reposition, params,
                    config_path,  # Pass the specific output dir for this config
                    margin_weight, mode, plot)

                # Store results
                repositioning_connectivity_scores.append(repo_conn)
                all_repositioning_pieces.extend(chosen_repos)

                final_connectivity_scores.append(final_conn)
                all_final_pieces.extend(chosen_final)

                #if repo_conn > final_conn:
                #    final_connectivity_scores.append(repo_conn)
                #    all_final_pieces.extend(chosen_repos)
                #else:
                #    final_connectivity_scores.append(final_conn)
                #    all_final_pieces.extend(chosen_final)

            except Exception as e:
                print(f"  ERROR processing config {config_id}: {e}")
                failed_configs.append(config_id)

        repositioning_connectivity_scores = np.array(repositioning_connectivity_scores)
        final_connectivity_scores = np.array(final_connectivity_scores)
        np.save(os.path.join(analysis_output_dir, "repositioning_connectivity_scores.npy"),
                repositioning_connectivity_scores)
        np.save(os.path.join(analysis_output_dir, "final_connectivity_scores.npy"),
                final_connectivity_scores)
    else:
        repositioning_connectivity_scores = np.load(
            os.path.join(analysis_output_dir, "repositioning_connectivity_scores.npy"))
        final_connectivity_scores = np.load(os.path.join(analysis_output_dir, "final_connectivity_scores.npy"))


    plt.figure(figsize=(12, 7))  # Adjusted figure size for better readability
    diff = final_connectivity_scores - repositioning_connectivity_scores
    # Plot KDE for Repositioning Scores
    sns.kdeplot(diff,
                label=f'Optimized - Repositioned',
                color='lightcoral',
                fill=True,
                alpha=0.6,
                clip=(0, 50000))  # Use alpha for transparency if areas overlap

    avg_diff = np.mean(diff)
    plt.axvline(avg_diff, color='red', linestyle='--', linewidth=1,
                label=f'Mean Change: {avg_diff:.1e}')
    plt.xlabel(f'Change in Connectivity Score ({connectivity_metric})', fontsize=14)  # Increased x-label font size
    plt.ylabel('Density', fontsize=14)  # Increased y-label font size
    plt.xticks(fontsize=14)  # Increased x-tick label font size
    plt.yticks(fontsize=14)  # Increased y-tick label font size
    plt.legend(fontsize=16)  # Increased legend font size
    plt.grid(False)  # Explicitly disable grid lines
    plt.tight_layout()  # Adjust layout

    combined_plot_filename = os.path.join(analysis_output_dir, f'connectivity_distribution_diff.svg')
    plt.savefig(combined_plot_filename)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.kdeplot(repositioning_connectivity_scores,
                label=f'After Repositioning',
                color='skyblue',
                fill=True,
                alpha=0.6,
                clip=(0, 50000))  # Use alpha for transparency if areas overlap

    # Plot KDE for Final Scores on the same axes
    sns.kdeplot(final_connectivity_scores,
                label=f'After Connectivity Optimization',
                color='lightcoral',
                fill=True,
                alpha=0.6,
                clip=(0, 50000))

    avg_final_repos = np.mean(repositioning_connectivity_scores)
    plt.axvline(avg_final_repos, color='blue', linestyle='--', linewidth=1,
                label=f'Mean Repositioned: {avg_final_repos:.1e}')

    avg_final_conn = np.mean(final_connectivity_scores)
    plt.axvline(avg_final_conn, color='red', linestyle='--', linewidth=1,
                label=f'Mean Final: {avg_final_conn:.1e}')
    # --- End Optional Mean Lines ---

    plt.xlabel(f'Connectivity Score ({connectivity_metric})', fontsize=25)  # Increased x-label font size
    plt.ylabel('Density', fontsize=25)  # Increased y-label font size
    plt.xticks(fontsize=25)  # Increased x-tick label font size
    plt.yticks(fontsize=25)  # Increased y-tick label font size
    plt.legend(fontsize=25)  # Increased legend font size
    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')
    plt.grid(False)  # Explicitly disable grid lines
    plt.tight_layout()  # Adjust layout

    # Save the combined plot
    combined_plot_filename = os.path.join(analysis_output_dir, f'combined_connectivity_distribution.svg')
    plt.savefig(combined_plot_filename)
    plt.close()

    if all_plots:
        print("\n--- Plotting Combined Intervention Maps ---")

        gap_patches_mpl = []  # List to hold matplotlib patches for gaps
        print("Converting all plots to GeoDataFrame...")
        all_plots_gdf = create_gdf_from_plots(all_plots_list)

        if all_plots_gdf is not None and not all_plots_gdf.empty:
            print("Calculating unary union and convex hull...")
            try:
                # Ensure valid geometries before union/hull
                all_plots_gdf['geometry'] = all_plots_gdf.geometry.buffer(0)
                all_plots_gdf = all_plots_gdf[all_plots_gdf.is_valid & ~all_plots_gdf.is_empty]

                if not all_plots_gdf.empty:
                    occupied_area = unary_union(all_plots_gdf.geometry)
                    if not occupied_area.is_valid:
                        warnings.warn("Unary union resulted in invalid geometry. Applying buffer(0).")
                        occupied_area = occupied_area.buffer(0)

                    overall_hull = occupied_area.convex_hull  # Hull of the combined plots
                    if not overall_hull.is_valid:
                        warnings.warn("Convex hull resulted in invalid geometry. Applying buffer(0).")
                        overall_hull = overall_hull.buffer(0)

                    print("Calculating gaps (hull - occupied area)...")
                    gaps_geom = overall_hull.difference(occupied_area)

                    if gaps_geom and not gaps_geom.is_empty:
                        print("Processing gap geometries for plotting...")
                        if gaps_geom.geom_type == 'Polygon':
                            if gaps_geom.exterior:  # Check if it has coordinates
                                gap_patches_mpl.append(MplPolygon(list(gaps_geom.exterior.coords), closed=True))
                        elif isinstance(gaps_geom, MultiPolygon):
                            for poly in gaps_geom.geoms:
                                if poly and not poly.is_empty and poly.exterior:
                                    gap_patches_mpl.append(MplPolygon(list(poly.exterior.coords), closed=True))
                        print(f"Created {len(gap_patches_mpl)} gap patches for plotting.")
                    else:
                        print("No significant gaps found between plots within the convex hull.")
                else:
                    print("No valid geometries remained after buffering for union/hull calculation.")

            except Exception as e:
                print(f"Error during gap calculation: {e}")
                # import traceback
                # traceback.print_exc() # Uncomment for detailed traceback
                gap_patches_mpl = []  # Ensure it's empty on error
        else:
            print("Skipping gap calculation as GeoDataFrame could not be created or was empty.")

        # --- Plot combined repositioning results (with gaps) ---
        if all_repositioning_pieces or gap_patches_mpl:  # Plot even if only gaps exist
            print(f"Generating combined repositioning map...")
            fig_repo, ax_repo = plt.subplots(figsize=(14, 11))
            plot_farms(ax_repo, all_plots_list, all_repositioning_pieces, gap_patches=gap_patches_mpl,
                       title=f"Combined Repositioning ({connectivity_metric}, {len(num_configs)} Configs)")
            repo_combined_plot_filename = os.path.join(analysis_output_dir,
                                                       f'combined_repositioning_with_gaps_{connectivity_metric}.svg')
            try:
                plt.savefig(repo_combined_plot_filename, bbox_inches='tight', dpi=150)
                print(f"Saved combined repositioning plot to: {repo_combined_plot_filename}")
            except Exception as e:
                print(f"ERROR saving combined repositioning plot: {e}")
            plt.close(fig_repo)
        else:
            print("No repositioning pieces or gaps to plot.")

        # --- Plot combined final results (with gaps) ---
        if all_final_pieces or gap_patches_mpl:  # Plot even if only gaps exist
            print(f"Generating combined final optimization map...")
            fig_final, ax_final = plt.subplots(figsize=(14, 11))
            plot_farms(ax_final, all_plots_list, all_final_pieces, gap_patches=gap_patches_mpl,
                       title=f"Combined Final Optimization ({connectivity_metric}, {len(num_configs)} Configs)")
            final_combined_plot_filename = os.path.join(analysis_output_dir,
                                                        f'combined_final_with_gaps_{connectivity_metric}.svg')
            try:
                plt.savefig(final_combined_plot_filename, bbox_inches='tight', dpi=150)
                print(f"Saved combined final optimization plot to: {final_combined_plot_filename}")
            except Exception as e:
                print(f"ERROR saving combined final plot: {e}")
            plt.close(fig_final)
        else:
            print("No final chosen pieces or gaps to plot.")

    elif load:
        print(
            "\n--- Skipping Combined Intervention Maps (Run with load=False and ensure pieces are saved/loaded if needed) ---")
    elif not all_plots_list:
        print("\n--- Skipping Combined Intervention Maps (Base plots could not be loaded) ---")

    if failed_configs:
        print(f"\n--- WARNING: Processing failed for {len(failed_configs)} configurations: {failed_configs} ---")

    print("\n--- Analysis Complete ---")


def calculate_touching_plot_neighbors(plots):
    plot_neighbors = defaultdict(int)

    valid_plots_info = []
    for idx, p in enumerate(plots):
        geom = p.get('geometry')
        if geom and not geom.is_empty:  # and geom.is_valid:
            valid_plots_info.append((idx, geom))

    if not valid_plots_info:
        return dict(plot_neighbors)

    num_valid_plots = len(valid_plots_info)

    for k, l in itertools.combinations(range(num_valid_plots), 2):
        original_idx_i, geom_i = valid_plots_info[k]
        original_idx_j, geom_l = valid_plots_info[l]

        if geom_i.touches(geom_l):
            plot_neighbors[original_idx_i] += 1
            plot_neighbors[original_idx_j] += 1
    return dict(plot_neighbors)


def analyze_connectivity_hubs():
    def run_hubs(chosen_pieces, plots, adjacency_dist, top_n_percent=10):
        if not chosen_pieces:
            print("No chosen pieces to analyze.")
            return pd.DataFrame(), nx.Graph(), {}

        print(f"\n--- Analyzing Connectivity Hubs ({len(chosen_pieces)} chosen pieces) ---")

        # 1. Build the Connectivity Graph
        G = build_connectivity_graph_from_chosen_pieces(chosen_pieces, adjacency_dist)
        print(f"Connectivity graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

        if G.number_of_nodes() == 0:
            print("Graph is empty, cannot perform centrality analysis.")
            return pd.DataFrame(), G, {}

        intra_farm_score_total = 0.0
        inter_farm_score_total = 0.0
        intra_edges = []
        inter_edges = []
        total_pairwise_score = 0.0
        plot_inter_weight = defaultdict(float)  # Store sum of inter-farm edge weights per plot_index
        plot_intra_weight = defaultdict(float)  # Store sum of intra-farm edge weights per plot_index

        for u, v in G.edges():
            if u < len(chosen_pieces) and v < len(chosen_pieces):
                pc_i = chosen_pieces[u]
                pc_j = chosen_pieces[v]
                farm_id_i = pc_i['farm_id']
                farm_id_j = pc_j['farm_id']
                plot_idx_i = pc_i['plot_index']  # Get the original plot index for piece i
                plot_idx_j = pc_j['plot_index']  # Get the original plot index for piece j

                ai = pc_i.get('area', 0.0)
                aj = pc_j.get('area', 0.0)
                li = pc_i.get('length', 0.0)
                lj = pc_j.get('length', 0.0)

                is_i_full_habitat = pc_i['type'] == "full_habitat"
                is_j_full_habitat = pc_j['type'] == "full_habitat"

                # Calculate w_ij based on IIC-like logic from build_connectivity_ilp
                if is_i_full_habitat or is_j_full_habitat:
                    w_ij = (li * lj) + (ai * aj) + al_factor * (ai * lj + aj * li)
                else:
                    w_ij = (li * lj) + (ai * aj) + (ai * lj) + (aj * li)

                total_pairwise_score += w_ij

                if farm_id_i == farm_id_j:
                    intra_farm_score_total += w_ij
                    intra_edges.append((u, v))
                    # Add weight to both plots involved in the intra-farm edge
                    plot_intra_weight[plot_idx_i] += w_ij
                    plot_intra_weight[plot_idx_j] += w_ij
                else:
                    inter_farm_score_total += w_ij
                    inter_edges.append((u, v))
                    # Add weight to both plots involved in the inter-farm edge
                    plot_inter_weight[plot_idx_i] += w_ij
                    plot_inter_weight[plot_idx_j] += w_ij
            else:
                print(f"Warning: Edge ({u}, {v}) references index outside chosen_pieces range.")

        # Store plot weights for return
        plot_weights = {
            'inter': dict(plot_inter_weight),
            'intra': dict(plot_intra_weight)
        }

        print("\n--- Inter- vs Intra-Farm Connectivity Analysis (Overall Landscape) ---")
        if total_pairwise_score > 1e-9:
            percent_intra = (intra_farm_score_total / total_pairwise_score) * 100
            percent_inter = (inter_farm_score_total / total_pairwise_score) * 100
            print(f"Pairwise Connectivity Score Contribution:")
            print(f"  Intra-Farm: {intra_farm_score_total:.4f} ({percent_intra:.1f}%) - {len(intra_edges)} edges")
            print(f"  Inter-Farm: {inter_farm_score_total:.4f} ({percent_inter:.1f}%) - {len(inter_edges)} edges")
            print(f"  Total Pairwise Score: {total_pairwise_score:.4f}")
        else:
            print("No connected pairs found (Total Pairwise Score is zero).")

        # 2. Calculate Centrality Metrics
        try:
            degree_centrality = nx.degree_centrality(G)
        except Exception as e:
            print(f"Could not calculate degree centrality: {e}")
            degree_centrality = {n: 0 for n in G.nodes()}

        try:
            # Using normalized betweenness centrality. Weight=None treats edges equally.
            betweenness_centrality = nx.betweenness_centrality(G, normalized=True, weight=None)
        except Exception as e:
            print(f"Could not calculate betweenness centrality: {e}")
            betweenness_centrality = {n: 0 for n in G.nodes()}

        # Store centralities for potential later use
        centralities = {
            'degree': degree_centrality,
            'betweenness': betweenness_centrality
        }
        # Add centrality scores as node attributes
        nx.set_node_attributes(G, degree_centrality, "degree_centrality")
        nx.set_node_attributes(G, betweenness_centrality, "betweenness_centrality")

        # 3. Aggregate Centrality by Plot
        plot_aggregated_centrality = defaultdict(lambda: {'degree_sum': 0.0, 'betweenness_sum': 0.0, 'piece_count': 0})
        plot_piece_indices = defaultdict(list)

        for i, piece in enumerate(chosen_pieces):
            # Ensure node i exists in the graph (it should if G was built correctly)
            if i in G:
                plot_idx = piece['plot_index']
                plot_aggregated_centrality[plot_idx]['degree_sum'] += degree_centrality.get(i, 0.0)
                plot_aggregated_centrality[plot_idx]['betweenness_sum'] += betweenness_centrality.get(i, 0.0)
                plot_aggregated_centrality[plot_idx]['piece_count'] += 1
                plot_piece_indices[plot_idx].append(i)  # Store which pieces belong to this plot

        plot_neighbor_counts = calculate_touching_plot_neighbors(plots)

        existing_habitat_geoms = [p['geometry'] for p in plots if
                                  p['plot_type'] == 'hab_plots' and p['geometry'] and not p['geometry'].is_empty]
        existing_habitat_multipoly = None
        if existing_habitat_geoms:
            valid_habitat_geoms = [g for g in existing_habitat_geoms if g.is_valid]
            if valid_habitat_geoms:
                existing_habitat_multipoly = MultiPolygon(valid_habitat_geoms)

        plot_dist_to_habitat = {}
        for plot_idx, p in enumerate(plots):
            if p['geometry'] and not p['geometry'].is_empty and p['geometry'].centroid:
                centroid = p['geometry'].centroid
                if not centroid.is_empty:
                    if existing_habitat_multipoly:
                        plot_dist_to_habitat[plot_idx] = centroid.distance(existing_habitat_multipoly)
                    else:
                        plot_dist_to_habitat[plot_idx] = np.inf  # No habitat exists
                else:
                    plot_dist_to_habitat[plot_idx] = np.nan  # Cannot calculate distance
            else:
                plot_dist_to_habitat[plot_idx] = np.nan

        # 4. Identify Hub Plots & 5. Analyze Characteristics
        plot_analysis_data = []
        num_ag_plots = sum(1 for p in plots if p['plot_type'] == 'ag_plot')
        num_hub_plots = max(1, int(num_ag_plots * (top_n_percent / 100.0)))

        for plot_idx, p in enumerate(plots):
            # We need info for ALL ag_plots and hab_plots for comparison, not just those with chosen pieces
            if p['plot_type'] not in ['ag_plot', 'hab_plots']:
                continue
            if not p['geometry'] or p['geometry'].is_empty:
                continue

            agg_data = plot_aggregated_centrality.get(plot_idx,
                                                      {'degree_sum': 0.0, 'betweenness_sum': 0.0, 'piece_count': 0})

            # Calculate num_sides (simple version for Polygon)
            num_sides = np.nan
            if p['geometry'].geom_type == 'Polygon':
                coords = list(p['geometry'].exterior.coords)
                if len(coords) > 1:
                    num_sides = len(coords) - 1  # Subtract 1 because start/end point is duplicated

            inter_w = plot_weights['inter'].get(plot_idx, 0.0)
            intra_w = plot_weights['intra'].get(plot_idx, 0.0)
            total_w = inter_w + intra_w
            percent_inter_w = (inter_w / total_w) * 100 if total_w > 1e-9 else 0.0

            analysis_entry = {
                'plot_index': plot_idx,
                'farm_id': p['farm_id'],
                'plot_id': p['plot_id'],
                'plot_type': p['plot_type'],
                'crop_label': p.get('label', 'N/A'),
                'area_ha': p['geometry'].area / 10000,
                'perimeter_km': p['geometry'].length / 1000 if p['geometry'].length > 0 else 0,
                'original_margin_frac': p.get('margin_frac', 0.0),  # Keep original info if needed
                'original_habitat_frac': p.get('habitat_frac', 0.0),  # Keep original info if needed
                'num_chosen_pieces': agg_data['piece_count'],
                'sum_betweenness_centrality': agg_data['betweenness_sum'],
                'avg_betweenness_centrality': (agg_data['betweenness_sum'] / agg_data['piece_count'])
                if agg_data['piece_count'] > 0 else 0,
                'yield': p.get('yield', np.nan),  # Get yield
                'num_sides': num_sides,  # Add calculated sides
                'num_neighbors': plot_neighbor_counts.get(plot_idx, 0),  # Add neighbor count
                'dist_habitat': plot_dist_to_habitat.get(plot_idx, np.nan),  # Add dist to habitat
                'sum_inter_farm_weight': inter_w,
                'sum_intra_farm_weight': intra_w,
                'percent_inter_farm_weight': percent_inter_w
            }
            plot_analysis_data.append(analysis_entry)

        if not plot_analysis_data:
            print("No plots found with chosen pieces for analysis.")
            return pd.DataFrame(), G, centralities

        # Create DataFrame for easier analysis
        hub_analysis_df = pd.DataFrame(plot_analysis_data)

        # Sort by betweenness centrality (often good for stepping stones)
        hub_analysis_df = hub_analysis_df.sort_values(by='sum_betweenness_centrality', ascending=False)

        # Add a flag for top hub plots
        hub_analysis_df['is_hub'] = False
        ag_plot_indices = hub_analysis_df[hub_analysis_df['plot_type'] == 'ag_plot'].index
        top_indices = ag_plot_indices[:num_hub_plots]
        hub_analysis_df.loc[top_indices, 'is_hub'] = True

        print(f"\n--- Top {num_hub_plots} Potential Hub Plots (by Summed Betweenness Centrality) ---")
        print(hub_analysis_df[hub_analysis_df['is_hub']][
                  ['plot_index', 'farm_id', 'plot_id', 'crop_label', 'area_ha', 'num_chosen_pieces',
                   'sum_betweenness_centrality']].head(num_hub_plots))

        print("\n--- Characteristics Comparison ---")
        hub_plots_stats = hub_analysis_df[hub_analysis_df['is_hub']].agg({
            'area_ha': ['mean', 'median', 'std'],
            'num_chosen_pieces': ['mean', 'median', 'std'],
            'sum_betweenness_centrality': ['mean', 'median', 'std']
        })
        non_hub_plots_stats = hub_analysis_df[~hub_analysis_df['is_hub']].agg({
            'area_ha': ['mean', 'median', 'std'],
            'num_chosen_pieces': ['mean', 'median', 'std'],
            'sum_betweenness_centrality': ['mean', 'median', 'std']
        })

        print("\nHub Plot Stats:")
        print(hub_plots_stats)
        print("\nNon-Hub Plot Stats:")
        print(non_hub_plots_stats)

        hub_crop_dist = hub_analysis_df[hub_analysis_df['is_hub']]['crop_label'].value_counts(normalize=True)
        print("\nHub Plot Crop Distribution (%):")
        print(hub_crop_dist * 100)

        print("\n--- End Hub Analysis ---")

        return hub_analysis_df, G, centralities, intra_edges, inter_edges, intra_farm_score_total, inter_farm_score_total, plot_weights

    def plot_hubs(summary_df, output_dir, connectivity_metric_name='IIC'):
        plot_filename = "aggregate_connectivity_vs_hubs.svg"
        plot_filepath = os.path.join(output_dir, plot_filename)

        plot_df = summary_df.dropna(subset=['conn_val_final', 'num_hubs_identified']).copy()

        plt.figure(figsize=(8, 6))
        if plot_df['num_hubs_identified'].nunique() < 15:
            sns.stripplot(data=plot_df, x='num_hubs_identified', y='conn_val_final', jitter=0.2, alpha=0.7)
        else:
            sns.scatterplot(data=plot_df, x='num_hubs_identified', y='conn_val_final', alpha=0.7)
            # Optional: Add a regression line for scatter plots
            sns.regplot(data=plot_df, x='num_hubs_identified', y='conn_val_final', scatter=False, color='red',
                        line_kws={'linewidth': 1})

        plt.xlabel('Number of Hub Plots Identified (Top 10%)')
        plt.ylabel(f'Final Connectivity ({connectivity_metric_name})')
        plt.tight_layout()
        plt.savefig(plot_filepath)
        plt.close()

        plot_filename = "aggregate_conn_vs_initial_habitat_ratio.svg"
        col_name = 'initial_habitat_ratio'
        plot_df = summary_df.dropna(subset=['conn_val_final', col_name]).copy()
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=plot_df, x=col_name, y='conn_val_final', alpha=0.7)
        sns.regplot(data=plot_df, x=col_name, y='conn_val_final', scatter=False, color='red',
                    line_kws={'linewidth': 1})
        plt.xlabel('Initial Habitat Area Ratio (Habitat Area / Total Area)')
        plt.ylabel(f'Final Connectivity ({connectivity_metric_name})')
        # plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()

        plot_filename = "aggregate_hub_dist_boundary_distrib.svg"
        col_name = 'avg_hub_dist_to_boundary'
        plot_df = summary_df.dropna(subset=[col_name]).copy()
        plt.figure(figsize=(8, 5))
        sns.histplot(plot_df[col_name], kde=True, bins=15)
        plt.xlabel('Average Distance from Hub Plot Centroid to Boundary (meters)')
        plt.ylabel('Frequency (Number of Runs)')
        # plt.grid(False) # Grid removed
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()

        plot_filename = "aggregate_hub_dist_habitat_distrib.svg"
        col_name = 'avg_hub_dist_to_habitat'
        plot_df = summary_df.dropna(subset=[col_name]).copy()
        plot_df = plot_df[np.isfinite(plot_df[col_name])]
        plt.figure(figsize=(8, 5))
        sns.histplot(plot_df[col_name], kde=True, bins=15)
        plt.xlabel('Average Distance from Hub Plot Centroid to Nearest Habitat (meters)')
        plt.ylabel('Frequency (Number of Runs)')
        # plt.grid(False) # Grid removed
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()

    def plot_farms_with_connectivity_graph(
            ax,
            plots,
            chosen_pieces,
            G_conn,
            node_centralities,
            intra_edges,
            inter_edges
    ):
        # 1. Draw the base farm plots and highlighted chosen pieces
        #    plot_farms will add the patches, lines, and its own legend to ax.
        plot_farms(ax, plots, chosen_pieces, title="")  # Use empty title here, set final title later

        base_handles = []
        base_handles.append(
            mpatches.Patch(facecolor='lightgray', edgecolor='black', alpha=0.5, label='Agricultural Plots'))
        base_handles.append(mpatches.Patch(facecolor='forestgreen', edgecolor='black', alpha=0.5,
                                           label='Existing Habitats'))
        base_handles.append(mlines.Line2D([0], [0], color='red', linewidth=2, label='Margin Interventions'))
        base_handles.append(
            mpatches.Patch(facecolor='lime', edgecolor='green', alpha=0.5, label='Habitat Conversions'))

        # 3. Calculate geographic positions for graph nodes
        pos = {}
        for i, piece in enumerate(chosen_pieces):
            # Ensure node i exists in the graph (should always be true if built correctly)
            if i in G_conn:
                geom = piece['geom']
                if geom and not geom.is_empty:
                    # Use centroid for positioning the node marker
                    centroid = geom.centroid
                    if centroid.is_empty:  # Handle rare cases like zero-length lines
                        if isinstance(geom, ShapelyLineString):
                            pt = geom.interpolate(0.5, normalized=True)
                            pos[i] = pt.coords[0]
                        else:  # Skip if cannot determine position
                            print(f"Warning: Cannot determine centroid for piece {i}, skipping node position.")
                            continue
                    else:
                        pos[i] = centroid.coords[0]  # Get (x, y) tuple
                else:
                    print(f"Warning: Empty geometry for piece {i}, skipping node position.")

        # 4. Prepare Node Colors based on Betweenness Centrality
        min_centrality = 0.0
        max_centrality = 0.0
        node_colors = 'gray'  # Default
        centrality_data_available = False
        if 'betweenness' in node_centralities and node_centralities['betweenness']:
            # Create a list of colors ONLY for nodes that have a calculated position
            colors_for_plotting = []
            nodes_for_plotting = list(pos.keys())  # Get nodes that we can actually plot

            for node_id in nodes_for_plotting:
                colors_for_plotting.append(node_centralities['betweenness'].get(node_id, 0.0))

            if colors_for_plotting:
                node_colors = colors_for_plotting
                min_centrality = min(node_colors)
                max_centrality = max(node_colors) if node_colors else 0.0  # Handle empty list case
                centrality_data_available = True
            else:
                print("Warning: Could not extract centrality values for positioned nodes. Coloring nodes gray.")
        else:
            print("Warning: 'betweenness' centrality data not found or empty. Coloring nodes gray.")

        # 5. Draw the NetworkX Graph using geographic positions
        cmap = plt.cm.viridis
        node_size = 75  # Make nodes slightly larger to be visible on the map

        # Draw nodes using the geographic 'pos' dictionary
        # Only draw nodes for which we have a position
        nodes = nx.draw_networkx_nodes(G_conn, pos, ax=ax, nodelist=list(pos.keys()),  # Ensure nodelist matches pos
                                       node_size=node_size,
                                       node_color=node_colors if centrality_data_available else 'gray',
                                       cmap=cmap if centrality_data_available else None,
                                       vmin=min_centrality if centrality_data_available else None,
                                       vmax=max_centrality if centrality_data_available else None,
                                       alpha=0.95,  # Make nodes slightly more opaque
                                       linewidths=0.5,  # Optional: add edge to nodes
                                       edgecolors='k'  # Optional: black edge color
                                       )

        # Draw edges (only between nodes that have positions)
        valid_intra_edges = [(u, v) for u, v in intra_edges if u in pos and v in pos]
        nx.draw_networkx_edges(G_conn, pos, ax=ax, edgelist=valid_intra_edges,
                               alpha=0.4, width=0.6, edge_color='gray', style='dashed')

        # Draw edges - Inter-farm (thicker, blue, solid)
        valid_inter_edges = [(u, v) for u, v in inter_edges if u in pos and v in pos]
        nx.draw_networkx_edges(G_conn, pos, ax=ax, edgelist=valid_inter_edges,
                               alpha=0.6, width=1.0, edge_color='blue', style='solid')

        connectivity_handles = []
        if valid_intra_edges:  # Only add legend if edges were drawn
            connectivity_handles.append(
                mlines.Line2D([0], [0], color='gray', linestyle='dashed', linewidth=0.6, alpha=0.4,
                              label='Intra-Farm Connection'))
        if valid_inter_edges:  # Only add legend if edges were drawn
            connectivity_handles.append(
                mlines.Line2D([0], [0], color='blue', linestyle='solid', linewidth=1.0, alpha=0.6,
                              label='Inter-Farm Connection'))

        colorbar_handle = None
        if nodes is not None and centrality_data_available:
            try:
                norm = plt.Normalize(vmin=min_centrality, vmax=max_centrality)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                # Adjust colorbar size and padding
                cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=15, pad=0.03)
                cbar.set_label('Betweenness Centrality', rotation=270, labelpad=12, fontsize=25)
                cbar.ax.tick_params(labelsize=25)
                colorbar_handle = cbar  # Store handle if needed, though usually not directly manipulated
            except Exception as cb_err:
                print(f"Warning: Could not create colorbar. Error: {cb_err}")
                nodes = None  # Indicate nodes weren't effectively colored for legend purposes
        else:
            nodes = None  # Indicate no nodes drawn for colorbar or no data to map

        # 7. Finalize Plot Aesthetics
        ax.set_aspect('equal', 'box')
        ax.autoscale_view()
        ax.set_xlabel("Longitude / Easting", fontsize=25)
        ax.set_ylabel("Latitude / Northing", fontsize=25)
        all_handles = base_handles + connectivity_handles
        if all_handles:
            ax.legend(handles=all_handles, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=25)

    def plot_aggregate_hub_vs_nonhub_characteristics(combined_hub_details_df, output_dir):
        plot_filename = "aggregate_hub_vs_nonhub_comparison.svg"
        plot_filepath = os.path.join(output_dir, plot_filename)

        characteristics_to_plot = [
            'yield',
            'area_ha',
            'perimeter_km',
            'num_sides',
            'num_neighbors',
            'dist_habitat'
        ]

        # Filter only columns that actually exist in the combined dataframe
        valid_characteristics = [col for col in characteristics_to_plot if col in combined_hub_details_df.columns]
        if not valid_characteristics:
            print(f"Skipping plot: None of the requested characteristic columns found in combined DataFrame.")
            return

        # Determine subplot layout
        n_charts = len(valid_characteristics)
        ncols = 3
        nrows = int(np.ceil(n_charts / ncols))

        # Prepare data: replace Inf distance, drop rows where ALL characteristics are NaN
        plot_df = combined_hub_details_df.dropna(subset=valid_characteristics, how='all').copy()
        if 'dist_habitat' in plot_df.columns:
            plot_df['dist_habitat'] = plot_df['dist_habitat'].replace([np.inf, -np.inf], np.nan)

        # Final check after potential NaN dropping
        if plot_df.empty or plot_df['is_hub'].nunique() < 2:
            print(f"Skipping plot: Not enough valid data or hub variation after filtering NaNs/Infs.")
            return

        # --- Plotting ---
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4.5),
                                 squeeze=False)  # Increased height slightly
        axes = axes.flatten()  # Flatten axes array for easy iteration

        for i, char in enumerate(valid_characteristics):
            ax = axes[i]
            # Check if column has non-NA data for both hub categories IN AGGREGATE
            if plot_df.groupby('is_hub')[char].count().min() > 0:
                # Consider removing outliers for better visualization of the boxes if needed
                sns.boxplot(data=plot_df, x='is_hub', y=char, ax=ax, palette="Set2",
                            showfliers=True)  # showfliers=False to hide outliers
                ax.set_title(f'{char.replace("_", " ").title()}', fontsize=12)
                ax.set_xticklabels(['Non-Hub', 'Hub'])
                ax.set_ylabel('')  # Y-label is clear from title
                ax.set_xlabel('')  # Remove redundant x-label
                # Optional: Add log scale for skewed data like area, perimeter, distance
                if char in ['area_ha', 'perimeter_km', 'dist_habitat']:
                    # Check for non-positive values before applying log scale
                    if (plot_df[char] > 0).all():
                        ax.set_yscale('log')
                        ax.set_ylabel('Log Scale')  # Indicate log scale
                    else:
                        # Cannot use log scale if zeros or negatives are present
                        pass  # Or add annotation

            else:
                ax.text(0.5, 0.5, 'Insufficient data\nfor comparison', ha='center', va='center', fontsize=10,
                        color='gray')
                ax.set_title(f'{char.replace("_", " ").title()}', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plot_df = combined_hub_details_df.copy()
        # fig.suptitle(f'Aggregate Comparison: Hub vs. Non-Hub Plot Characteristics (All Runs)', fontsize=16,
        #             y=1.03)  # Adjusted y
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust layout
        plt.savefig(plot_filepath, bbox_inches='tight')
        plt.close(fig)

        # --- Plot 1: Compare Hubs vs Non-Hubs on % Inter-Farm Weight ---
        plot_filename = "aggregate_hub_vs_nonhub_percent_inter.svg"
        # Check if 'is_hub' column exists and has variation
        if 'is_hub' in plot_df.columns and plot_df[
            'is_hub'].nunique() > 1 and 'percent_inter_farm_weight' in plot_df.columns:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.boxplot(data=plot_df, x='is_hub', y='percent_inter_farm_weight', palette="Set2")
            ax.set_xlabel('Plot Type', fontsize=25)
            ax.set_ylabel('Percent of Connection Weight from Inter-Farm Links (%)', fontsize=25)
            plt.xticks([0, 1], ['Non-Hub', 'Hub'])
            ax.tick_params(axis='both', which='major', labelsize=25)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, plot_filename))
            plt.close()
            print(f"Saved: {plot_filename}")
        else:
            print(f"Skipping {plot_filename}: requires 'is_hub' variation and 'percent_inter_farm_weight' column.")

        # --- Plot 2: Compare Characteristics based on Inter-Farm Dominance ---
        # Define a category: plots where > 50% of connection weight is inter-farm
        if 'percent_inter_farm_weight' in plot_df.columns:
            plot_df['connectivity_focus'] = np.where(
                plot_df['percent_inter_farm_weight'] > 50,
                'Inter-Farm',
                'Intra-Farm'
            )

            characteristics_to_compare = [
                'area_ha',
                'perimeter_km',
                'sum_betweenness_centrality',
            ]
            # Filter only columns that actually exist in the combined dataframe
            valid_characteristics = [col for col in characteristics_to_compare if col in plot_df.columns]

            if not valid_characteristics or plot_df['connectivity_focus'].nunique() < 2:
                print(
                    "Skipping characteristic comparison by inter/intra dominance: Not enough valid data or variation.")
            else:
                n_charts = len(valid_characteristics)
                ncols = 3
                nrows = int(np.ceil(n_charts / ncols))

                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4.5), squeeze=False)
                axes = axes.flatten()

                for i, char in enumerate(valid_characteristics):
                    ax = axes[i]
                    # Ensure there's data for both categories for this characteristic
                    if plot_df.groupby('connectivity_focus')[char].count().min() > 0:
                        # Replace Inf distance before plotting
                        if char == 'dist_habitat':
                            plot_char_df = plot_df.replace({'dist_habitat': {np.inf: np.nan}}).dropna(subset=[char])
                        else:
                            plot_char_df = plot_df.dropna(subset=[char])

                        if plot_char_df.empty or plot_char_df['connectivity_focus'].nunique() < 2:
                            ax.text(0.5, 0.5, f'Insufficient valid data\nfor {char}', ha='center', va='center',
                                    fontsize=10, color='gray')
                        else:
                            sns.boxplot(data=plot_char_df, x='connectivity_focus', y=char, ax=ax, palette="coolwarm",
                                        showfliers=False)  # Hide outliers for clarity
                            ax.set_title(f'{char.replace("_", " ").title()}', fontsize=12)
                            ax.set_xlabel('')
                            ax.set_ylabel('')
                            ax.tick_params(axis='x', labelsize=25)  # Rotate labels slightly

                            # Optional: Log scale for highly skewed data
                            if char in ['area_ha', 'perimeter_km', 'dist_habitat', 'sum_betweenness_centrality']:
                                # Check for non-positive values before applying log scale, ignore if issue
                                if (plot_char_df[char] > 0).all():
                                    try:
                                        ax.set_yscale('log')
                                        ax.set_ylabel('Log Scale')  # Indicate log scale
                                    except ValueError:  # Handle potential issues with log scale
                                        ax.set_yscale('linear')  # Revert if log scale fails
                                        ax.set_ylabel('')
                                else:
                                    ax.set_ylabel('')  # Indicate linear scale

                    else:
                        ax.text(0.5, 0.5, 'Insufficient data\nfor comparison', ha='center', va='center', fontsize=10,
                                color='gray')
                        ax.set_title(f'{char.replace("_", " ").title()}', fontsize=12)
                        ax.set_xticks([])
                        ax.set_yticks([])

                # Hide unused subplots
                for j in range(i + 1, len(axes)):
                    axes[j].set_visible(False)

                # fig.suptitle('Plot Characteristics: Inter-Farm vs. Intra-Farm Connectivity Focus', fontsize=16, y=1.02)
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                plot_filename_chars = "aggregate_chars_by_inter_intra_focus.svg"
                plt.savefig(os.path.join(output_dir, plot_filename_chars), bbox_inches='tight')
                plt.close(fig)
                print(f"Saved: {plot_filename_chars}")
        else:
            print("Skipping characteristic comparison: 'percent_inter_farm_weight' column missing.")

    def plot_inter_intra_summary(summary_df, output_dir, connectivity_metric_name='IIC'):
        # Calculate percentages if scores exist
        plot_df = summary_df.dropna(subset=['intra_farm_pairwise_score', 'inter_farm_pairwise_score']).copy()
        plot_df['total_pairwise'] = plot_df['intra_farm_pairwise_score'] + plot_df['inter_farm_pairwise_score']
        # Avoid division by zero if total is near zero
        plot_df['percent_inter'] = np.where(
            plot_df['total_pairwise'] > 1e-9,
            (plot_df['inter_farm_pairwise_score'] / plot_df['total_pairwise']) * 100,
            0  # Assign 0 if total is zero
        )

        if plot_df.empty:
            print("Skipping inter/intra summary plots: No valid data.")
            return

        # Plot 1: Distribution of Inter-Farm Contribution Percentage
        plot_filename = "aggregate_percent_inter_farm_distrib.svg"
        plt.figure(figsize=(8, 5))
        sns.histplot(plot_df['percent_inter'], kde=True, bins=15)
        plt.xlabel('Percentage of Pairwise Score from Inter-Farm Connections (%)')
        plt.ylabel('Frequency (Number of Runs)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()
        print(f"Saved: {plot_filename}")

        # Plot 2: Total Final Connectivity vs. Percentage Inter-Farm
        plot_filename = "aggregate_conn_vs_percent_inter.svg"
        plot_df_conn = plot_df.dropna(subset=['conn_val_final', 'percent_inter']).copy()  # Ensure conn_val_final exists
        if not plot_df_conn.empty:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=plot_df_conn, x='percent_inter', y='conn_val_final', alpha=0.7)
            sns.regplot(data=plot_df_conn, x='percent_inter', y='conn_val_final', scatter=False, color='red',
                        line_kws={'linewidth': 1})
            plt.xlabel('Percentage of Pairwise Score from Inter-Farm Connections (%)')
            plt.ylabel(f'Final Connectivity ({connectivity_metric_name})')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, plot_filename))
            plt.close()

    results_list = []
    all_runs_hub_details = []
    if farms == "syn_farms":
        output_dir = os.path.join(syn_farm_dir, "plots", "ec", "hubs")
    else:
        output_dir = os.path.join(syn_farm_dir, "plots", "ec", "hubs_real")
    os.makedirs(output_dir, exist_ok=True)
    load = False
    if not load:
        for config_id in num_configs:
            print(f"\n=========== Running config: {config_id} ===========")
            config_path = os.path.join(base_farm_dir, f"config_{config_id}")
            all_plots_geojson = os.path.join(config_path, f"all_plots_interventions_og.geojson")

            try:
                chosen_final, optim_val_final, conn_val_final, conn_val_repos, plots, _, _, _ = optimize_landscape_connectivity(
                    all_plots_geojson, boundary_seg_count,
                    interior_cell_count, adjacency_dist,
                    connectivity_metric, al_factor,
                    max_loss_ratio,
                    neib_dist, exit_tol, reposition, params,
                    config_path, margin_weight, mode, plot)
            except Exception as e:
                continue

            if not chosen_final:
                continue

            (hub_analysis_df, G_conn, node_centralities, intra_edges, inter_edges, intra_score, inter_score,
             plot_weights) = run_hubs(chosen_final, plots, adjacency_dist, top_n_percent=10)

            # Save hub analysis results
            hub_analysis_df.to_csv(os.path.join(config_path, f"hub_analysis.csv"), index=False)

            if not hub_analysis_df.empty:
                hub_analysis_df['config_id'] = config_id  # Add config_id for tracking
                all_runs_hub_details.append(hub_analysis_df.copy())

            fig_combined, ax_combined = plt.subplots(figsize=(12, 10))
            plot_farms_with_connectivity_graph(
                ax=ax_combined,  # Pass the axes
                plots=plots,
                chosen_pieces=chosen_final,
                G_conn=G_conn,
                node_centralities=node_centralities,
                intra_edges=intra_edges,
                inter_edges=inter_edges
            )
            output_filename_combined = f"combined_farm_connectivity_graph.svg"
            output_filepath_combined = os.path.join(config_path, output_filename_combined)
            plt.savefig(output_filepath_combined, bbox_inches='tight', dpi=300)
            plt.close(fig_combined)

            num_ag_plots = sum(1 for p in plots if p['plot_type'] == 'ag_plot')
            initial_fragmentation_metric = num_ag_plots
            num_hubs_identified_val = hub_analysis_df['is_hub'].sum() if not hub_analysis_df.empty else 0
            avg_hub_betweenness = 0
            if not hub_analysis_df.empty and hub_analysis_df['is_hub'].any():
                avg_hub_betweenness = hub_analysis_df.loc[
                    hub_analysis_df['is_hub'], 'sum_betweenness_centrality'].mean()
            connectivity_gain = conn_val_final - conn_val_repos

            total_landscape_area = sum(p['geometry'].area for p in plots if not p['geometry'].is_empty)
            initial_habitat_area = sum(
                p['geometry'].area for p in plots if p['plot_type'] == 'hab_plots' and not p['geometry'].is_empty)

            if total_landscape_area > 0:
                initial_habitat_ratio = initial_habitat_area / total_landscape_area
            else:
                initial_habitat_ratio = 0.0

            all_plot_geoms = [p['geometry'] for p in plots if p['geometry'] and not p['geometry'].is_empty]
            if not all_plot_geoms:
                # Handle case with no valid geometries
                landscape_boundary_geom = None
                existing_habitat_multipoly = None
            else:
                # Might be slow for many plots:
                landscape_boundary_geom = unary_union(all_plot_geoms).boundary
                existing_habitat_geoms = [p['geometry'] for p in plots if
                                          p['plot_type'] == 'hab_plots' and p['geometry'] and not p[
                                              'geometry'].is_empty]
                if existing_habitat_geoms:
                    existing_habitat_multipoly = MultiPolygon(existing_habitat_geoms)  # Faster distance calculation
                else:
                    existing_habitat_multipoly = None  # No existing habitats
            # --- End Pre-calculation ---

            avg_hub_dist_to_boundary = np.nan
            avg_hub_dist_to_habitat = np.nan

            if not hub_analysis_df.empty and hub_analysis_df['is_hub'].any():
                hub_plot_indices = hub_analysis_df[hub_analysis_df['is_hub']].index.tolist()
                hub_distances_to_boundary = []
                hub_distances_to_habitat = []

                if hub_plot_indices:  # Check if list is not empty
                    for plot_idx in hub_plot_indices:
                        # Ensure plot_idx is valid and geometry exists
                        if 0 <= plot_idx < len(plots) and plots[plot_idx]['geometry'] and not plots[plot_idx][
                            'geometry'].is_empty:
                            hub_plot_geom = plots[plot_idx]['geometry']
                            hub_centroid = hub_plot_geom.centroid
                            if hub_centroid.is_empty: continue  # Skip if centroid calculation fails

                            # Calculate distance to boundary
                            if landscape_boundary_geom:
                                dist_boundary = hub_centroid.distance(landscape_boundary_geom)
                                hub_distances_to_boundary.append(dist_boundary)

                            # Calculate distance to nearest existing habitat
                            if existing_habitat_multipoly:
                                dist_habitat = hub_centroid.distance(existing_habitat_multipoly)
                                hub_distances_to_habitat.append(dist_habitat)
                            else:
                                hub_distances_to_habitat.append(np.inf)  # Or handle as needed if no habitat

                    if hub_distances_to_boundary:
                        avg_hub_dist_to_boundary = np.mean(hub_distances_to_boundary)
                    if hub_distances_to_habitat and not np.isinf(hub_distances_to_habitat).all():
                        # Calculate mean excluding infinite distances if any
                        valid_habitat_distances = [d for d in hub_distances_to_habitat if np.isfinite(d)]
                        if valid_habitat_distances:
                            avg_hub_dist_to_habitat = np.mean(valid_habitat_distances)

                # Store results for overall summary
            results_list.append({
                'config_id': config_id,
                'conn_val_final': conn_val_final,
                'conn_val_repos': conn_val_repos,
                'optim_val_final': optim_val_final,
                'num_chosen_pieces': len(chosen_final),
                'num_hubs_identified': num_hubs_identified_val,
                'initial_fragmentation_metric': initial_fragmentation_metric,
                'avg_hub_betweenness': avg_hub_betweenness,
                'connectivity_gain': connectivity_gain,
                'initial_habitat_ratio': initial_habitat_ratio,
                'avg_hub_dist_to_boundary': avg_hub_dist_to_boundary,
                'avg_hub_dist_to_habitat': avg_hub_dist_to_habitat,
                'intra_farm_pairwise_score': intra_score if np.isfinite(intra_score) else None,
                'inter_farm_pairwise_score': inter_score if np.isfinite(inter_score) else None,
            })

        summary_df = pd.DataFrame(results_list)
        summary_df.to_csv(os.path.join(output_dir, f"optimization_summary.csv"), index=False)

        combined_hub_details_df = pd.concat(all_runs_hub_details, ignore_index=True)
        combined_hub_details_df.to_csv(os.path.join(output_dir, f"combined_hubs.csv"), index=False)
    else:
        summary_df = pd.read_csv(os.path.join(output_dir, f"optimization_summary.csv"))
        combined_hub_details_df = pd.read_csv(os.path.join(output_dir, f"combined_hubs.csv"))

    plot_hubs(summary_df, output_dir)
    plot_inter_intra_summary(summary_df, output_dir, connectivity_metric_name=connectivity_metric)
    plot_aggregate_hub_vs_nonhub_characteristics(
        combined_hub_details_df=combined_hub_details_df,
        output_dir=output_dir
    )


def calculate_marginal_connectivity_values():
    def run(chosen_pieces, adjacency_dist, connectivity_metric):
        if not chosen_pieces:
            return []

        # Calculate the baseline connectivity with all chosen pieces
        base_graph = build_connectivity_graph_from_chosen_pieces(chosen_pieces, adjacency_dist)
        base_conn_val = compute_connectivity_metric(base_graph, connectivity_metric)
        print(f"--- Calculating Marginal Connectivity Values (Base Score: {base_conn_val:.4f}) ---")

        marginal_values = []

        for i, piece_to_remove in enumerate(chosen_pieces):
            # Create a temporary list without the current piece
            temp_pieces = chosen_pieces[:i] + chosen_pieces[i + 1:]

            if not temp_pieces:
                temp_conn_val = 0.0
            else:
                # Build graph and calculate connectivity for the reduced set
                temp_graph = build_connectivity_graph_from_chosen_pieces(temp_pieces, adjacency_dist)
                temp_conn_val = compute_connectivity_metric(temp_graph, connectivity_metric)

            # The marginal value is the loss in connectivity when this piece is removed
            marginal_value = base_conn_val - temp_conn_val

            # Store results - use WKT for geometry representation as the object itself is complex
            geom_wkt = shapely_wkt.dumps(piece_to_remove['geom'])
            marginal_values.append({
                'piece_index': i,  # Index within the original chosen_pieces list
                'piece_type': piece_to_remove['type'],
                'geom_wkt': geom_wkt,  # For identification
                'marginal_value': marginal_value
            })
            # Optional: Print progress
            # print(f"  Processed piece {i+1}/{len(chosen_pieces)}, Marginal Value: {marginal_value:.4f}")

        # Sort by marginal value (descending) to find the most important pieces
        marginal_values.sort(key=lambda x: x['marginal_value'], reverse=True)

        print("--- Marginal Connectivity Calculation Complete ---")
        return marginal_values

    def plot_marginal_value_distribution(data, output_dir, filename="marginal_distribution.png"):
        plt.figure(figsize=(12, 6))
        sns.histplot(data=data, x='marginal_value', hue='piece_type', kde=True, bins=50)
        plt.title('Distribution of Marginal Connectivity Values')
        plt.xlabel('Marginal Connectivity Value (Score Decrease if Removed)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    def plot_marginal_value_boxplot(data, output_dir, filename="marginal_boxplot.png"):
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=data, x='piece_type', y='marginal_value', showfliers=False)  # showfliers=False hides outliers
        # Or use violinplot: sns.violinplot(data=data, x='piece_type', y='marginal_value')
        plt.title('Comparison of Marginal Connectivity Values by Piece Type')
        plt.xlabel('Piece Type')
        plt.ylabel('Marginal Connectivity Value')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    all_marginal_data = []
    output_dir = os.path.join(syn_farm_dir, "plots", "ec", "marginal_conn")
    os.makedirs(output_dir, exist_ok=True)

    for config_id in num_configs:  # Assuming configs are numbered 0, 1, 2...
        config_path = os.path.join(base_farm_dir, f"config_{config_id}")
        all_plots_geojson = os.path.join(config_path, f"all_plots_interventions_og.geojson")
        interventions_filename = os.path.join(config_path, f"connectivity_interventions_{mode}.json")
        chosen_final, optim_val_final, conn_val_final, _, plots, _, _ = optimize_landscape_connectivity(all_plots_geojson,
                                                                                                  boundary_seg_count,
                                                                                                  interior_cell_count,
                                                                                                  adjacency_dist,
                                                                                                  connectivity_metric,
                                                                                                  al_factor,
                                                                                                  max_loss_ratio,
                                                                                                  neib_dist, exit_tol,
                                                                                                  reposition,
                                                                                                  params, config_path,
                                                                                                  margin_weight,
                                                                                                  mode, plot)

        marginal_values = run(chosen_final, adjacency_dist, connectivity_metric)
        for piece_data in marginal_values:
            piece_data['config_id'] = config_id
            if 'geom_wkt' in piece_data:
                try:
                    piece_data['geometry'] = shapely_wkt.loads(piece_data['geom_wkt'])
                except Exception as e:
                    print(
                        f"Warning: Could not parse WKT for piece {piece_data.get('piece_index', '')} in config {config_id}: {e}")
                    piece_data['geometry'] = None
            else:
                piece_data['geometry'] = None  # Placeholder

            all_marginal_data.append(piece_data)

    df = pd.DataFrame(all_marginal_data)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf['area'] = gdf.geometry.area
    gdf['length'] = gdf.geometry.length
    gdf['size'] = np.where(gdf['piece_type'] == 'margin', gdf['length'], gdf['area'])

    plot_marginal_value_distribution(gdf, output_dir)
    plot_marginal_value_boxplot(gdf, output_dir)


def run_max_loss_ratio():
    max_loss_ratios_to_test = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    results_mean_connectivity = []
    results_std_connectivity = []
    for loss_ratio in max_loss_ratios_to_test:
        print(f"Running loss_ratio: {loss_ratio}")
        connectivity_values_for_this_ratio = []
        for config_id in num_configs:
            print(f"Running config: {config_id}")
            config_path = os.path.join(base_farm_dir, f"config_{config_id}")
            all_plots_geojson = os.path.join(config_path, f"all_plots_interventions_og.geojson")
            try:
                _, _, conn_val_final, conn_val_repos, _, _, _, _ = optimize_landscape_connectivity(all_plots_geojson,
                                                                                          boundary_seg_count,
                                                                                          interior_cell_count,
                                                                                          adjacency_dist,
                                                                                          connectivity_metric,
                                                                                          al_factor,
                                                                                          loss_ratio,
                                                                                          neib_dist, exit_tol,
                                                                                          reposition,
                                                                                          params, config_path,
                                                                                          margin_weight,
                                                                                          mode, plot)
            except Exception as e:
                continue
            if conn_val_final is not None:
                connectivity_values_for_this_ratio.append(conn_val_final)
        mean_conn = np.mean(connectivity_values_for_this_ratio)
        std_conn = np.std(connectivity_values_for_this_ratio)
        results_mean_connectivity.append(mean_conn)
        results_std_connectivity.append(std_conn)

    valid_indices = ~np.isnan(results_mean_connectivity)
    filtered_loss_ratios = np.array(max_loss_ratios_to_test)[valid_indices]
    filtered_mean_connectivity = np.array(results_mean_connectivity)[valid_indices]
    filtered_std_connectivity = np.array(results_std_connectivity)[valid_indices]

    plt.figure(figsize=(10, 7))

    # Plot the mean line
    plt.plot(filtered_loss_ratios, filtered_mean_connectivity, marker='o', linestyle='-', label='Mean Connectivity')

    # Add shaded area for standard deviation (optional)
    plt.fill_between(filtered_loss_ratios,
                     filtered_mean_connectivity - filtered_std_connectivity,
                     filtered_mean_connectivity + filtered_std_connectivity,
                     alpha=0.2, label='Standard Deviation')

    plt.xlabel("Maximum Allowed NPV Loss Ratio")
    plt.ylabel(f"Mean Landscape Connectivity ({connectivity_metric})")
    # plt.title(f"Mean Connectivity vs. Economic Constraint (Avg. over {len(config_ids_to_process)} Configs)")
    plt.legend()
    plt.xticks(max_loss_ratios_to_test)  # Ensure all tested ratios are marked

    # Save the plot
    if farms == "syn_farms":
        output_dir = os.path.join(syn_farm_dir, "plots", "ec", "max_loss")
    else:
        output_dir = os.path.join(syn_farm_dir, "plots", "ec", "max_loss_real")
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f"tradeoff_curve_config.svg")
    plt.savefig(plot_filename)


def analyze_economic_impacts():
    def plot_aggregate_economic_impacts(aggregated_plot_df, results_df, base_output_dir):
        def gini(x):
            x = np.asarray(x)
            if np.amin(x) < 0:
                x_shifted = x - np.amin(x)
                print("Warning: Negative values detected in Gini input. Shifting data to non-negative for calculation.")
            else:
                x_shifted = x
            if np.any(x_shifted == 0):
                x_shifted = x_shifted + 1e-9  # Add small epsilon

            # Values must be sorted:
            x_sorted = np.sort(x_shifted)
            n = x_sorted.shape[0]
            if n == 0 or np.sum(x_sorted) == 0:  # Handle empty or all-zero array
                return 0.0  # Or np.nan?
            index = np.arange(1, n + 1)
            return ((np.sum((2 * index - n - 1) * x_sorted)) / (n * np.sum(x_sorted)))

        num_configs_analyzed = aggregated_plot_df['config_id'].nunique()
        num_plots_total = len(aggregated_plot_df)
        print(
            f"Analyzing aggregated data from {num_configs_analyzed} configurations, {num_plots_total} total plot instances.")


        # 2. Distribution Plot (Histogram) of Plot Relative NPV Changes (Pooled)
        plt.figure(figsize=(10, 6))
        plot_rel_change_finite = aggregated_plot_df['relative_npv_change'].replace([np.inf, -np.inf], np.nan).dropna()
        sns.histplot(plot_rel_change_finite, bins=50, kde=True)  # More bins for potentially more data
        plt.xlabel('Relative NPV Change')
        plt.ylabel('Number of Plot Instances')
        plt.tight_layout()
        plt.savefig(os.path.join(base_output_dir, f"aggregate_plot_relative_npv_change_hist.svg"))
        plt.close()

        # 3. Variance/Inequality Metrics (on Pooled Data)
        # Use finite values for variance calculation
        plot_npv_change_variance = plot_rel_change_finite.var()
        print(f"\nVariance of Plot Relative NPV Change (Pooled): {plot_npv_change_variance:.4f}")

        # Gini coefficient calculation (optional) on absolute changes
        try:
            # Ensure input to gini is finite
            abs_change_finite = np.abs(aggregated_plot_df['npv_change'].replace([np.inf, -np.inf], np.nan).dropna())
            plot_gini = gini(abs_change_finite)
            print(f"Gini Coefficient of Absolute Plot NPV Change (Pooled): {plot_gini:.4f}")
        except Exception as e:
            print(f"Could not calculate Gini coefficient on aggregated data: {e}")

        aggregated_plot_df = aggregated_plot_df.replace([np.inf, -np.inf], np.nan).copy()
        aggregated_plot_df = aggregated_plot_df.dropna(subset=['relative_npv_change'])

        y_var = 'relative_npv_change'
        x_vars = ['area', 'perimeter', 'yield', 'num_sides', 'num_neighbors']
        plot_titles = {
            'area': 'Plot Area (ha)',
            'perimeter': 'Plot Perimeter (km)',
            'yield': 'Yield',  # Changed label slightly
            'num_sides': 'Number of Sides',
            'num_neighbors': 'Number of Touching Neighbors'  # Updated based on new calc
        }

        # Determine common y-axis limits based on the overall selected data (sample or full finite)
        y_lower = aggregated_plot_df[y_var].quantile(0.01)
        y_upper = aggregated_plot_df[y_var].quantile(0.99)
        if y_upper == y_lower:  # Add buffer
            buffer = max(abs(y_lower * 0.01), 1e-6)
            y_upper += buffer
            y_lower -= buffer
        y_limits = (y_lower, y_upper)
        print(f"Setting panel Y-limits based on sample/finite data: {y_limits}")

        # --- Create Subplots ---
        num_plots = len(x_vars)
        # Decide layout (e.g., 2 rows for 5 plots needs 3 columns)
        ncols = 3
        nrows = math.ceil(num_plots / ncols)  # Calculate needed rows
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4.5),
                                 sharey=True)  # Create figure and axes grid, share Y axis
        axes = axes.flatten()  # Flatten axes array for easy 1D iteration

        plot_index = 0  # To track which axis to use

        for x_var in x_vars:
            if plot_index >= len(axes):
                print("Warning: More plots than axes slots available.")
                break  # Stop if we run out of axes

            current_ax = axes[plot_index]  # Get the specific axis for this plot

            # Check if the x-variable exists
            if x_var not in aggregated_plot_df.columns:
                print(f"Warning: Column '{x_var}' not found in data. Skipping subplot.")
                current_ax.axis('off')  # Turn off this unused axis explicitly
                # Do NOT increment plot_index here, keep current_ax for next potential plot? NO, axis is used.
                # Instead, maybe just skip the plotting for this x_var but keep the axis visible with a note?
                # Let's turn it off and continue the outer loop.
                continue  # Skip this x_var

            # Prepare data specifically for this subplot
            # Filter NaNs and invalid values (-1) in the current x_var from the sample/finite data
            plot_data_for_subplot = aggregated_plot_df.dropna(subset=[x_var])
            if x_var in ['num_sides', 'num_neighbors']:
                plot_data_for_subplot = plot_data_for_subplot[plot_data_for_subplot[x_var] >= 0]

            if plot_data_for_subplot.empty:
                print(f"Warning: No valid data points found for subplot with x='{x_var}'. Skipping.")
                current_ax.axis('off')  # Turn off this unused axis
                plot_index += 1  # Move to the next axis slot for the next iteration
                continue  # Skip this x_var

            print(f"Generating subplot: Relative NPV Change vs {plot_titles.get(x_var, x_var)}...")
            try:
                # Use regplot targeting the specific axis `ax=current_ax`
                sns.regplot(ax=current_ax, data=plot_data_for_subplot, x=x_var, y=y_var,
                            scatter_kws={'alpha': 0.3, 's': 10},  # Smaller points for potentially dense plots
                            line_kws={'color': 'red', 'linewidth': 1.5})

                current_ax.set_title(f'{plot_titles.get(x_var, x_var)}')
                current_ax.set_xlabel(plot_titles.get(x_var, x_var))

                # Only add y-label to the first plot in each row if sharey=True
                if plot_index % ncols == 0:
                    current_ax.set_ylabel('Relative NPV Change')
                else:
                    current_ax.set_ylabel('')  # Remove label for inner plots

                # Apply common y-limits calculated earlier
                if y_limits:
                    current_ax.set_ylim(y_limits)

                #current_ax.grid(linestyle='--', alpha=0.7)
                plot_index += 1  # Increment axis index only if plot was successful

            except Exception as e:
                print(f"Error generating subplot for x='{x_var}': {e}")
                current_ax.set_title(f"Error plotting {x_var}")  # Indicate error on plot
                plot_index += 1  # Move to next axis even if plot failed

        # --- Clean up any remaining unused axes ---
        for i in range(plot_index, len(axes)):
            axes[i].axis('off')

        # --- Final Figure Adjustments ---
        #fig.suptitle(f'Relative NPV Change vs Plot Characteristics ({data_source_label})', fontsize=16, y=1.02)
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust layout

        # Save the entire figure containing all subplots
        panel_filename = os.path.join(base_output_dir,
                                      f"aggregate_scatter_panel_RelNPVchange.svg")  # Use SVG for better quality
        try:
            plt.savefig(panel_filename)
            print(f"Scatter plot panel saved to: {panel_filename}")
        except Exception as e:
            print(f"Error saving scatter plot panel: {e}")
        finally:
            plt.close(fig)

        plot_summary_df = results_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=['relative_npv_change', 'connectivity_final']  # Ensure both x and y are finite
        )

        if not plot_summary_df.empty:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=plot_summary_df, x='relative_npv_change', y='connectivity_final',
                            legend='auto')  # Example using hue/size

            #plt.title(f'Final Landscape Connectivity ({connectivity_metric}) vs. Overall Relative NPV Change')
            plt.xlabel('Relative NPV Change (per Configuration)')
            plt.ylabel(f'Final Connectivity ({connectivity_metric})')

            plt.axvline(0, color='grey', linestyle=':', lw=1)
            plt.legend()

            plt.tight_layout()

            # Save the plot to the base directory
            plot_filename = os.path.join(output_dir,
                                         f"aggregate_scatter_{connectivity_metric}_vs_RelNPVchange.svg")
            try:
                plt.savefig(plot_filename)
                print(f"Connectivity vs NPV scatter plot saved to: {plot_filename}")
            except Exception as e:
                print(f"Error saving scatter plot: {e}")
            plt.close()

    all_config_summaries = []
    all_plot_impact_data = []
    if farms == "syn_farms":
        output_dir = os.path.join(syn_farm_dir, "plots", "ec", "plot_npv")
    else:
        output_dir = os.path.join(syn_farm_dir, "plots", "ec", "plot_npv_real")
    os.makedirs(output_dir, exist_ok=True)
    load = True
    if not load:
        for config_id in num_configs:
            print(f"Running config: {config_id}")
            config_path = os.path.join(base_farm_dir, f"config_{config_id}")
            all_plots_geojson = os.path.join(config_path, f"all_plots_interventions_og.geojson")
            try:
                chosen_final, _, conn_val_final, conn_val_repos, plots, plot_baseline_npv, optimized_plot_npvs, _ = optimize_landscape_connectivity(
                    all_plots_geojson,
                    boundary_seg_count,
                    interior_cell_count,
                    adjacency_dist,
                    connectivity_metric, al_factor,
                    max_loss_ratio,
                    neib_dist, exit_tol, reposition,
                    params, config_path,
                    margin_weight,
                    mode, plot)
            except Exception as e:
                continue

            neighbors_map = calculate_touching_plot_neighbors(plots)

            plot_impacts = []
            farm_npvs = defaultdict(lambda: {'baseline': 0.0, 'optimized': 0.0})
            epsilon = 1e-6  # For safe division

            for idx, p in enumerate(plots):
                num_sides = np.nan
                geom = p.get('geometry')
                if geom and not geom.is_empty:
                    if geom.geom_type == 'Polygon' and geom.exterior:
                        coords = list(geom.exterior.coords)
                        if len(coords) > 1:
                            num_sides = len(coords) - 1
                num_neighbors = neighbors_map.get(idx, 0)

                if p['plot_type'] == 'ag_plot':
                    baseline = plot_baseline_npv.get(idx, 0.0)
                    optimized = optimized_plot_npvs.get(idx, 0.0)
                    change = optimized - baseline
                    if abs(baseline) < epsilon:
                        relative_change = change / epsilon if abs(change) > epsilon else 0.0
                        relative_change = np.inf if change > 0 else (
                            -np.inf if change < 0 else 0.0)  # More explicit inf handling
                    else:
                        relative_change = change / baseline

                    plot_impacts.append({
                        'config_id': config_id,  # Add config identifier
                        'plot_index': idx,
                        'farm_id': p['farm_id'],
                        'plot_id_orig': p.get('plot_id', idx),  # Use original plot ID if available
                        'baseline_npv': baseline,
                        'optimized_npv': optimized,
                        'npv_change': change,
                        'relative_npv_change': relative_change,
                        'area': p['geometry'].area / 10000 if p['geometry'] else 0,  # ha
                        'perimeter': p['geometry'].length / 1000 if p['geometry'] else 0,  # km
                        'yield': p.get('yield', 0.0),
                        'label': p.get('label', ''),
                        'num_sides': num_sides,
                        'num_neighbors': num_neighbors
                    })
                    # Aggregate farm NPVs for summary
                    farm_npvs[p['farm_id']]['baseline'] += baseline
                    farm_npvs[p['farm_id']]['optimized'] += optimized

            # Create DataFrame for this config's plot impacts and add to list
            if plot_impacts:
                current_config_plot_df = pd.DataFrame(plot_impacts)
                all_plot_impact_data.append(current_config_plot_df)

            # Store summary results for this configuration
            config_summary = {
                'config_id': config_id,
                'status': 'Success',
                'loss_ratio': max_loss_ratio,
                'connectivity_final': conn_val_final,
                'connectivity_reposition': conn_val_repos,
                'total_baseline_npv': sum(f['baseline'] for f in farm_npvs.values()),
                'total_optimized_npv': sum(f['optimized'] for f in farm_npvs.values()),
                'num_chosen_pieces': len(chosen_final) if chosen_final else 0
            }
            config_summary['relative_npv_change'] = (config_summary['total_optimized_npv'] -
                                                     config_summary['total_baseline_npv']) / (config_summary['total_baseline_npv'] + epsilon)
            all_config_summaries.append(config_summary)

        results_df = pd.DataFrame(all_config_summaries)
        summary_file = os.path.join(output_dir, f"plot_npv_configs.csv")
        results_df.to_csv(summary_file, index=False)

        aggregated_plot_df = pd.concat(all_plot_impact_data, ignore_index=True)
        aggregated_plot_df.to_csv(os.path.join(output_dir, f"aggregate_plot_economic_impact.csv"),
                                  index=False)
    else:
        results_df = pd.read_csv(os.path.join(output_dir, f"plot_npv_configs.csv"))
        aggregated_plot_df = pd.read_csv(os.path.join(output_dir, f"aggregate_plot_economic_impact.csv"))

    plot_aggregate_economic_impacts(aggregated_plot_df, results_df, output_dir)



if __name__ == "__main__":
    cfg = Config()

    boundary_seg_count = 4
    interior_cell_count = 4
    adjacency_dist = 0.0
    connectivity_metric = 'IIC'
    al_factor = 1e-9  # 1e-9
    max_loss_ratio = 0.2
    params = cfg.params
    neib_dist = 1000
    exit_tol = 1e-6
    reposition = True
    margin_weight = 50
    mode = "og"
    farms = "real_farms" #syn_farms, real_farms
    plot = True

    syn_farm_dir = os.path.join(cfg.disk_dir, "crop_inventory", "syn_farms")
    if farms == "syn_farms":
        base_farm_dir = os.path.join(syn_farm_dir, "mc")
        num_configs = np.arange(1, 501)
    else:
        base_farm_dir = os.path.join(cfg.disk_dir, "crop_inventory", "farms_config_s")
        num_configs = np.arange(1, 571)

    # run_all_configs()
    # analyze_repositioning_results()
    # analyze_connectivity_hubs()
    # calculate_marginal_connectivity_values()
    analyze_economic_impacts()
    # run_max_loss_ratio()
