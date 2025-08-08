import os.path
from utils.tools import *
import pandas as pd
from ei_ec.config import Config
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import geopandas as gpd
from shapely.geometry import LineString, box, Polygon
from shapely.ops import substring
import numpy as np
import os
from shapely import wkt
import json
from shapely.ops import unary_union
from shapely.geometry import shape
import math
from shapely.strtree import STRtree


def plot_directions():
    def plot_polygon(ax, poly, fill=False, color=None):
        """
        Plots a shapely Polygon or MultiPolygon.
         - If fill=True, fills the polygon with the given color.
         - Otherwise, draws the polygon boundary with the given color.
        """
        if poly.is_empty:
            return

        # If it’s a simple Polygon
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            if fill:
                ax.fill(x, y, color=color)
            else:
                ax.plot(x, y, color=color)
            # Plot any interior holes (rings)
            for ring in poly.interiors:
                rx, ry = ring.xy
                ax.plot(rx, ry, color=color)

        # If it’s a MultiPolygon
        elif poly.geom_type == 'MultiPolygon':
            for single_poly in poly.geoms:
                plot_polygon(ax, single_poly, fill=fill, color=color)

    def get_quadrant_polygon(plot_poly: Polygon, direction: str) -> Polygon:
        """
        Returns the bounding box quadrant polygon (north-west, north-east,
        south-west, south-east) intersected with the original plot polygon.
        """
        minx, miny, maxx, maxy = plot_poly.bounds
        x_center = (minx + maxx) / 2.0
        y_center = (miny + maxy) / 2.0

        if direction == "north-west":
            box = Polygon([
                (minx, y_center), (x_center, y_center),
                (x_center, maxy), (minx, maxy)
            ])
        elif direction == "north-east":
            box = Polygon([
                (x_center, y_center), (maxx, y_center),
                (maxx, maxy), (x_center, maxy)
            ])
        elif direction == "south-west":
            box = Polygon([
                (minx, miny), (x_center, miny),
                (x_center, y_center), (minx, y_center)
            ])
        elif direction == "south-east":
            box = Polygon([
                (x_center, miny), (maxx, miny),
                (maxx, y_center), (x_center, y_center)
            ])
        else:
            # If it's not one of the four quadrants,
            # return an empty polygon
            return Polygon()

        return box.intersection(plot_poly)

    def plot_line(ax, line, color=None):
        """
        Plots a shapely LineString or MultiLineString in the given color.
        """
        if line.is_empty:
            return

        if line.geom_type == 'LineString':
            x, y = line.xy
            ax.plot(x, y, color=color)
        elif line.geom_type == 'MultiLineString':
            for ls in line.geoms:
                x, y = ls.xy
                ax.plot(x, y, color=color)

    conn_dir = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms", "farm_" + str(farm_id), "connectivity")
    heur_src_path = os.path.join(conn_dir, "generations", "best_heuristics_gem_gen_0.py")
    heur_dst_path = os.path.join(conn_dir, "best_heuristics_gem.py")
    shutil.copyfile(heur_src_path, heur_dst_path)

    input_path = os.path.join(conn_dir, "heuristics", "input.geojson")
    input_cp_path = os.path.join(conn_dir, "heuristics", "input_cp.geojson")
    shutil.copyfile(input_path, input_cp_path)

    os.chdir(conn_dir)
    _, _, _ = capture.run_python_script(heur_dst_path)

    output_path = os.path.join(conn_dir, "output.json")
    if not os.path.exists(output_path):
        print(f"No output for heuristic")
    else:
        gdf_input = gpd.read_file(input_cp_path)
        gdf_input['id'] = gdf_input['id'].astype(str)

        with open(output_path, 'r') as f:
            out_data = json.load(f)
        df_out = pd.DataFrame(out_data)
        if 'plot_id' in df_out.columns and 'id' not in df_out.columns:
            df_out.rename(columns={'plot_id': 'id'}, inplace=True)
        df_out['id'] = df_out['id'].astype(str)

        gdf_merged = gdf_input.merge(
            df_out[['id', 'margin_directions', 'habitat_directions']],
            on='id',
            how='left'
        )
        gdf_merged['margin_directions'] = gdf_merged['margin_directions'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        gdf_merged['habitat_directions'] = gdf_merged['habitat_directions'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        gdf_merged = gdf_merged.fillna(0)
        fig, ax = plt.subplots()

        for idx, row in gdf_merged.iterrows():
            plot_poly = row["geometry"]
            plot_polygon(ax, plot_poly, fill=False, color="grey")

            # 2) Margin directions are edges in those quadrants
            margin_directions = row["margin_directions"]
            for direction in margin_directions:
                # Get the quadrant polygon
                qpoly = get_quadrant_polygon(plot_poly, direction)
                # Intersect the plot boundary with that quadrant region
                boundary = plot_poly.boundary.intersection(qpoly)
                plot_line(ax, boundary, color='red')  # draws line with default settings

            # 3) Habitat directions are sub‐polygon areas in those quadrants
            habitat_directions = row["habitat_directions"]
            for direction in habitat_directions:
                qpoly = get_quadrant_polygon(plot_poly, direction)
                plot_polygon(ax, qpoly, fill=True, color='green')  # fill the sub‐quadrant area

        ax.set_aspect('equal', 'box')
        # Save the figure
        plt.savefig(os.path.join(conn_dir, "output_pred_directions.png"), dpi=300)
        plt.close(fig)

        
def partial_boundary(polygon, fraction):
    """
    Return a (single) continuous segment of the polygon's exterior
    that corresponds to `fraction` of the boundary length.
    Example: fraction=0.3 -> ~30% of the perimeter from the start.
    """
    if fraction <= 0:
        return None
    if fraction >= 1:
        # Entire boundary
        return polygon.exterior

    boundary = polygon.exterior
    length = boundary.length
    # Use shapely.ops.substring to get fraction of perimeter
    # substring(linestring, start_distance, end_distance, normalized=False)
    partial_line = substring(boundary, 0, fraction * length)
    return partial_line


def partial_polygon_left_to_right(polygon, fraction):
    """
    Return a (single) continuous sub-area by cutting from left to right
    until the area covers `fraction` of the original polygon's area.

    This is just one simple approach: we do a binary search
    on a vertical “cut line” that sweeps from minx to maxx.
    """
    if fraction <= 0:
        return None
    if fraction >= 1:
        # Entire polygon
        return polygon

    original_area = polygon.area
    target_area = fraction * original_area

    minx, miny, maxx, maxy = polygon.bounds

    # Simple edge case: if polygon has zero area or zero width
    if original_area == 0 or maxx == minx:
        return None

    # Binary search for the correct x-cut to get target_area
    left = minx
    right = maxx

    for _ in range(20):  # 20 iterations is usually enough to converge
        mid = (left + right) / 2
        # Create a rectangular "cut" from minx up to mid
        cut_poly = box(minx, miny, mid, maxy)
        clipped = polygon.intersection(cut_poly)
        clipped_area = clipped.area

        if clipped_area < target_area:
            left = mid
        else:
            right = mid

    # Final cut at the midpoint of left/right after searching
    final_cut_x = (left + right) / 2
    final_cut_poly = box(minx, miny, final_cut_x, maxy)
    sub_polygon = polygon.intersection(final_cut_poly)
    return sub_polygon


def get_margins_hab_fractions(farm_gdf):
    margin_records = []
    habitat_records = []

    for idx, row in farm_gdf.iterrows():
        poly = row.geometry
        mfrac = row.get('margin_intervention', 0.0)
        hfrac = row.get('habitat_conversion', 0.0)

        # 1) partial boundary for margin_intervention
        if mfrac > 0:
            partial_line = partial_boundary(poly, mfrac)
            if partial_line is not None and not partial_line.is_empty:
                margin_records.append({
                    'id': row['id'],
                    'geometry': partial_line
                })

        # 2) partial area for habitat_conversion
        if hfrac > 0:
            subpoly = partial_polygon_left_to_right(poly, hfrac)
            if subpoly is not None and not subpoly.is_empty:
                habitat_records.append({
                    'id': row['id'],
                    'geometry': subpoly
                })

    if len(margin_records) > 0:
        margin_lines_gdf = gpd.GeoDataFrame(margin_records, crs=farm_gdf.crs)
    else:
        margin_lines_gdf = None

    if len(habitat_records) > 0:
        converted_polys_gdf = gpd.GeoDataFrame(habitat_records, crs=farm_gdf.crs)
    else:
        converted_polys_gdf = None
    return margin_lines_gdf, converted_polys_gdf


def plot_farm(farm_path, farm_gdf, margin_lines_gdf, converted_polys_gdf, mode="og"):
    fig, ax = plt.subplots(figsize=(8, 6))

    # a) Plot entire boundary in grey
    farm_gdf.boundary.plot(ax=ax, color='grey', aspect=1)
    farm_gdf = farm_gdf.rename_geometry("farm_geom")

    # b) Plot partial boundary lines (red)
    if margin_lines_gdf is not None:
        margin_lines_gdf.plot(ax=ax, color='red', linewidth=2, aspect=1)
        margin_lines_gdf["margin_wkt"] = margin_lines_gdf.geometry.to_wkt()
        margin_lines_gdf = margin_lines_gdf.drop(columns="geometry")
        common_cols = set(farm_gdf.columns).intersection(margin_lines_gdf.columns) - {'id', 'margin_wkt'}
        margin_lines_gdf = margin_lines_gdf.drop(columns=common_cols)
        combined_gdf = farm_gdf.merge(margin_lines_gdf[['id', 'margin_wkt']], on='id', how='left')

    # c) Plot partial polygons for habitat conversion (green)
    if converted_polys_gdf is not None:
        if margin_lines_gdf is None:
            combined_gdf = farm_gdf
        converted_polys_gdf.plot(ax=ax, color='green', alpha=0.5, aspect=1)
        converted_polys_gdf["converted_wkt"] = converted_polys_gdf.geometry.to_wkt()
        converted_polys_gdf = converted_polys_gdf.drop(columns="geometry")
        common_cols = set(farm_gdf.columns).intersection(converted_polys_gdf.columns) - {'id', 'converted_wkt'}
        converted_polys_gdf = converted_polys_gdf.drop(columns=common_cols)
        combined_gdf = combined_gdf.merge(converted_polys_gdf[['id', 'converted_wkt']], on='id', how='left')

    # d) Plot existing habitats (blue)
    hab_plots_gdf = farm_gdf[farm_gdf['type'] == 'hab_plots']
    if not hab_plots_gdf.empty:
        hab_plots_gdf.plot(ax=ax, color='blue', alpha=0.5, aspect=1)

    # Add legend
    patches = [
        mpatches.Patch(color='red', label='Margin Interventions'),
        mpatches.Patch(color='green', label='Habitat Conversions'),
        mpatches.Patch(color='blue', label='Existing Habitats'),
        mpatches.Patch(color='grey', label='Field Boundaries')
    ]
    plt.legend(handles=patches, loc='upper left')

    # plt.title("Example: Partial Boundary vs. Partial Area")
    plt.tight_layout()
    plt.savefig(os.path.join(farm_path, f"interventions_{mode}.svg"), dpi=200)
    plt.close(fig)

    if margin_lines_gdf is None and converted_polys_gdf is None:
        combined_gdf = farm_gdf

    combined_gdf = combined_gdf.set_geometry("farm_geom", inplace=False)
    combined_gdf.to_file(os.path.join(farm_path, f"combined_{mode}.geojson"), driver="GeoJSON")


def plot_combined(combined_path, mode):
    combined_gdf = gpd.read_file(os.path.join(combined_path, f"all_plots_interventions_{mode}.geojson"))

    # 2) Convert margin_wkt to geometry (if present)
    if "margin_wkt" in combined_gdf.columns:
        margin_lines_gdf = combined_gdf[combined_gdf["margin_wkt"].notna()].copy()
        if not margin_lines_gdf.empty:
            margin_lines_gdf["geometry"] = margin_lines_gdf["margin_wkt"].apply(wkt.loads)
            margin_lines_gdf = gpd.GeoDataFrame(margin_lines_gdf, geometry="geometry", crs=combined_gdf.crs)
        else:
            margin_lines_gdf = None
    else:
        margin_lines_gdf = None

    # 3) Convert converted_wkt to geometry (if present)
    if "converted_wkt" in combined_gdf.columns:
        converted_polys_gdf = combined_gdf[combined_gdf["converted_wkt"].notna()].copy()
        if not converted_polys_gdf.empty:
            converted_polys_gdf["geometry"] = converted_polys_gdf["converted_wkt"].apply(wkt.loads)
            converted_polys_gdf = gpd.GeoDataFrame(converted_polys_gdf, geometry="geometry", crs=combined_gdf.crs)
        else:
            converted_polys_gdf = None
    else:
        converted_polys_gdf = None

    fig, ax = plt.subplots(figsize=(8, 8))
    combined_gdf.plot(ax=ax, facecolor='lightgray', edgecolor='black', alpha=0.5, aspect=1, linewidth=1)

    # b) Plot margin lines in red
    if margin_lines_gdf is not None:
        margin_lines_gdf.plot(ax=ax, color='red', linewidth=2, aspect=1)

    # c) Plot converted polygons in green
    if converted_polys_gdf is not None:
        converted_polys_gdf.plot(ax=ax, facecolor='lime', edgecolor='green', alpha=0.5, aspect=1)

    # d) Plot existing habitats (blue) if 'type' column is present
    if 'type' in combined_gdf.columns:
        hab_plots_gdf = combined_gdf[combined_gdf['type'] == 'hab_plots']
        if not hab_plots_gdf.empty:
            hab_plots_gdf.plot(ax=ax, facecolor='forestgreen', edgecolor='black', alpha=0.5, aspect=1)

    # 5) Add legend
    patches = [
        mpatches.Patch(color='lightgray', label='Agricultural Plots'),
        mpatches.Patch(color='forestgreen', label='Existing Habitats'),
        mpatches.Patch(color='red', label='Margin Interventions'),
        mpatches.Patch(color='lime', label='Habitat Conversions')
    ]
    plt.legend(handles=patches, loc='upper left', fontsize=18, title_fontsize=18)
    ax.set_xlabel('X-Coordinate', fontsize=18)
    ax.set_ylabel('Y-Coordinate', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')

    plt.tight_layout()
    plt.savefig(os.path.join(combined_path, f"all_plots_interventions_{mode}.svg"), dpi=300)
    plt.close(fig)


def run_farm_combined_interventions_og():
    for farm_id in farm_ids:
        farm_path = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms", f"farm_{farm_id}")
        farm_gdf = gpd.read_file(os.path.join(farm_path, "input.geojson"))
        interventions = gpd.read_file(os.path.join(farm_path, "output_gt.geojson"))
        interventions = interventions.drop(["geometry"], axis=1)
        common_cols = set(farm_gdf.columns).intersection(interventions.columns) - {'id'}
        interventions_subset = interventions.drop(columns=common_cols)
        farm_gdf = farm_gdf.merge(interventions_subset, on="id", how="left")
        farm_gdf = farm_gdf.fillna(0)
        margin_lines_gdf, converted_polys_gdf = get_margins_hab_fractions(farm_gdf)
        plot_farm(farm_path, farm_gdf, margin_lines_gdf, converted_polys_gdf, mode="og")


def run_farm_combined_interventions_pred():
    for farm_id in farm_ids:
        print(f"Running farm: {farm_id}")
        farm_path = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms", f"farm_{farm_id}")
        input_src = os.path.join(farm_path, "input.geojson")
        input_dst = os.path.join(farm_path, "input_cp.geojson")
        shutil.copyfile(input_src, input_dst)

        farm_gdf = gpd.read_file(os.path.join(farm_path, "input.geojson"))

        os.chdir(farm_path)
        code, _, err = capture.run_python_script("best_heuristics_gem.py")
        if not code:
            interventions = gpd.read_file(os.path.join(farm_path, "output.geojson"))
        else:
            print(err)
            continue

        shutil.copyfile(input_dst, input_src)

        if "geometry" in interventions:
            interventions = interventions.drop(["geometry"], axis=1)
        if "type" in interventions:
            interventions = interventions.drop(["type"], axis=1)

        farm_gdf = farm_gdf.merge(interventions, on="id", how="left")
        farm_gdf.fillna(0)
        margin_lines_gdf, converted_polys_gdf = get_margins_hab_fractions(farm_gdf)
        plot_farm(farm_path, farm_gdf, margin_lines_gdf, converted_polys_gdf, mode="pred")


def combine_farms():
    gdfs = []
    ref_cols = ['type', 'id', 'label', 'yield']
    for farm_id in farm_ids:
        farm_path = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms", f"farm_{farm_id}")
        gdf = gpd.read_file(os.path.join(farm_path, f"combined_{mode}.geojson"))
        if mode == "pred":
            columns = gdf.columns
            for gdf_col in columns:
                for col in ref_cols:
                    if col in gdf_col:
                        if col != gdf_col:
                            if col in columns or col in gdf.columns:
                                gdf = gdf.drop([gdf_col], axis=1)
                            else:
                                gdf.rename({gdf_col: col}, axis=1, inplace=True)
                        break
        gdf["farm_id"] = farm_id
        gdfs.append(gdf)

    combined_gdf = pd.concat(gdfs, ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry="geometry")
    combined_gdf.to_file(os.path.join(combined_path, f"all_plots_interventions_{mode}.geojson"), driver="GeoJSON")

    plot_combined(combined_path, mode)


def plot_farm_redone(ax, plots, title="Farm Plots"):
    patches = []

    # Plot each farm polygon (the underlying fields).
    for p in plots:
        geom_wkt_str = p.get("geometry_wkt")
        if not geom_wkt_str:
            continue
        try:
            geom = wkt.loads(geom_wkt_str)
        except Exception as e:
            print(f"Error loading plot geometry: {geom_wkt_str} - {e}")
            continue
        if geom.is_empty:
            continue

        color = 'lightgray' if p.get('plot_type') == 'ag_plot' else 'brown'

        # Handle single or multi polygons.
        if geom.geom_type == 'Polygon':
            coords = list(geom.exterior.coords)
            poly_patch = MplPolygon(coords, closed=True, facecolor=color,
                                    edgecolor='black', alpha=0.5)
            patches.append(poly_patch)
        elif geom.geom_type == 'MultiPolygon':
            for subg in geom.geoms:
                coords = list(subg.exterior.coords)
                poly_patch = MplPolygon(coords, closed=True, facecolor=color,
                                        edgecolor='black', alpha=0.5)
                patches.append(poly_patch)

    # Actually add the underlying polygon patches to the plot.
    if patches:
        pc = PatchCollection(patches, match_original=True)
        ax.add_collection(pc)

    # Plot margin geometries (shown in red).
    for p in plots:
        margin_wkt_str = p.get("margin_wkt")
        if not margin_wkt_str:
            continue
        try:
            margin_geom = wkt.loads(margin_wkt_str)
        except Exception as e:
            print(f"Error loading margin geometry: {margin_wkt_str} - {e}")
            continue

        if margin_geom.is_empty:
            continue

        geom_type = margin_geom.geom_type
        if geom_type in ['LineString', 'LinearRing']:
            x, y = margin_geom.xy
            ax.plot(x, y, color='red', linewidth=2)
        elif geom_type == 'MultiLineString':
            for line in margin_geom.geoms:
                x, y = line.xy
                ax.plot(x, y, color='red', linewidth=2)
        elif geom_type == 'Polygon':
            x, y = margin_geom.exterior.xy
            ax.plot(x, y, color='red', linewidth=2)
        elif geom_type == 'MultiPolygon':
            for subg in margin_geom.geoms:
                x, y = subg.exterior.xy
                ax.plot(x, y, color='red', linewidth=2)
        elif geom_type == 'GeometryCollection':
            for part in margin_geom.geoms:
                if part.geom_type in ['LineString', 'LinearRing']:
                    x, y = part.xy
                    ax.plot(x, y, color='red', linewidth=2)
                elif part.geom_type == 'Polygon':
                    x, y = part.exterior.xy
                    ax.plot(x, y, color='red', linewidth=2)

    # Plot habitat geometries (shown in bright green).
    for p in plots:
        habitat_wkt_str = p.get("habitat_wkt")
        if not habitat_wkt_str:
            continue
        try:
            habitat_geom = wkt.loads(habitat_wkt_str)
        except Exception as e:
            print(f"Error loading habitat geometry: {habitat_wkt_str} - {e}")
            continue

        if habitat_geom.is_empty:
            continue

        if habitat_geom.geom_type == 'Polygon':
            coords = list(habitat_geom.exterior.coords)
            patch = MplPolygon(coords, closed=True, facecolor='lime',
                               edgecolor='green', alpha=0.5)
            ax.add_patch(patch)
        elif habitat_geom.geom_type == 'MultiPolygon':
            for sub_poly in habitat_geom.geoms:
                coords = list(sub_poly.exterior.coords)
                patch = MplPolygon(coords, closed=True, facecolor='lime',
                                   edgecolor='green', alpha=0.5)
                ax.add_patch(patch)
        elif habitat_geom.geom_type == 'GeometryCollection':
            for part in habitat_geom.geoms:
                if part.geom_type == 'Polygon':
                    coords = list(part.exterior.coords)
                    patch = MplPolygon(coords, closed=True, facecolor='lime',
                                       edgecolor='green', alpha=0.5)
                    ax.add_patch(patch)

    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.autoscale_view()


def debug_plot_subregions(plot_geom, habitat_subregions, margin_subregions):
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot the plot boundary.
    if plot_geom.geom_type == 'Polygon':
        x, y = plot_geom.exterior.xy
        ax.plot(x, y, color='black', linewidth=2, label='Plot boundary')
    elif plot_geom.geom_type == 'MultiPolygon':
        for poly in plot_geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='black', linewidth=2, label='Plot boundary')
    # Plot habitat subregions.
    for direction, subregion in habitat_subregions.items():
        if not subregion.is_empty and subregion.geom_type == 'Polygon':
            x, y = subregion.exterior.xy
            ax.plot(x, y, color='blue', linestyle='--', linewidth=1, label=f'Habitat: {direction}')
            centroid = subregion.centroid
            ax.text(centroid.x, centroid.y, direction, color='blue', fontsize=8,
                    ha='center', va='center')
    # Plot margin subregions.
    for direction, candidate in margin_subregions.items():
        if not candidate.is_empty:
            if candidate.geom_type == 'LineString':
                x, y = candidate.xy
                ax.plot(x, y, color='red', linewidth=2, label=f'Margin: {direction}')
                centroid = candidate.centroid
                ax.text(centroid.x, centroid.y, direction, color='red', fontsize=8,
                        ha='center', va='center')
            elif candidate.geom_type == 'MultiLineString':
                for line in candidate.geoms:
                    x, y = line.xy
                    ax.plot(x, y, color='red', linewidth=2, label=f'Margin: {direction}')
                    centroid = line.centroid
                    ax.text(centroid.x, centroid.y, direction, color='red', fontsize=8,
                            ha='center', va='center')
    ax.set_aspect('equal')
    ax.set_title("Debug: Habitat and Margin Subregions")
    ax.legend(loc='best', fontsize='small')
    plt.show()
    plt.close()
    print("done")


def merge_geometries(chosen_pieces, geometry_key, measure_key):
    geoms = []
    for cp in chosen_pieces:
        if measure_key == 'length':
            value = cp.get(measure_key, 0)
            cond = value > 0
        else:
            cond = True

        if cond:
            geom_str = cp.get(geometry_key)
            if geom_str:
                try:
                    g = wkt.loads(geom_str)
                    if not g.is_empty:
                        geoms.append(g)
                except Exception as e:
                    print(f"Warning: could not parse geometry for piece {cp}: {e}")
                    continue

    if not geoms:
        return None

    merged = unary_union(geoms)
    simplified = merged.simplify(tolerance=0.01, preserve_topology=True)
    return wkt.dumps(simplified, rounding_precision=2)


def process_farm_data(input_filename):
    """
    Processes the farm data by:
      - Loading JSON with 'plots' and 'chosen_pieces'.
      - Merging margin and habitat geometries for each plot.
      - Determining directional subregions (north-east, north-west, south-east, south-west).
      - Saving two JSON outputs per farm: one with geometry, one without.
      - Creating a plot PNG for each farm as well.
    """

    # Load the original JSON data.
    with open(input_filename, "r") as f:
        data = json.load(f)

    # Group plots and chosen pieces by farm_id.
    farms = {}
    for plot in data.get("plots", []):
        farm_id = plot.get("farm_id")
        if farm_id is None:
            continue
        farms.setdefault(farm_id, {"plots": [], "chosen_pieces": []})
        farms[farm_id]["plots"].append(plot)

    for cp in data.get("chosen_pieces", []):
        farm_id = cp.get("farm_id")
        if farm_id is None:
            continue
        farms[farm_id]["chosen_pieces"].append(cp)

    # Process each farm.
    for farm_id, farm_data in farms.items():
        output_dir = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms", f"farm_{farm_id}", "connectivity")
        processed_plots = []

        # Map chosen pieces by the plot they belong to (plot_index or plot_id).
        chosen_by_plot = {}
        for cp in farm_data.get("chosen_pieces", []):
            plot_index = cp.get("plot_index")
            mod_index = (plot_index + 1) % 9
            plot_index = mod_index if mod_index != 0 else 9
            if plot_index is not None:
                key = str(plot_index)
                chosen_by_plot.setdefault(key, []).append(cp)

        # Process each plot and merge margin/habitat geometries.
        for plot in farm_data.get("plots", []):
            plot_id = plot.get("plot_id")
            if plot_id is None:
                continue
            key = str(plot_id)

            # Create a new dictionary with the desired fields.
            new_plot = {
                "id": plot_id,
                "type": plot.get("plot_type"),
                "label": plot.get("label"),
                "geometry_wkt": plot.get("geometry_wkt")
            }

            # Gather chosen pieces for this plot.
            cps = chosen_by_plot.get(key, [])

            # Merge margin geometries (where length > 0).
            margin_wkt_result = merge_geometries(
                chosen_pieces=cps,
                geometry_key="geom_wkt",
                measure_key="length"
            )

            # Merge habitat geometries (where area > 0).
            habitat_wkt_result = merge_geometries(
                chosen_pieces=cps,
                geometry_key="geom_wkt",
                measure_key="area"
            )

            plot_geom = None
            if new_plot["geometry_wkt"]:
                try:
                    plot_geom = wkt.loads(new_plot["geometry_wkt"])
                except Exception as e:
                    print(f"Warning: invalid plot geometry for plot {plot_id}: {e}")

                # --- Directions block (ONLY 4 QUADRANTS) ---
                # Define dividing lines (one vertical, one horizontal).
                minx, miny, maxx, maxy = plot_geom.bounds
                x_center = (minx + maxx) / 2.0
                y_center = (miny + maxy) / 2.0

                base_boxes = {
                    "north-west": Polygon([
                        (minx, y_center), (x_center, y_center),
                        (x_center, maxy), (minx, maxy)
                    ]),
                    "north-east": Polygon([
                        (x_center, y_center), (maxx, y_center),
                        (maxx, maxy), (x_center, maxy)
                    ]),
                    "south-west": Polygon([
                        (minx, miny), (x_center, miny),
                        (x_center, y_center), (minx, y_center)
                    ]),
                    "south-east": Polygon([
                        (x_center, miny), (maxx, miny),
                        (maxx, y_center), (x_center, y_center)
                    ])
                }

                # Intersect with the plot polygon to ensure we only consider
                # parts within the actual plot boundary (for habitat).
                habitat_subregions = {
                    direction: box.intersection(plot_geom)
                    for direction, box in base_boxes.items()
                }

                # For margins, we intersect with the plot boundary (lines) in each quadrant.
                plot_boundary = plot_geom.boundary
                margin_subregions = {
                    direction: plot_boundary.intersection(box)
                    for direction, box in base_boxes.items()
                }

                # Debug plotting if desired:
                # debug_plot_subregions(plot_geom, habitat_subregions, margin_subregions)

                # Process margin directions.
                margin_directions = []
                if margin_wkt_result:
                    margin_geom = wkt.loads(margin_wkt_result)
                    chosen_margin_geoms = []
                    for direction, candidate in margin_subregions.items():
                        total_candidate_length = candidate.length if not candidate.is_empty else 0
                        if total_candidate_length == 0:
                            continue
                        inter_margin = margin_geom.intersection(candidate)

                        # Calculate intersection length:
                        if inter_margin.is_empty:
                            inter_length = 0
                        elif inter_margin.geom_type == 'MultiPoint':
                            pts = list(inter_margin.geoms)
                            pts_sorted = sorted(pts, key=lambda pt: candidate.project(pt))
                            inter_length = 0
                            if len(pts_sorted) > 1:
                                new_line = LineString([pt.coords[0] for pt in pts_sorted])
                                inter_length = new_line.length
                        else:
                            inter_length = inter_margin.length

                        # If more than 25% of that quadrant boundary is covered, mark it.
                        if (total_candidate_length > 0) and ((inter_length / total_candidate_length) > 0.2):
                            margin_directions.append(direction)
                            chosen_margin_geoms.append(candidate)

                    if chosen_margin_geoms:
                        merged_margin = unary_union(chosen_margin_geoms)
                        new_plot["margin_wkt"] = wkt.dumps(merged_margin, rounding_precision=2)
                    else:
                        new_plot["margin_wkt"] = None
                new_plot["margin_directions"] = margin_directions

                # Process habitat directions.
                habitat_directions = []
                if habitat_wkt_result:
                    habitat_geom = wkt.loads(habitat_wkt_result)
                    if not habitat_geom.is_valid:
                        habitat_geom = habitat_geom.buffer(0)

                    chosen_habitat_geoms = []
                    for direction, candidate in habitat_subregions.items():
                        total_candidate_area = candidate.area if not candidate.is_empty else 0
                        if total_candidate_area == 0:
                            continue
                        inter_habitat = habitat_geom.intersection(candidate)
                        inter_area = inter_habitat.area if not inter_habitat.is_empty else 0

                        # If more than 25% of that quadrant area is covered, mark it.
                        if (total_candidate_area > 0) and ((inter_area / total_candidate_area) > 0.8):
                            habitat_directions.append(direction)
                            chosen_habitat_geoms.append(candidate)

                    if chosen_habitat_geoms:
                        merged_habitat = unary_union(chosen_habitat_geoms)
                        new_plot["habitat_wkt"] = wkt.dumps(merged_habitat, rounding_precision=2)
                    else:
                        new_plot["habitat_wkt"] = None
                new_plot["habitat_directions"] = habitat_directions
                # --- End directions block ---

            processed_plots.append(new_plot)

        # Save data with geometry.
        output_filename_with = os.path.join(output_dir, "output_gt_with_poly_geometry_directions.json")
        with open(output_filename_with, "w") as out_f:
            json.dump(processed_plots, out_f, indent=2)
        print(f"Saved farm {farm_id} data with geometry to {output_filename_with}")

        # Save data without raw geometry WKT fields.
        processed_no_geom = []
        for p in processed_plots:
            if p["type"] == "hab_plots":
                continue

            p_copy = p.copy()
            # Remove large geometry fields
            p_copy.pop("geometry_wkt", None)
            p_copy.pop("margin_wkt", None)
            p_copy.pop("habitat_wkt", None)
            processed_no_geom.append(p_copy)

        output_filename_without = os.path.join(output_dir, "output_gt_directions.json")
        with open(output_filename_without, "w") as out_f:
            json.dump(processed_no_geom, out_f)
        print(f"Saved farm {farm_id} data without geometry to {output_filename_without}")

        # Finally, create and save a plot image.
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_farm_redone(ax, processed_plots, title=f"Farm {farm_id}")
        plot_filename = os.path.join(output_dir, "output_gt_directions.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close(fig)
        print(f"Saved plot for farm {farm_id} to {plot_filename}")


def prep_for_graph():
    connectivity_json = os.path.join(combined_path, "connectivity_interventions_pred.json")
    process_farm_data(connectivity_json)


def get_cardinal_direction(central, other):
    """
    Computes a cardinal direction (immediate neighbour) from central to other
    based on the angle between their centroids.
    """
    dx = other.x - central.x
    dy = other.y - central.y
    angle = math.degrees(math.atan2(dy, dx))
    # Map angle to one of eight directions:
    if -22.5 <= angle < 22.5:
        return "east"
    elif 22.5 <= angle < 67.5:
        return "north-east"
    elif 67.5 <= angle < 112.5:
        return "north"
    elif 112.5 <= angle < 157.5:
        return "north-west"
    elif angle >= 157.5 or angle < -157.5:
        return "west"
    elif -157.5 <= angle < -112.5:
        return "south-west"
    elif -112.5 <= angle < -67.5:
        return "south"
    elif -67.5 <= angle < -22.5:
        return "south-east"
    return "unknown"


def get_extended_direction(central_centroid, second_centroid, immediate_neighbor_indices, centroids):
    """
    For a second-level neighbour, find the immediate neighbour (of the central polygon)
    whose centroid is closest to the second-level neighbour. Then compute the direction
    from the central polygon to that immediate neighbour (d1) and from that neighbour to
    the second-level neighbour (d2). Return the combined direction as "d1-d2".
    """
    best_neighbor = None
    best_distance = float('inf')
    for idx in immediate_neighbor_indices:
        m_centroid = centroids[idx]
        dist = second_centroid.distance(m_centroid)
        if dist < best_distance:
            best_distance = dist
            best_neighbor = m_centroid
    d1 = get_cardinal_direction(central_centroid, best_neighbor)
    d2 = get_cardinal_direction(best_neighbor, second_centroid)
    return f"{d1}-{d2}"


def debug_plot_polygons(i, central_poly, immediate_ids, second_level_ids, polygons, centroids):
    """
    Plots the central polygon along with its immediate (blue) and second-level (green)
    neighbours. The central polygon is drawn in red.
    """
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Plot the central polygon in red.
    x, y = central_poly.exterior.xy
    ax.plot(x, y, color="red", linewidth=2, label="central")
    ax.plot(centroids[i].x, centroids[i].y, marker="o", color="red")

    # Plot immediate neighbours in blue.
    for idx, j in enumerate(immediate_ids):
        poly = polygons[j]
        x, y = poly.exterior.xy
        # Only label the first immediate neighbour for legend clarity.
        label = "immediate" if idx == 0 else None
        ax.plot(x, y, color="blue", linewidth=2, label=label)
        ax.plot(centroids[j].x, centroids[j].y, marker="o", color="blue")

    # Plot second-level neighbours in green.
    for idx, j in enumerate(second_level_ids):
        poly = polygons[j]
        x, y = poly.exterior.xy
        label = "second level" if idx == 0 else None
        ax.plot(x, y, color="green", linewidth=2, label=label)
        ax.plot(centroids[j].x, centroids[j].y, marker="o", color="green")

    ax.set_title(f"Polygon {i} and its neighbours")
    ax.legend()

    # Save the plot to the polygon's folder for later inspection.
    plt.show()
    plt.close()

def prep_position_data():
    # Load the GeoJSON file
    with open(os.path.join(cfg.data_dir, "biomass", "quadrants_landuse.geojson"), "r") as f:
        geojson_data = json.load(f)

    features = geojson_data["features"]
    num_features = len(features)

    # Convert geometries to shapely objects and compute centroids
    polygons = []
    centroids = []
    for feature in features:
        poly = shape(feature["geometry"])
        polygons.append(poly)
        centroids.append(poly.centroid)

    tree = STRtree(polygons)

    immediate_neighbors = {i: [] for i in range(num_features)}
    for i, poly in enumerate(polygons):
        # Query the spatial index for candidates.
        possible_neighbors = tree.query(poly)
        for c_id in possible_neighbors:
            candidate = polygons[c_id]
            if i != c_id and poly.touches(candidate):
                immediate_neighbors[i].append(c_id)
        #print(f"number of immediate neibs of {i}: {len(immediate_neighbors[i])}")

    # Build dictionary of second-level neighbours: neighbours of neighbours
    second_level_neighbors = {i: [] for i in range(num_features)}
    for i in range(num_features):
        # For each immediate neighbour of i, check its immediate neighbours.
        second_level = set()
        for neigh in immediate_neighbors[i]:
            for n2 in immediate_neighbors[neigh]:
                # Exclude the central polygon and already included immediate neighbours.
                if n2 != i and n2 not in immediate_neighbors[i]:
                    second_level.add(n2)
        second_level_neighbors[i] = list(second_level)
        #print(f"number of second neibs of {i}: {len(second_level_neighbors[i])}")

    # Process each polygon feature by feature
    output_gt = {}
    for i, feature in enumerate(features):
        folder = os.path.join(cfg.data_dir, "biomass", "landscape", str(i))
        os.makedirs(folder, exist_ok=True)

        central_props = feature["properties"]
        # Build the central dictionary
        central_dict = {
            "oat_yield": central_props["oat_yield"],
            "wheat_yield": central_props["wheat_yield"],
            "corn_yield": central_props["corn_yield"],
            "soy_yield": central_props["soy_yield"],
            "ecological_connectivity": central_props["ecological_connectivity"],
            "position": "central"
        }

        input_list = [central_dict]  # first entry is the central polygon

        # Add immediate neighbours with their computed direction
        for j in immediate_neighbors[i]:
            neighbor_props = features[j]["properties"]
            direction = get_cardinal_direction(centroids[i], centroids[j])
            neighbor_dict = {
                "oat_yield": neighbor_props["oat_yield"],
                "wheat_yield": neighbor_props["wheat_yield"],
                "corn_yield": neighbor_props["corn_yield"],
                "soy_yield": neighbor_props["soy_yield"],
                "ecological_connectivity": neighbor_props["ecological_connectivity"],
                "position": direction
            }
            input_list.append(neighbor_dict)

        # Add second-level neighbours with extended direction notation
        for j in second_level_neighbors[i]:
            neighbor_props = features[j]["properties"]
            direction = get_extended_direction(centroids[i], centroids[j], immediate_neighbors[i], centroids)
            neighbor_dict = {
                "oat_yield": neighbor_props["oat_yield"],
                "wheat_yield": neighbor_props["wheat_yield"],
                "corn_yield": neighbor_props["corn_yield"],
                "soy_yield": neighbor_props["soy_yield"],
                "ecological_connectivity": neighbor_props["ecological_connectivity"],
                "position": direction
            }
            input_list.append(neighbor_dict)

        #debug_plot_polygons(i, polygons[i], immediate_neighbors[i], second_level_neighbors[i], polygons, centroids)

        # Write the input.json file in the folder for this polygon
        input_path = os.path.join(folder, "input.json")
        with open(input_path, "w") as f:
            json.dump(input_list, f)

        # Create output_gt.json mapping this polygon's id to its land_use_decision
        output_gt[str(i)] = central_props["land_use_decision"]
    output_gt_path = os.path.join(cfg.data_dir, "biomass", "landscape", "output_gt.json")
    with open(output_gt_path, "w") as f:
        json.dump(output_gt, f)


def get_lu_accuracy_from_slurm():
    slurm_file = os.path.join(cfg.src_dir, "slurm-4.out")
    rewards = []
    with open(slurm_file, 'r') as f:
        for line in f:
            # We only process lines that contain the table cell separator.
            if "│" not in line:
                continue
            # Split the line by the vertical bar (the table uses the Unicode box drawing character).
            parts = line.split("│")
            # Expecting at least three parts, where the second-to-last is the reward cell.
            if len(parts) < 3:
                continue
            reward_str = parts[-3].strip()
            try:
                # Convert the reward string to a float.
                reward = float(reward_str)
                rewards.append(reward)
            except ValueError:
                # If conversion fails, this cell isn’t a reward number.
                continue
    total = len(rewards)
    count_above_4 = sum(1 for r in rewards if r > 4)
    fraction = count_above_4 / total

    print(f"Total rewards: {total}")
    print(f"Rewards > 4: {count_above_4}")
    print(f"Fraction: {fraction:.2f}")



if __name__ == "__main__":
    cfg = Config()
    farm_ids = np.arange(6, 11)
    capture = CommandOutputCapture()
    mode = "pred"
    combined_path = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms", "connectivity", "run_3")

    # run eco_intensification
    # run_farm_combined_interventions_og() # 1
    # run farm_evo_strat
    # copy best # 1.5
    # run_farm_combined_interventions_pred() # 2
    # combine_farms() # 3
    # run graph_connectivity # 4
    # prep_for_graph() # 5
    # run graph_evo_strat
    # run nudge_evo_strat

    # prep_position_data()
    # get_lu_accuracy_from_slurm()
    print("done")
