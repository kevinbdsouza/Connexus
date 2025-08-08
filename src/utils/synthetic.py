from ei_ec.config import Config
import os
import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPoint, box
from shapely.ops import unary_union
from shapely import voronoi_polygons
import matplotlib.pyplot as plt
import json
import random
import pandas as pd
import numpy as np


def extract_all_farm_boundaries(all_farms):
    boundary_features = []
    for farm_fc in all_farms:
        # The boundary is the 0th feature
        boundary_feature = [feat for feat in farm_fc["features"] if feat["properties"]["label"] == "farm"][0]
        boundary_features.append(boundary_feature)

    # Construct a new FeatureCollection with just boundary features
    result_fc = {
        "type": "FeatureCollection",
        "features": boundary_features
    }
    return result_fc


def partition_polygon_into_subplots(polygon, n_subplots):
    """
    Partition a single Polygon into n_subplots sub-polygons using
    a random Voronoi approach (on random sample points within the Polygon).

    Returns a list of Shapely Polygons that exactly fill the original polygon
    (except for tiny floating-point tolerances).
    """
    minx, miny, maxx, maxy = polygon.bounds

    # Generate random points inside the polygon
    points = []
    # Add a check for very small polygons where generating points might fail
    if polygon.area < 1e-6: # Avoid issues with tiny slivers
        return [polygon]

    attempts = 0
    max_attempts = n_subplots * 100 # Limit attempts to avoid infinite loops

    while len(points) < n_subplots and attempts < max_attempts:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        p = Point(x, y)
        # Only accept points that lie inside the polygon
        if polygon.contains(p):
            points.append(p)
        attempts += 1

    # If not enough points could be generated (e.g., very thin polygon),
    # return the original polygon as a single subplot.
    if len(points) < 2: # Voronoi needs at least 2 points
         return [polygon]

    # Create a MultiPoint geometry
    mp = MultiPoint(points)

    # Create a Voronoi diagram from these points, clipped by the polygon
    try:
        # Extend slightly beyond bounds to ensure full coverage after intersection
        envelope = polygon.buffer(1e-9).envelope # Use buffer to handle potential invalid geometry
        voronoi_result = voronoi_polygons(mp, extend_to=envelope)
    except Exception as e:
        print(f"Voronoi generation failed: {e}. Returning single polygon.")
        return [polygon] # Fallback if voronoi fails

    # The result is a GeometryCollection. Clip each cell by the farm polygon.
    sub_polygons = []
    for cell in voronoi_result.geoms:
        # Ensure intersection happens with the original polygon
        clipped = cell.intersection(polygon)
        if not clipped.is_empty and clipped.area > 1e-9: # Filter out empty or near-empty results
            if clipped.geom_type == "Polygon":
                sub_polygons.append(clipped)
            elif clipped.geom_type == "MultiPolygon":
                sub_polygons.extend(list(g for g in clipped.geoms if g.area > 1e-9)) # Filter parts of MultiPolygon too

    # Check if the resulting subplots cover the original polygon sufficiently
    total_sub_area = unary_union(sub_polygons).area
    if not np.isclose(total_sub_area, polygon.area, rtol=1e-3):
         # If coverage is poor, possibly due to clipping issues, return the original
         # print(f"Warning: Subplot area ({total_sub_area}) differs significantly from original ({polygon.area}). Returning single polygon.")
         return [polygon]


    return sub_polygons


def create_farm_polygon_definitions(num_farms=20, target_avg_farm_area_sqm=1e6, buffer_factor=1.5):
    """
    Generates a specified number of farm polygons using Voronoi tessellation,
    aiming for a target average area. Calculates geometric properties.
    Introduces shuffling to randomize size distribution while maintaining full tiling.
    """
    topology_name = "voronoi"
    target_total_area = num_farms * target_avg_farm_area_sqm

    # Estimate the side length of a square bounding box needed
    side_length = np.sqrt(target_total_area) * buffer_factor
    min_coord = 0
    max_coord = side_length
    bounding_box = box(min_coord, min_coord, max_coord, max_coord)

    # --- Revert to generating exactly num_farms points --- 
    xs = np.random.uniform(min_coord, max_coord, num_farms)
    ys = np.random.uniform(min_coord, max_coord, num_farms)
    points = MultiPoint(list(zip(xs, ys)))

    # Generate Voronoi polygons clipped to the bounding box
    try:
        voronoi_regions = list(voronoi_polygons(points, extend_to=bounding_box).geoms)
    except Exception as e:
        print(f"Error during Voronoi generation: {e}. Trying with buffer.")
        try:
            buffered_box = bounding_box.buffer(1e-6)
            voronoi_regions = list(voronoi_polygons(points, extend_to=buffered_box).geoms)
            voronoi_regions = [poly.intersection(bounding_box) for poly in voronoi_regions]
        except Exception as e2:
             print(f"Voronoi generation failed even with buffer: {e2}. Returning empty.")
             return {}, topology_name

    # Filter out any potential empty/invalid geometries and keep only Polygons
    farm_polygons_raw = [poly for poly in voronoi_regions if isinstance(poly, Polygon) and not poly.is_empty and poly.is_valid]

    # --- Removed sampling logic, as we now generate exactly num_farms points ---
    # Check if the number of valid polygons matches num_farms (it should)
    actual_num_farms = len(farm_polygons_raw)
    if actual_num_farms != num_farms:
         print(f"Warning: Generated {actual_num_farms} valid polygons, but expected {num_farms}. This might indicate issues.")
         # Decide how to handle: maybe proceed with actual_num_farms, or return error
         if actual_num_farms == 0:
             print("No valid farm polygons were generated.")
             return {}, topology_name
         # else: proceed with actual_num_farms

    # --- Calculate properties for each raw polygon BEFORE shuffling ---
    temp_farm_info = []
    # Use actual_num_farms for the loop range
    for i in range(actual_num_farms):
        poly = farm_polygons_raw[i]
        area = poly.area
        perimeter = poly.length
        num_sides = len(poly.exterior.coords) - 1 if poly.exterior else 0

        # Calculate neighbours based on original positions
        num_neighbours = 0
        # Use actual_num_farms for the inner loop range as well
        for j in range(actual_num_farms):
            if i == j:
                continue
            other_poly = farm_polygons_raw[j]
            if poly.buffer(1e-9).intersects(other_poly.buffer(1e-9)):
                 num_neighbours += 1

        temp_farm_info.append({
            "polygon": poly,
            "area": area,
            "perimeter": perimeter,
            "num_sides": num_sides,
            "num_neighbours": num_neighbours
        })

    # --- Shuffle the list containing polygons and their properties ---
    random.shuffle(temp_farm_info)

    # --- Populate the final farm_data dictionary from the shuffled list ---
    farm_data = {}
    # Assign IDs based on the number of polygons we actually have
    for i in range(actual_num_farms):
        farm_id = i + 1
        farm_data[farm_id] = temp_farm_info[i]

    return farm_data, topology_name


def main():
    # Configuration (can be adjusted)
    NUM_FARMS_TO_GENERATE = random.randint(5, 10) # Vary number of farms per run
    TARGET_AVG_AREA = 500_000 # 1 sq km in sqm

    # 1) Create the farm polygons dynamically
    farm_definitions, topology_name = create_farm_polygon_definitions(
        num_farms=NUM_FARMS_TO_GENERATE,
        target_avg_farm_area_sqm=TARGET_AVG_AREA
    )

    if not farm_definitions:
        print("No farms were generated. Exiting.")
        return

    farm_collections = []
    farm_boundaries = []

    # 2) Partition each farm polygon into subplots
    for farm_id, data in farm_definitions.items():
        polygon = data["polygon"]
        n_subplots = random.randint(5, 10) # Vary number of subplots
        subplots = partition_polygon_into_subplots(polygon, n_subplots=n_subplots)

        # Build a GeoDataFrame for this farm (1 boundary feature + subplots)
        # The first feature: the farm boundary with new properties
        farm_boundary = {
            "type": "Feature",
            "properties": {
                "label": "farm",
                "type": "farm",
                "id": farm_id,
                "yield": 0, # Original yield property
                "area": data["area"],
                "perimeter": data["perimeter"],
                "num_sides": data["num_sides"],
                "num_neighbours": data["num_neighbours"],
            },
            "geometry": polygon.__geo_interface__
        }
        farm_boundaries.append(farm_boundary)

        # The subplot features
        subplot_features = []
        for i, spoly in enumerate(subplots, start=1):
            # Calculate subplot geometric properties
            subplot_area = spoly.area
            subplot_perimeter = spoly.length
            subplot_num_sides = len(spoly.exterior.coords) - 1

            # Calculate subplot neighbours (within the same farm)
            subplot_num_neighbours = 0
            for j, other_spoly in enumerate(subplots):
                if i - 1 == j: # Don't compare with self (using i-1 because enumerate starts at 1)
                    continue
                # Check for touching using relate or buffer/intersects
                if spoly.buffer(1e-9).intersects(other_spoly.buffer(1e-9)):
                     # Refine check like in farm neighbour calculation
                     relate_pattern = spoly.relate(other_spoly)
                     if relate_pattern[0] in ('F', '1', '2') and relate_pattern[4] in ('T', '1'):
                         subplot_num_neighbours += 1

            # Assign subplot properties (using existing random logic)
            plot_type = random.choices(["ag_plot", "hab_plots"], weights=(3, 2))[0]
            # Ensure weights are not all zero before calling random.choices
            current_ag_weights = ag_weights if sum(ag_weights) > 0 else ([1] * len(ag_labels) if ag_labels else [1])
            current_ag_labels = ag_labels if ag_labels else ["Unknown Ag"]
            label = random.choices(current_ag_labels, weights=current_ag_weights)[0]
            if plot_type == "hab_plots":
                current_hab_weights = hab_weights if sum(hab_weights) > 0 else ([1] * len(hab_labels) if hab_labels else [1])
                current_hab_labels = hab_labels if hab_labels else ["Unknown Hab"]
                label = random.choices(current_hab_labels, weights=current_hab_weights)[0]

            # Ensure yield lists/weights are valid
            current_yield_weights = yield_weights if sum(yield_weights) > 0 else ([1] * len(yields) if yields else [1])
            current_yields = yields if yields else [0.5]

            yield_choice = random.choices(current_yields, weights=current_yield_weights)[0] if plot_type == "ag_plot" else 0
            subplot_features.append({
                "type": "Feature",
                "properties": {
                    "label": label,
                    "type": plot_type,
                    "id": i, # This is subplot ID within the farm
                    "yield": yield_choice,
                    "area": subplot_area,
                    "perimeter": subplot_perimeter,
                    "num_sides": subplot_num_sides,
                    "num_neighbours": subplot_num_neighbours # Neighbours within this farm
                },
                "geometry": spoly.__geo_interface__
            })

        # Combine subplots into one FeatureCollection per farm for individual farm output
        fc = {
            "type": "FeatureCollection",
            "features": subplot_features
        }
        farm_collections.append((farm_id, fc)) # Keep track of farm_id

    # Create the overall farm boundaries GeoJSON
    farm_boundaries_json = {
            "type": "FeatureCollection",
            "features": farm_boundaries
        }

    # Define output paths using run_dir from the __main__ block
    # Ensure run_dir is accessible here. It might need to be passed as an argument
    # or accessed via a global variable if main is called from __main__.
    # Assuming run_dir is globally accessible from the __main__ block context for now.
    global run_dir # Declare run_dir as global if it's defined in __main__

    shape_file = os.path.join(run_dir, "farms.geojson")
    img_path = os.path.join(run_dir, "farms.png")

    # Write the farm boundaries GeoJSON
    with open(shape_file, 'w') as f:
        json.dump(farm_boundaries_json, f, indent=2) # Added indent for readability

    # Create and plot the GeoDataFrame for farm boundaries
    gdf_farms = gpd.read_file(shape_file)
    gdf_farms["id"] = gdf_farms["id"].astype(int)
    # Plotting farms coloured by ID
    plot_gdf(gdf_farms, img_path, column="id")

    # Write individual farm subplot files and plot them
    farm_crs = gdf_farms.crs # Get CRS from the boundaries GeoDataFrame
    for farm_id, farm_fc in farm_collections:
        farm_dir = os.path.join(run_dir, f"farm_{farm_id}") # Use farm_id in dir name
        os.makedirs(farm_dir, exist_ok=True)

        shape_file_farm = os.path.join(farm_dir, "input.geojson")
        with open(shape_file_farm, 'w') as f:
            json.dump(farm_fc, f, indent=2) # Added indent

        img_path_farm = os.path.join(farm_dir, "input.png")
        # Check if features exist before trying to read/plot
        if farm_fc["features"]:
            try:
                gdf_farm_subplots = gpd.GeoDataFrame.from_features(farm_fc["features"], crs=farm_crs) # Use same CRS
                if not gdf_farm_subplots.empty:
                     plot_gdf(gdf_farm_subplots, img_path_farm, column="label")
            except Exception as e:
                print(f"Error processing or plotting farm {farm_id} subplots: {e}")
        #else:
        #    print(f"Farm {farm_id} has no subplot features to plot.")


    print(f"Generated {len(farm_definitions)} farms using '{topology_name}' topology.")


def plot_gdf(gdf, img_path, column):
    if gdf.empty:
        print(f"Skipping plot generation for {img_path} as GeoDataFrame is empty.")
        return
    fig, ax = plt.subplots(figsize=(10, 10))
    # Check if the column exists
    if column not in gdf.columns:
        print(f"Warning: Column '{column}' not found in GeoDataFrame for plotting {img_path}. Plotting without color.")
        gdf.plot(ax=ax, aspect=1)
    else:
        # Determine if the column is categorical or numeric for plotting
        # Check if column dtype is explicitly categorical or object
        is_categorical_dtype = gdf[column].dtype == 'object' or isinstance(gdf[column].dtype, pd.CategoricalDtype)
        # Heuristic: if few unique values compared to total rows, treat as categorical
        is_likely_categorical = gdf[column].nunique() < 20 if len(gdf) > 0 else False

        is_categorical = is_categorical_dtype or is_likely_categorical

        try:
            gdf.plot(ax=ax, column=column, legend=True, aspect=1, categorical=is_categorical, legend_kwds={'loc': 'upper left', 'bbox_to_anchor': (1.02, 1.0)}) # Adjust legend position
        except TypeError as e:
             print(f"Plotting warning for {img_path} (column: {column}): {e}. Trying without categorical flag.")
             # Fallback if categorical plotting fails (e.g., mixed types)
             gdf.plot(ax=ax, column=column, legend=True, aspect=1, legend_kwds={'loc': 'upper left', 'bbox_to_anchor': (1.02, 1.0)})
        except ValueError as e:
             print(f"Plotting error for {img_path} (column: {column}): {e}. Plotting without column.")
             gdf.plot(ax=ax, aspect=1) # Fallback to plot without column colors


    plt.title(os.path.basename(img_path).replace('.png','')) # Add title based on filename
    plt.xlabel("Easting (meters)") # Assuming projected CRS in meters
    plt.ylabel("Northing (meters)")
    plt.ticklabel_format(style='plain', axis='both') # Prevent scientific notation
    ax.set_aspect('equal', adjustable='box') # Ensure aspect ratio is equal
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.savefig(img_path, dpi=150, bbox_inches='tight') # Use bbox_inches='tight' still helpful
    plt.close(fig) # Close the figure to free memory


def get_freqs():
    label_counts = {}
    type_counts = {}
    yield_counts = {}

    # Ensure farms_dir is defined globally or passed correctly
    global farms_dir

    # Check if the base directory exists
    if not os.path.isdir(farms_dir):
        print(f"Warning: Frequency source directory '{farms_dir}' not found. Using default distributions.")
        # Return default counts leading to uniform distributions
        return ({}, {}, [0.5], [1])

    # Loop through potential farm subdirectories (adjust range if needed)
    # This part is heuristic, assuming a structure like farm_1, farm_2...
    potential_max_farm_id = 2500 # Adjust based on expected number of source farms
    farm_found = False
    for i in range(1, potential_max_farm_id):
        farm_folder = os.path.join(farms_dir, f"farm_{i}")
        # Look for a specific file, e.g., 'input.geojson' or a filled version if that's the source
        # Using 'input.geojson' as an example - CHANGE if the source file name is different
        geojson_file = os.path.join(farm_folder, "input.geojson") # *** ADJUST FILENAME if necessary ***

        if not os.path.isfile(geojson_file):
            # Try another common pattern if the first fails
            geojson_file = os.path.join(farm_folder, f"farm_{i}_filled.geojson")
            if not os.path.isfile(geojson_file):
                continue # Skip if neither file exists

        farm_found = True # Found at least one farm directory with data
        try:
            with open(geojson_file, 'r') as f:
                data = json.load(f)
                features = data.get("features", [])

                for feature in features:
                    properties = feature.get("properties", {})
                    label_val = properties.get("label")
                    type_val = properties.get("type")
                    yield_val = properties.get("yield")

                    # Ensure values are hashable before using as dict keys
                    label_val = str(label_val) if label_val is not None else 'None'
                    type_val = str(type_val) if type_val is not None else 'None'
                    # Yield might be float, handle None
                    yield_key = float(yield_val) if yield_val is not None else 'None'


                    label_counts[label_val] = label_counts.get(label_val, 0) + 1
                    type_counts[type_val] = type_counts.get(type_val, 0) + 1
                    if yield_key != 'None': # Don't count None yields here
                         yield_counts[yield_key] = yield_counts.get(yield_key, 0) + 1
        except Exception as e:
            print(f"Error reading or processing {geojson_file}: {e}")
            continue # Skip to next file on error

    if not farm_found:
        print(f"Warning: No farm data files found in '{farms_dir}'. Using default distributions.")
        return ({}, {}, [0.5], [1])


    # Process yield counts (handle potential 0.0 if it exists)
    if 0.0 in yield_counts:
        yield_counts[0.5] = yield_counts.get(0.5, 0) + yield_counts[0.0]
        del yield_counts[0.0]
    # Remove 'None' key if it was added
    if 'None' in label_counts: del label_counts['None']
    if 'None' in type_counts: del type_counts['None']
    # 'None' yield already excluded from yield_counts

    yields_out, yield_weights_out = [], []
    for k, v in yield_counts.items():
        if np.isnan(k):
            continue
        yields_out.append(k)
        yield_weights_out.append(v)

    # Return default values if counts are empty to avoid errors later
    if not yields_out:
        yields_out = [0.5] # Default yield
        yield_weights_out = [1] # Default weight
    # Ensure labels have counts, otherwise provide uniform
    if not label_counts: label_counts = {l: 1 for l in ag_labels + hab_labels}
    if not type_counts: type_counts = {"ag_plot": 1, "hab_plots": 1}


    return label_counts, type_counts, yields_out, yield_weights_out


def combine_and_plot_geojsons(run_dir, num_farms):
    """Combines individual farm input.geojson files into one and plots it."""
    main_folder = run_dir # Use the current run directory

    # Dynamically create the list of farm folders based on num_farms generated
    farm_folders = [f"farm_{i}" for i in range(1, num_farms + 1)]

    gdfs = []
    combined_crs = None # To store the CRS

    for farm_folder in farm_folders:
        subfolder_path = os.path.join(main_folder, farm_folder)
        input_geojson_path = os.path.join(subfolder_path, "input.geojson")

        if os.path.exists(input_geojson_path):
            try:
                gdf = gpd.read_file(input_geojson_path)
                if not gdf.empty:
                    gdfs.append(gdf)
                    if combined_crs is None and gdf.crs:
                        combined_crs = gdf.crs # Get CRS from the first valid file
                #else:
                #    print(f"Note: Empty GeoDataFrame in {input_geojson_path}")
            except Exception as e:
                 print(f"[WARNING] Failed to read {input_geojson_path}: {e}")
        #else:
        #    print(f"[WARNING] File not found: {input_geojson_path}")


    if not gdfs:
        print("[ERROR] No valid GeoDataFrames found to combine. Skipping combine/plot.")
        return

    # Combine all GeoDataFrames
    try:
        combined_gdf = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True),
            crs=combined_crs # Set the CRS for the combined GDF
        )
    except Exception as e:
        print(f"[ERROR] Failed to concatenate GeoDataFrames: {e}. Skipping combine/plot.")
        return


    # Save the combined GeoDataFrame
    all_plots_path = os.path.join(main_folder, "all_subplots.geojson") # Renamed for clarity
    combined_gdf.to_file(all_plots_path, driver="GeoJSON")
    print(f"Combined subplots GeoJSON saved to: {all_plots_path}")

    # Plot the combined GeoDataFrame
    plot_path = os.path.join(main_folder, "all_subplots.png") # Renamed for clarity
    plot_gdf(combined_gdf, plot_path, column="label")
    print(f"Combined subplots plot saved to: {plot_path}")


# Global definitions needed in main and potentially get_freqs
# These should be defined before they are used in main
cfg = Config()
ag_labels = ["Soybeans", "Oats", "Corn", "Canola/rapeseed", "Barley", "Spring wheat"]
hab_labels = ["Broadleaf", "Coniferous", "Exposed land/barren", "Grassland",
              "Shrubland", "Water", "Wetland"]

syn_farms_dir = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms", "mc")
os.makedirs(syn_farms_dir, exist_ok=True)

# IMPORTANT: Define the source directory for frequency calculation.
# Adjust this path if your reference farm data is located elsewhere.
farms_dir = os.path.join(cfg.data_dir, "crop_inventory", "farms") # Potentially change "farms"

# Get frequencies - Needs to run once before the loop if weights are constant per run
# Make sure `farms_dir` points to the correct location containing reference farm geojson files
label_counts, type_counts, yields, yield_weights = get_freqs()

# Handle cases where labels might be missing in the frequency data
ag_weights = [label_counts.get(it, 0) for it in ag_labels]
hab_weights = [label_counts.get(it, 0) for it in hab_labels]
# Normalize weights if they are all zero to avoid errors in random.choices
if not yields: yields = [0.5] # Ensure yields list is not empty
if sum(ag_weights) == 0: ag_weights = [1] * len(ag_labels)
if sum(hab_weights) == 0: hab_weights = [1] * len(hab_labels)
if sum(yield_weights) == 0: yield_weights = [1] * len(yields)

# Declare run_dir globally so it can be accessed by main() if needed
# Alternatively, pass it as an argument to main()
run_dir = None

if __name__ == "__main__":

    num_runs = 500
    for n in range(1, num_runs + 1):
        # Define run_dir here so it's accessible by main() and combine_and_plot_geojsons()
        # This makes run_dir global for the scope of the script execution when run directly
        globals()['run_dir'] = os.path.join(syn_farms_dir, f"config_{n}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"Starting Run {n} in {run_dir}...")

        # Call main generation logic
        main() # main now uses global run_dir

        # Combine and plot the subplots from the generated farms for this run
        # Need to know how many farms were actually generated by main()
        # We can read the farms.geojson generated by main to find out
        farms_geojson_path = os.path.join(run_dir, "farms.geojson")
        if os.path.exists(farms_geojson_path):
             try:
                 with open(farms_geojson_path, 'r') as f:
                     farms_data = json.load(f)
                     num_generated_farms = len(farms_data.get("features", []))
                 if num_generated_farms > 0:
                    print(f"Combining and plotting {num_generated_farms} farms' subplots...")
                    combine_and_plot_geojsons(run_dir, num_generated_farms)
                 else:
                    print(f"No farm features found in {farms_geojson_path}, skipping combine/plot.")
             except Exception as e:
                 print(f"Error reading {farms_geojson_path} for combining plots: {e}")
        else:
            print(f"File not found: {farms_geojson_path}, skipping combine/plot.")

        print(f"Run {n} finished. Output in {run_dir}")