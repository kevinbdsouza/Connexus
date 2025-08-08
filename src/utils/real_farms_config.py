import geopandas as gpd
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from ei_ec.config import Config
from matplotlib.patches import Patch
import warnings

CONFIG_SIZE_MIN = 2
CONFIG_SIZE_MAX = 10
OUTPUT_CRS = None

# --- Helper Functions ---

def calculate_properties(gdf, id_col_name="id"):
    """Calculates area, perimeter, num_sides, and renumbers ID."""
    gdf_proj = gdf  # Assume already projected

    # Recalculate properties even if they exist, to ensure consistency
    gdf_proj['area'] = gdf_proj.geometry.area
    gdf_proj['perimeter'] = gdf_proj.geometry.length

    # Calculate num_sides (handle potential MultiPolygons gracefully, taking the first polygon's exterior)
    def get_num_sides(geom):
        if geom is None:
            return 0
        try:
            if geom.geom_type == 'Polygon':
                return len(geom.exterior.coords) - 1
            elif geom.geom_type == 'MultiPolygon':
                # Use the exterior of the first polygon in the MultiPolygon
                if len(geom.geoms) > 0:
                    return len(geom.geoms[0].exterior.coords) - 1
                else:
                    return 0
            else:
                return 0  # Or handle other types if necessary
        except Exception:
            return 0  # Catch potential errors with invalid geometries

    gdf_proj['num_sides'] = gdf_proj.geometry.apply(get_num_sides)

    # Renumber IDs starting from 1
    gdf_proj[id_col_name] = range(1, len(gdf_proj) + 1)

    return gdf_proj


def calculate_neighbours(gdf, id_col_name="id"):
    """Calculates the number of neighbours within the same GeoDataFrame."""
    # Ensure IDs are set for joining
    if id_col_name not in gdf.columns:
        gdf[id_col_name] = range(1, len(gdf) + 1)  # Ensure id column exists

    # Build spatial index
    sindex = gdf.sindex

    # Find neighbours (touching polygons)
    input_indices, tree_indices = sindex.query(gdf.geometry, predicate='touches')

    # Reconstruct the list-of-lists structure where possible_matches_list[i]
    # contains the positional indices of geometries touching geometry i.
    num_geometries = len(gdf)
    possible_matches_list = [[] for _ in range(num_geometries)]
    for i_input, i_tree in zip(input_indices, tree_indices):
        possible_matches_list[i_input].append(i_tree)

    # Create a dictionary to store neighbour counts
    neighbour_counts = {idx: 0 for idx in gdf.index}

    for i, touching_indices in enumerate(possible_matches_list):
        current_index = gdf.index[i]
        # Filter out self-matches and count unique neighbours
        neighbours = set(gdf.index[idx] for idx in touching_indices if gdf.index[idx] != current_index)
        neighbour_counts[current_index] = len(neighbours)

    gdf['num_neighbours'] = gdf.index.map(neighbour_counts)
    return gdf


def plot_geojson(gdf, output_path, title, color_column=None, cmap='viridis'):
    """Plots a GeoDataFrame and saves it to a file."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if color_column and color_column in gdf.columns:
        gdf.plot(column=color_column, ax=ax, legend=True, cmap=cmap, aspect=1,
                 legend_kwds={'title': color_column, 'loc': 'center left', 'bbox_to_anchor': (1.02, 0.5)})
    else:
        gdf.plot(ax=ax, aspect=1)

    # Add IDs as text if not too many features
    if len(gdf) < 50:  # Adjust threshold as needed
        gdf.apply(lambda x: ax.text(x.geometry.centroid.x, x.geometry.centroid.y,
                                    str(x['id']), fontsize=8, ha='center'), axis=1)

    ax.set_title(title)
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   - Saved plot: {output_path}")
    except Exception as e:
        print(f"   - Error saving plot {output_path}: {e}")
    plt.close(fig)  # Close the figure to free memory


def create_configs():
    print(f"Starting processing in: {DATA_ROOT}")

    gdf_farms_master = gpd.read_file(FARMS_MASTER_FILE)
    print(f"Loaded master farms: {FARMS_MASTER_FILE}")
    OUTPUT_CRS = gdf_farms_master.crs
    gdf_farms_master = gdf_farms_master.to_crs(OUTPUT_CRS)

    # Ensure farm_id is suitable for matching folder names
    gdf_farms_master['farm_folder_name'] = gdf_farms_master['farm_id'].apply(lambda x: f"farm_{x}")

    # 2. Group Farms into Configurations
    print("\nGrouping farms into configurations...")
    farm_folders = {p.name for p in FARMS_ROOT.iterdir() if p.is_dir() and p.name.startswith('farm_')}
    available_farm_ids = gdf_farms_master['farm_id'].tolist()
    grouped_farm_ids = set()
    configurations = []
    config_index = 1

    # Build adjacency list for all farms
    sindex_master = gdf_farms_master.sindex
    input_indices, tree_indices = sindex_master.query(gdf_farms_master.geometry, predicate='touches')
    num_inputs = len(gdf_farms_master)
    possible_matches_list = [[] for _ in range(num_inputs)]
    for i_input, i_tree in zip(input_indices, tree_indices):
        # Add the index of the touching geometry (i_tree) to the list for the input geometry (i_input)
        possible_matches_list[i_input].append(i_tree)

    adjacency = {
        gdf_farms_master.iloc[i]['farm_id']: set(
            gdf_farms_master.iloc[idx]['farm_id']  # Get farm_id using the tree index 'idx'
            for idx in possible_matches_list[i]  # Iterate through neighbours found for farm i
            if idx != i)  # Exclude self-matches (where input index equals tree index)
        for i in range(len(gdf_farms_master))  # Iterate through each input farm index 'i'
    }

    print(f"Built adjacency list for {len(adjacency)} farms.")

    for farm_id in available_farm_ids:
        if farm_id not in grouped_farm_ids:
            current_config = []
            queue = [farm_id]
            visited_for_config = {farm_id}  # Track visited nodes for this config search

            while queue and len(current_config) < CONFIG_SIZE_MAX:
                current_farm_id = queue.pop(0)

                if current_farm_id not in grouped_farm_ids:
                    current_config.append(current_farm_id)
                    grouped_farm_ids.add(current_farm_id)

                    # Find neighbours that are not already grouped and not already in queue/visited for this config
                    neighbours = adjacency.get(current_farm_id, set())
                    for neighbour_id in neighbours:
                        if neighbour_id not in grouped_farm_ids and neighbour_id not in visited_for_config:
                            if len(current_config) + len(queue) < CONFIG_SIZE_MAX:
                                queue.append(neighbour_id)
                                visited_for_config.add(neighbour_id)

            # Ensure minimum size (this might group non-contiguous farms if necessary, which is not ideal)
            # A better approach might involve different clustering algorithms if strict contiguity and size are hard requirements
            # For now, we just form the config if it reaches the min size. Leftovers are handled below.
            if len(current_config) >= CONFIG_SIZE_MIN:
                configurations.append(current_config)
                print(f"  - Formed config {config_index} with {len(current_config)} farms: {current_config}")
                config_index += 1
            elif len(current_config) > 0:
                continue
                # Handle leftovers - add them to the last valid config if possible without exceeding max size,
                # otherwise form a smaller-than-min config (or handle as error)
                # if configurations and (len(configurations[-1]) + len(current_config)) <= CONFIG_SIZE_MAX:
                #    print(
                #        f"  - Adding {len(current_config)} leftover farms {current_config} to config {len(configurations)}")
                #    configurations[-1].extend(current_config)
                # else:
                #    # Create a potentially smaller config if it cannot be merged
                #    configurations.append(current_config)
                #    print(
                #        f"  - Warning: Formed config {config_index} with only {len(current_config)} farms (below min): {current_config}")
                #    config_index += 1

    print(f"\nCreated {len(configurations)} configurations.")

    # 3. Process Each Configuration
    for i, farm_ids_in_config in enumerate(configurations):
        config_num = i + 1
        config_name = f"config_{config_num}"
        config_path = FARMS_CONFIG / config_name

        print(f"\nProcessing {config_name}...")
        config_path.mkdir(exist_ok=True)

        # --- 3a. Move Farm Folders ---
        print("  Moving farm folders...")
        farm_folders_in_config = []
        farm_ids_proper = []
        for farm_id in farm_ids_in_config:
            farm_folder_name = f"farm_{farm_id}"
            source_path = FARMS_ROOT / farm_folder_name
            dest_path = config_path / farm_folder_name

            if farm_folder_name in farm_folders:
                if source_path.exists():
                    try:
                        if not dest_path.exists():  # Avoid moving if already there (e.g., rerunning script)
                            shutil.copytree(str(source_path), str(dest_path))
                            print(f"    - Moved {farm_folder_name} to {config_name}")
                        else:
                            print(f"    - Folder {farm_folder_name} already exists in {config_name}, skipping move.")
                        farm_folders_in_config.append(farm_folder_name)
                        farm_ids_proper.append(farm_id)
                    except Exception as e:
                        print(f"    - Error moving {farm_folder_name}: {e}")
                else:
                    continue
            else:
                print(
                    f"   - Warning: Farm folder {farm_folder_name} (ID: {farm_id}) not found in {DATA_ROOT}, but listed in config.")

        farm_ids_in_config = farm_ids_proper
        # --- 3b. Create farms.geojson for the config ---
        print(f"  Creating {config_name}/farms.geojson...")
        gdf_config_farms = gdf_farms_master[gdf_farms_master['farm_id'].isin(farm_ids_in_config)].copy()

        if not gdf_config_farms.empty:
            original_id_to_folder_map = gdf_config_farms.set_index('farm_id')['farm_folder_name'].to_dict()
            original_folder_to_id_map = gdf_config_farms.set_index('farm_folder_name')['farm_id'].to_dict()

            gdf_config_farms = calculate_properties(gdf_config_farms,
                                                    id_col_name='id')  # Renumbers ID 1..N for the config

            print(f"  Renaming farm folders in {config_name} to match new sequential IDs...")
            new_farm_folders_in_config = []  # To store the potentially new folder names

            for index, farm_row in gdf_config_farms.iterrows():
                # Need the original folder name to find the folder to rename
                # And the new ID to determine the new name
                original_farm_id = farm_row['farm_id']  # Original ID must still be present
                new_sequential_id = farm_row['id']  # New ID (1, 2, ...)

                # Find the original folder name associated with this original_farm_id
                if original_farm_id in original_id_to_folder_map:
                    old_folder_name = original_id_to_folder_map[original_farm_id]
                    old_path = config_path / old_folder_name
                    new_folder_name = f"farm_{new_sequential_id}"
                    new_path = config_path / new_folder_name

                    old_path.rename(new_path)  # Rename the folder
                    print(f"    - Renamed '{old_folder_name}' to '{new_folder_name}'")
                    current_folder_to_process = new_folder_name
                    new_farm_folders_in_config.append(current_folder_to_process)

                    old_file_name = f"{old_folder_name}_filled.geojson"  # Original file name pattern
                    new_file_name = f"{new_folder_name}_filled.geojson"  # New file name pattern
                    current_folder_path = config_path / current_folder_to_process
                    old_file_path = current_folder_path / old_file_name
                    new_file_path = current_folder_path / new_file_name
                    old_file_path.rename(new_file_path)

            farm_folders_in_config = new_farm_folders_in_config

            # Calculate neighbours *within the config*
            gdf_config_farms = calculate_neighbours(gdf_config_farms, id_col_name='id')

            # Add required properties
            gdf_config_farms['label'] = 'farm'
            gdf_config_farms['type'] = 'farm'
            gdf_config_farms['yield'] = 0

            # Select and order columns for output
            output_gdf_farms = gdf_config_farms[[
                'geometry', 'label', 'type', 'id', 'yield',
                'area', 'perimeter', 'num_sides', 'num_neighbours'
            ]].copy()  # Make a copy to avoid SettingWithCopyWarning

            # Save to GeoJSON
            output_farms_path = config_path / "farms.geojson"
            try:
                # Ensure correct GeoJSON structure (FeatureCollection with properties)
                output_gdf_farms.to_file(output_farms_path, driver='GeoJSON')
                print(f"   - Saved {output_farms_path}")
            except Exception as e:
                print(f"   - Error saving {output_farms_path}: {e}")

            # Plot farms.geojson
            plot_farms_path = config_path / "farms.png"
            plot_geojson(output_gdf_farms, plot_farms_path, f"{config_name} - Farms (IDs renumbered)")
        else:
            print(f"   - Warning: No farm geometries found for config {config_num}.")

        # --- 3c. Create all_subplots.geojson for the config ---
        print(f"  Creating {config_name}/all_subplots.geojson...")
        all_subplots_features = []

        # Iterate through the farm folders *that are now inside the config folder*
        for farm_folder_name in farm_folders_in_config:
            farm_filled_path = config_path / farm_folder_name / f"{farm_folder_name}_filled.geojson"

            if farm_filled_path.exists():
                try:
                    gdf_farm_filled = gpd.read_file(farm_filled_path)
                    # Filter out features with null label (often the farm boundary itself)
                    gdf_farm_plots = gdf_farm_filled[gdf_farm_filled['label'].notna()].copy()
                    if not gdf_farm_plots.empty:
                        all_subplots_features.append(gdf_farm_plots)
                    # else:
                    #    print(f"    - No valid subplots found in {farm_filled_path}")

                except Exception as e:
                    print(f"    - Error reading or processing {farm_filled_path}: {e}")
            else:
                print(f"    - Warning: File not found {farm_filled_path}")

        if all_subplots_features:
            # Combine all subplots from this config into one GeoDataFrame
            gdf_config_subplots = pd.concat(all_subplots_features, ignore_index=True)
            gdf_config_subplots.crs = OUTPUT_CRS  # Assign CRS

            # Calculate properties (Area, Perimeter, Sides) - Renumbers ID 1..N across all subplots in config
            gdf_config_subplots = calculate_properties(gdf_config_subplots, id_col_name='id')

            # Calculate neighbours *within the config's subplots*
            gdf_config_subplots = calculate_neighbours(gdf_config_subplots, id_col_name='id')

            # Ensure yield is numeric, set 0 for hab_plots if null/missing
            gdf_config_subplots.loc[gdf_config_subplots['yield'] == 0, 'yield'] = 0.5
            gdf_config_subplots.loc[gdf_config_subplots['type'] == 'hab_plots', 'yield'] = 0.0
            gdf_config_subplots['yield'] = pd.to_numeric(gdf_config_subplots['yield'], errors='coerce').fillna(0)

            # Select and order columns
            output_gdf_subplots = gdf_config_subplots[[
                'geometry', 'label', 'type', 'id', 'yield',
                'area', 'perimeter', 'num_sides', 'num_neighbours'
            ]].copy()

            # Define the output path
            output_subplots_path = config_path / "all_subplots.geojson"

            # Define the crs section manually for exact match if needed, although geopandas usually handles it
            # crs_dict = { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } }
            # If using a different CRS, adjust this dict or let geopandas handle it.
            # For simplicity, we let geopandas write the CRS info it has.

            try:
                output_gdf_subplots.to_file(output_subplots_path, driver='GeoJSON')
                # Optional: Manually add the 'name' field if strictly required by downstream tools
                # with open(output_subplots_path, 'r') as f:
                #     data = json.load(f)
                # data['name'] = 'all_subplots'
                # with open(output_subplots_path, 'w') as f:
                #     json.dump(data, f, indent=2)

                print(f"   - Saved {output_subplots_path}")
            except Exception as e:
                print(f"   - Error saving {output_subplots_path}: {e}")

            # Plot all_subplots.geojson
            plot_subplots_path = config_path / "all_subplots.png"
            plot_geojson(output_gdf_subplots, plot_subplots_path, f"{config_name} - All Subplots", color_column='label',
                         cmap='tab20')

        else:
            print(f"   - Warning: No subplots found across all farms in config {config_num}.")

        # --- 3d. Create input.geojson for each farm within the config ---
        print(f"  Creating input.geojson for each farm...")
        # Iterate again through the farm folders *inside* the config folder
        for farm_folder_name in farm_folders_in_config:
            farm_path = config_path / farm_folder_name
            farm_filled_path = farm_path / f"{farm_folder_name}_filled.geojson"
            ei_path = farm_path / "ei"
            output_input_geojson_path = ei_path / "input.geojson"

            ei_path.mkdir(exist_ok=True)  # Create ei folder

            gdf_farm_filled = gpd.read_file(farm_filled_path)
            gdf_farm_input = gdf_farm_filled[gdf_farm_filled['label'].notna()].copy()  # Filter non-null labels

            # Calculate properties (Area, Perimeter, Sides) - Renumbers ID 1..N within this farm
            gdf_farm_input = calculate_properties(gdf_farm_input, id_col_name='id')

            # Calculate neighbours *within this specific farm's plots*
            gdf_farm_input = calculate_neighbours(gdf_farm_input, id_col_name='id')

            # Ensure yield is numeric, set 0 for hab_plots if null/missing
            gdf_farm_input.loc[gdf_farm_input['yield'] == 0, 'yield'] = 0.5
            gdf_farm_input.loc[gdf_farm_input['type'] == 'hab_plots', 'yield'] = 0.0
            gdf_farm_input['yield'] = pd.to_numeric(gdf_farm_input['yield'], errors='coerce').fillna(0)

            # Select and order columns
            output_gdf_input = gdf_farm_input[[
                'geometry', 'label', 'type', 'id', 'yield',
                'area', 'perimeter', 'num_sides', 'num_neighbours'
            ]].copy()

            # Save to input.geojson
            output_gdf_input.to_file(output_input_geojson_path, driver='GeoJSON')
            print(f"    - Saved {output_input_geojson_path}")

            plot_subplots_path = ei_path / "input.png"
            plot_geojson(output_gdf_input, plot_subplots_path, f"{farm_folder_name} - plots", color_column='label',
                         cmap='tab20')

    print("\nProcessing completed.")
    pass


def fill_all_gaps_and_plot(data_root_path,
                           output_geojson_filename="combined_subplots_filled_hull.geojson",
                           output_plot_filename="combined_subplots_filled_hull.pdf",
                           compute=True):
    """
    Fills gaps between subplots AND between farm clusters (within their convex hull)
    with 'Exposed land/barren' polygons, saves the complete GeoJSON, and plots.

    Args:
        data_root_path (str or Path): Path to the main directory containing config_* folders.
        output_geojson_filename (str): Name for the output GeoJSON file.
        output_plot_filename (str): Name for the output plot image file.
    """
    data_root = Path(data_root_path)
    if compute:
        if not data_root.is_dir():
            print(f"Error: Data root directory not found: {data_root}")
            return

        print(f"Searching for config folders in: {data_root}")
        config_folders = sorted([p for p in data_root.glob('config_*') if p.is_dir()])

        if not config_folders:
            print(f"No 'config_*' folders found in {data_root}.")
            return

        all_farm_gdfs = [] # Still need farms for the hull boundary
        all_subplots_gdfs = []
        base_crs = None

        print("Reading farm boundaries (farms.geojson) and subplots (all_subplots.geojson)...")
        # --- Read Farm Data (for Convex Hull) ---
        for config_path in config_folders:
            farm_file = config_path / "farms.geojson"
            if farm_file.exists():
                try:
                    gdf_f = gpd.read_file(farm_file)
                    if not gdf_f.empty:
                        if base_crs is None: base_crs = gdf_f.crs
                        elif gdf_f.crs != base_crs:
                            warnings.warn(f"CRS mismatch in {farm_file.name} ({gdf_f.crs}) vs expected ({base_crs}). Attempting reprojection.")
                            try: gdf_f = gdf_f.to_crs(base_crs)
                            except Exception as e:
                                print(f"  - Failed to reproject {farm_file.name}: {e}. Skipping file.")
                                continue
                        all_farm_gdfs.append(gdf_f)
                except Exception as e: print(f"  - Error reading {farm_file.relative_to(data_root)}: {e}")

        # --- Read Subplot Data ---
        for config_path in config_folders:
            subplots_file = config_path / "all_subplots.geojson"
            if subplots_file.exists():
                try:
                    gdf_s = gpd.read_file(subplots_file)
                    if not gdf_s.empty:
                        if base_crs is None: base_crs = gdf_s.crs # Set CRS if not already set by farms
                        elif gdf_s.crs != base_crs:
                             warnings.warn(f"CRS mismatch in {subplots_file.name} ({gdf_s.crs}) vs expected ({base_crs}). Attempting reprojection.")
                             try: gdf_s = gdf_s.to_crs(base_crs)
                             except Exception as e:
                                 print(f"  - Failed to reproject {subplots_file.name}: {e}. Skipping file.")
                                 continue
                        all_subplots_gdfs.append(gdf_s)
                except Exception as e: print(f"  - Error reading {subplots_file.relative_to(data_root)}: {e}")

        # --- Initial Data Checks ---
        if not all_farm_gdfs:
            print("Error: No farm boundary data (farms.geojson) could be read to define overall area.")
            return
        if not all_subplots_gdfs:
            print("Error: No subplot data (all_subplots.geojson) could be read.")
            return
        if base_crs is None:
            print("Error: Could not determine Coordinate Reference System (CRS).")
            return

        print("Combining data...")
        try:
            combined_farms_gdf = pd.concat(all_farm_gdfs, ignore_index=True)
            combined_farms_gdf = gpd.GeoDataFrame(combined_farms_gdf, geometry='geometry', crs=base_crs)

            combined_subplots_gdf = pd.concat(all_subplots_gdfs, ignore_index=True)
            combined_subplots_gdf = gpd.GeoDataFrame(combined_subplots_gdf, geometry='geometry', crs=base_crs)
        except Exception as e:
            print(f"Error during data concatenation: {e}")
            return

        # --- Fill Gaps (Internal and External using Convex Hull) ---
        print("Calculating overall boundary (convex hull of farms)...")
        try:
            # Attempt to fix invalid farm geometries before union/hull
            combined_farms_gdf['geometry'] = combined_farms_gdf.geometry.buffer(0)
            combined_farms_gdf = combined_farms_gdf[~combined_farms_gdf.is_empty]
            if combined_farms_gdf.empty: raise ValueError("No valid farm geometries found for hull calculation.")

            farm_union = combined_farms_gdf.unary_union
            # A single geometry might still be invalid after buffer(0) on parts
            if not farm_union.is_valid:
                 warnings.warn("Farm union resulted in invalid geometry. Applying buffer(0).")
                 farm_union = farm_union.buffer(0)
            overall_boundary = farm_union.convex_hull
            # Check hull validity too
            if not overall_boundary.is_valid:
                 warnings.warn("Convex hull resulted in invalid geometry. Applying buffer(0).")
                 overall_boundary = overall_boundary.buffer(0)

            print(f"Overall boundary (convex hull) type: {overall_boundary.geom_type}")

            # Attempt to fix invalid subplot geometries before union/difference
            combined_subplots_gdf['geometry'] = combined_subplots_gdf.geometry.buffer(0)
            combined_subplots_gdf = combined_subplots_gdf[~combined_subplots_gdf.is_empty]
            if combined_subplots_gdf.empty: raise ValueError("No valid subplot geometries found.")

            total_subplot_area = combined_subplots_gdf.unary_union
            if not total_subplot_area.is_valid:
                 warnings.warn("Subplot union resulted in invalid geometry. Applying buffer(0).")
                 total_subplot_area = total_subplot_area.buffer(0)

            print("Calculating geometric difference (hull - subplots)...")
            # Calculate difference between the overall hull and all subplots
            gaps_geometry = overall_boundary.difference(total_subplot_area)
            print(f"Resulting gaps geometry type: {gaps_geometry.geom_type}")
        except Exception as e:
            print(f"Error during geometric operations: {e}")
            return

        gdf_gaps = None
        if gaps_geometry and not gaps_geometry.is_empty:
             print("Processing gap geometries...")
             gap_polygons = []
             # Handle MultiPolygons or GeometryCollections resulting from difference
             if gaps_geometry.geom_type == 'Polygon':
                  gap_polygons.append(gaps_geometry)
             elif hasattr(gaps_geometry, 'geoms'): # Handles MultiPolygon, GeometryCollection
                  for geom in gaps_geometry.geoms:
                      # Extract only valid Polygons
                      if geom.geom_type == 'Polygon' and not geom.is_empty:
                           gap_polygons.append(geom)

             if gap_polygons:
                try:
                    gdf_gaps = gpd.GeoDataFrame(geometry=gap_polygons, crs=base_crs)
                    gdf_gaps['label'] = "Exposed land/barren"
                    gdf_gaps['type'] = "gap_fill"
                    # Initialize other columns to match subplots_gdf for concatenation
                    for col in ['yield', 'id', 'area', 'perimeter', 'num_sides', 'num_neighbours']:
                         if col not in gdf_gaps.columns:
                              default_val = 0.0 if col == 'yield' else 0 # ID, area etc will be recalc'd
                              gdf_gaps[col] = default_val
                    print(f"Created {len(gdf_gaps)} gap features.")
                except Exception as e:
                    print(f"Error creating GeoDataFrame for gaps: {e}")
                    gdf_gaps = None
             else:
                 print("No valid Polygon gaps found after difference operation.")
        else:
             print("No gaps found or gap geometry is empty.")

        # --- Combine original subplots and new gaps ---
        print("Combining original subplots with gap features...")
        if gdf_gaps is not None and not gdf_gaps.empty:
             cols_to_keep = ['geometry', 'label', 'type', 'yield'] # Base columns
             # Add others if they exist, ensuring consistency
             for col in combined_subplots_gdf.columns:
                 if col not in cols_to_keep and col in gdf_gaps.columns:
                     cols_to_keep.append(col)
             try:
                 # Use copies to avoid modifying original dfs if concat fails
                 final_gdf = pd.concat(
                     [combined_subplots_gdf[cols_to_keep].copy(), gdf_gaps[cols_to_keep].copy()],
                     ignore_index=True
                 )
                 # Ensure it's still a GeoDataFrame
                 final_gdf = gpd.GeoDataFrame(final_gdf, geometry='geometry', crs=base_crs)
             except Exception as e:
                 print(f"Error during final concatenation: {e}")
                 return
        else:
             print("No gap features to add. Using original subplots.")
             final_gdf = combined_subplots_gdf.copy()


        # --- Recalculate properties for the final combined dataset ---
        print("Recalculating properties (ID, area, perimeter, sides)...")
        try:
            # Assuming calculate_properties handles geometry checks, resets 'id', calculates area, etc.
            final_gdf = calculate_properties(final_gdf, id_col_name='id')

            # Optional: Recalculate neighbours (can be slow for large/complex datasets)
            # print("Recalculating neighbours...")
            # final_gdf = calculate_neighbours(final_gdf, id_col_name='id')

            # Ensure default values for potentially missing columns
            if 'num_neighbours' not in final_gdf.columns: final_gdf['num_neighbours'] = 0
            final_gdf['num_neighbours'] = final_gdf['num_neighbours'].fillna(0)
            if 'yield' not in final_gdf.columns: final_gdf['yield'] = 0.0
            final_gdf['yield'] = final_gdf['yield'].fillna(0.0) # Ensure yield is filled

        except Exception as e:
            print(f"Error during property calculation: {e}")
            return


        # --- Save Combined GeoJSON ---
        output_geojson_path = data_root / output_geojson_filename
        print(f"Saving combined data with all gaps to: {output_geojson_path} ({len(final_gdf)} features)")
        try:
            cols_order = ['id', 'label', 'type', 'area', 'perimeter', 'num_sides', 'yield', 'num_neighbours', 'geometry']
            cols_present = [c for c in cols_order if c in final_gdf.columns]
            final_gdf[cols_present].to_file(output_geojson_path, driver='GeoJSON', index=False)
            print("GeoJSON saved successfully.")
        except Exception as e:
            print(f"Error saving GeoJSON file '{output_geojson_path}': {e}")
    else:
        output_geojson_path = data_root / output_geojson_filename
        final_gdf = gpd.read_file(output_geojson_path)

    from shapely.geometry import Polygon
    import seaborn as sns
    import matplotlib.patches as mpatches
    d = {'label': ['Water', 'Forest', 'Urban', 'Exposed land/barren', 'Forest'],
         'geometry': [Polygon([(0, 0), (1, 1), (1, 0)]),
                      Polygon([(2, 2), (3, 3), (3, 2)]),
                      Polygon([(0, 2), (1, 3), (1, 2)]),
                      Polygon([(1, 1), (2, 2), (2, 1)]),  # The 'gap'
                      Polygon([(0, 1), (1, 2), (1, 1)])]}
    # output_plot_filename = "professional_land_cover.png"
    # data_root = Path(".")
    # --- End of Mock Data ---

    ## Plotting Code
    # --- 1. Configuration for a Professional Look ---
    print("Configuring plot style...")
    FONT_SIZE = 14
    # Use seaborn for a clean, modern aesthetic with a grid
    sns.set_style("ticks")
    # Centralize font and style settings for consistency
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',  # A clean, standard font
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE + 2,  # Title is often slightly larger
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
        'legend.title_fontsize': FONT_SIZE,
    })

    # --- 2. Data & Color Preparation ---
    gap_label = "Exposed land/barren"
    gap_alpha = 0.4  # Slightly increased for better visibility

    # Get all unique labels and create a color map
    unique_labels = sorted(final_gdf['label'].unique())
    cmap = plt.get_cmap('tab20')  # A good colormap for many categories

    # Build the color dictionary
    color_map = {label: cmap(i / len(unique_labels)) for i, label in enumerate(unique_labels)}
    # You can still override specific colors if you wish
    color_map[gap_label] = "forestgreen"

    # --- 3. Plotting ---
    print("Generating professional plot...")
    # Use a figure size that fits a report or presentation well
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Plot each category individually for full control over style
    for label, data in final_gdf.groupby('label'):
        color = color_map[label]
        alpha = gap_alpha if label == gap_label else 1.0

        data.plot(
            ax=ax,
            color=color,
            alpha=alpha,
            edgecolor='black',  # Adds a crisp border to polygons
            linewidth=0.5,
            aspect=1
        )

    # --- 4. Custom Legend ---
    # Create legend handles manually to correctly display the alpha
    legend_patches = []
    for label in unique_labels:
        patch = mpatches.Patch(
            color=color_map[label],
            alpha=gap_alpha if label == gap_label else 1.0,
            label=label
        )
        legend_patches.append(patch)

    ax.legend(
        handles=legend_patches,
        title='Land Type',
        loc='center left',
        bbox_to_anchor=(1.02, 0.5)  # Position cleanly outside the plot
    )

    # --- 5. Final Touches ---
    #ax.set_title("Land Cover Classification", fontweight='bold')
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect('equal', adjustable='box')

    # Remove top and right plot borders for a cleaner look
    sns.despine(ax=ax, top=True, right=True)

    # --- 6. Saving the Figure ---
    output_plot_path = data_root / output_plot_filename
    print(f"Saving final plot to: {output_plot_path}")
    # Save at a higher resolution for better quality
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print("Plot saved successfully.")
    plt.close(fig)

# --- Main Script ---

if __name__ == "__main__":
    cfg = Config()
    DATA_ROOT = Path(os.path.join(cfg.disk_dir, "crop_inventory"))
    FARMS_ROOT = Path(os.path.join(DATA_ROOT, "farms_s"))
    FARMS_CONFIG = Path(os.path.join(DATA_ROOT, "farms_config_s"))
    FARMS_MASTER_FILE = os.path.join(DATA_ROOT, "farms_m_s.geojson")

    # create_configs()

    fill_all_gaps_and_plot(FARMS_CONFIG, compute=False)
