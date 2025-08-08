import geopandas as gp
import matplotlib.pyplot as plt
import os
import shutil

from ei_ec.config import Config
import json
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from skimage import measure
from shapely.ops import unary_union, cascaded_union
from shapely.geometry import shape
from shapely.geometry import MultiPolygon
from shapely.strtree import STRtree
from PIL import Image
from pyproj import Transformer
from pyproj import CRS
from collections import Counter



def plot_shp():
    #shp = gp.read_file(os.path.join(cfg.data_dir, "temp", "lcsd000b16a_e", "lcsd000b16a_e.shp"))
    # shp.plot(ax=ax, facecolor="oldlace", edgecolor="dimgray")

    gdf = gp.read_file(os.path.join(cfg.data_dir, "crop_inventory", "plots.geojson"))
    list = ["Barley", "Broadleaf", "Canola/rapeseed", "Corn",
            "Grassland", "Oats", "Soybeans", "Spring wheat", "Urban/developed", "Water"]
    colors_list = ["red", "green", "blue", "yellow", "cyan", "magenta", "lightcoral", "grey", "orange", "goldenrod",
                   "olive", "lawngreen", "darkviolet"]

    for idx, _ in gdf.iterrows():
        if gdf.loc[idx, "label"] not in list:
            gdf.loc[idx, "label"] = "others"

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, column="label", legend=True, aspect=1)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    leg = ax.get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((1.1, 1.1))
    plt.savefig(os.path.join(cfg.plot_dir, "agri2.png"), )
    plt.close()
    pass

def create_plots_from_tif():
    # Load the raster data
    tif_path = os.path.join(cfg.data_dir, "crop_inventory", "cropped.tif")
    csv_path = os.path.join(cfg.data_dir, "crop_inventory", "aci_crop_classifications_iac_classifications_des_cultures.csv")

    with rasterio.open(tif_path) as src:
        raster_data_band1 = src.read(1)
        raster_data_band2 = src.read(2)
        raster_data_band3 = src.read(3)
        transform = src.transform
        crs = src.crs

    # Load the CSV data with labels
    csv_data = pd.read_csv(csv_path, encoding='ISO-8859-1')
    csv_data.rename(columns={"Code": "class_code", "Label": "label"}, inplace=True)\

    raster_data = np.stack([raster_data_band1, raster_data_band2, raster_data_band3], axis=-1)

    # Extract unique colors for segmentation
    unique_colors = np.unique(raster_data.reshape(-1, raster_data.shape[2]), axis=0)

    # Separate the farm boundaries and plot areas based on color codes
    farm_color = [0, 0, 0]  # Assuming farm boundaries are marked with white color (255, 255, 255)
    plot_colors = [color for color in unique_colors if not np.array_equal(color, farm_color)]

    # Initialize lists to hold farm and plot geometries
    farms = []
    plots = []

    # Create masks for farms and plots, and generate polygons using contours
    for color in plot_colors:
        mask = np.all(raster_data == color, axis=-1)
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            poly_coords = [(transform * (coord[1], coord[0])) for coord in contour]
            plot_polygon = Polygon(poly_coords)
            if plot_polygon.is_valid:
                # Find the label for this plot color based on RGB values
                label_row = csv_data[
                    (csv_data['Red'] == color[0]) & (csv_data['Green'] == color[1]) & (csv_data['Blue'] == color[2])
                    ]
                label = label_row['label'].values[0] if not label_row.empty else "Unknown"
                plots.append({"geometry": plot_polygon, "label": label})

    plot_gdf = gpd.GeoDataFrame(plots, crs=crs)

    # Combine farms and plots into a single GeoJSON-like structure
    result_geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    # Add plot features
    for idx, plot in plot_gdf.iterrows():
        plot_feature = {
            "type": "Feature",
            "properties": {
                "label": plot["label"]
            },
            "geometry": plot["geometry"].__geo_interface__
        }
        result_geojson["features"].append(plot_feature)

    # Save the result to a GeoJSON file
    output_geojson_path = os.path.join(cfg.data_dir, "crop_inventory", "plots.geojson")
    with open(output_geojson_path, "w") as geojson_file:
        json.dump(result_geojson, geojson_file)


def are_adjacent(p1, p2):
    return p1.touches(p2) or p1.intersects(p2)


def create_farm_polygons():
    plots_geojson_path = os.path.join(cfg.data_dir, "crop_inventory", "plots_m.geojson")
    with open(plots_geojson_path, 'r') as f:
        plot_data = json.load(f)

    plot_polygons = []
    plot_labels = []
    for feature in plot_data['features']:
        plot_polygons.append(shape(feature['geometry']))
        plot_labels.append(feature["properties"]["label"])
    polygon_tree = STRtree(plot_polygons)

    # Group adjacent polygons into farms
    farms = []
    used = set()

    for i, feature in enumerate(plot_data['features']):
        if i+1 in used:
            continue

        crop_list = ["Barley", "Canola/rapeseed", "Corn", "Oats", "Soybeans", "Spring wheat"]

        polygon = shape(feature['geometry'])
        label = feature["properties"]["label"]

        if label not in crop_list:
            continue

        current_farm = [{"pol": polygon, "label": label, "plot_id": i+1}]
        used.add(i)

        # Find all potential adjacent polygons using the R-tree
        potential_neighbors = polygon_tree.query(polygon)

        for neighbor_idx in potential_neighbors:
            neighbor_pol = plot_polygons[neighbor_idx]
            neighbor_label = plot_labels[neighbor_idx]
            if neighbor_idx+1 not in used and are_adjacent(polygon, neighbor_pol) and neighbor_label in crop_list:
                current_farm.append({"pol": neighbor_pol, "label": neighbor_label, "plot_id": neighbor_idx+1})
                used.add(neighbor_idx+1)
            if len(current_farm) == 5:
                    break

        # If the farm has fewer than 40 polygons, merge with another nearby farm
        cf_plot_pols = [cf["pol"] for cf in current_farm]
        cf_plot_labels = [cf["label"] for cf in current_farm]
        cf_plot_ids = [cf["plot_id"] for cf in current_farm]
        if len(current_farm) < 2:
            merged = False
            for idx, farm in enumerate(farms):
                if any(are_adjacent(cf["pol"], farm["pol"]) for cf in current_farm):
                    # Merge with an existing farm
                    if len(farm["plot_pols"]) > 5:
                        continue

                    farms[idx]["pol"] = unary_union([farm["pol"]] + cf_plot_pols)
                    for ip, plot_pol in enumerate(cf_plot_pols):
                        farms[idx]["plot_pols"].append(plot_pol)
                        farms[idx]["plot_labels"].append(cf_plot_labels[ip])
                        farms[idx]["plot_ids"].append(cf_plot_ids[ip])

                    merged = True
                    break
            if not merged:
                farm_pol = unary_union(cf_plot_pols)
                farms.append({"pol": farm_pol, "plot_pols": cf_plot_pols, "plot_labels": cf_plot_labels, "plot_ids": cf_plot_ids})
        else:
            farm_pol = unary_union(cf_plot_pols)
            farms.append({"pol": farm_pol, "plot_pols": cf_plot_pols, "plot_labels": cf_plot_labels, "plot_ids": cf_plot_ids})

    farm_polygons = [farm["pol"] for farm in farms]
    farm_rtree = STRtree(farm_polygons)

    # Locate habitat plots that are inside a farm or within a certain distance of a farm
    habitat_labels = ["Broadleaf", "Coniferous", "Exposed land/barren", "Grassland", "Shrubland", "Water",
                      "Wetland"]
    farm_neighbors = [[] for _ in farms]
    distance_threshold = 200  # Euclidean distance threshold in meters

    for i, feature in enumerate(plot_data['features']):
        label = feature["properties"]["label"]
        if label not in habitat_labels:
            continue

        habitat_polygon = shape(feature['geometry'])
        search_area = habitat_polygon.buffer(distance_threshold)
        candidate_farm_ids = farm_rtree.query(search_area)
        for farm_idx in candidate_farm_ids:
            candidate_farm_poly = farm_polygons[farm_idx]
            if (habitat_polygon.within(candidate_farm_poly) or
                    habitat_polygon.distance(candidate_farm_poly) <= distance_threshold):
                if len(farm_neighbors[farm_idx]) < 5:
                    farm_neighbors[farm_idx].append(i + 1)

    # Generate new GeoJSON features for each farm
    farm_features = []
    for idx, farm_dict in enumerate(farms):
        farm_features.append({
            "type": "Feature",
            "geometry": farm_dict["pol"].__geo_interface__,
            "properties": {
                "farm_id": idx + 1,
                "nb_hab_plot_ids": farm_neighbors[idx],
                "plot_ids": [int(j) for j in farm_dict["plot_ids"]]
            }
        })

    # Create the new farm-level GeoJSON structure
    farm_geojson = {
        "type": "FeatureCollection",
        "features": farm_features
    }

    # Save the new GeoJSON file
    output_path = os.path.join(cfg.data_dir, "crop_inventory", "farms_m_s.geojson")
    with open(output_path, 'w') as f:
        json.dump(farm_geojson, f)


def combine_habitat_polygons():
    # Load farms.geojson
    farms_geojson_path = os.path.join(cfg.data_dir, "crop_inventory", "farms_m_s.geojson")
    with open(farms_geojson_path, 'r') as f:
        farms_data = json.load(f)

    # Load plots.geojson
    plots_geojson_path = os.path.join(cfg.data_dir, "crop_inventory", "plots_m.geojson")
    with open(plots_geojson_path, 'r') as f:
        plots_data = json.load(f)

    # Prepare habitat polygons and labels
    plot_polygons = []
    plot_labels = []
    plot_ids = []
    for i, feature in enumerate(plots_data['features']):
        plot_polygons.append(shape(feature['geometry']))
        plot_labels.append(feature['properties']['label'])
        plot_ids.append(int(i+1))

    habitat_geojson_features = []
    habitat_id = 1

    # Iterate through farms to find habitat plots
    for farm in farms_data['features']:
        nb_hab_plot_ids = farm['properties']['nb_hab_plot_ids']
        if not nb_hab_plot_ids:
            continue

        # Extract habitat polygons and labels
        habitat_polygons = []
        habitat_labels = []
        habitat_plot_ids = []

        for plot_id in nb_hab_plot_ids:
            idx = plot_ids.index(plot_id)
            habitat_polygons.append(plot_polygons[idx])
            habitat_labels.append(plot_labels[idx])
            habitat_plot_ids.append(plot_id)

        # Combine habitat polygons if they are touching and have the same label
        used = set()
        farm_habitat_ids = []

        for i, habitat_polygon in enumerate(habitat_polygons):
            if i in used:
                continue

            current_label = habitat_labels[i]
            current_group = [habitat_polygon]
            used.add(i)

            for j in range(i + 1, len(habitat_polygons)):
                if j in used:
                    continue
                if habitat_labels[j] == current_label and habitat_polygon.touches(habitat_polygons[j]):
                    current_group.append(habitat_polygons[j])
                    used.add(j)

            # Create a combined polygon and add it to habitat_geojson
            combined_polygon = unary_union(current_group)
            habitat_geojson_features.append({
                "type": "Feature",
                "geometry": combined_polygon.__geo_interface__,
                "properties": {
                    "habitat_id": habitat_id,
                    "label": current_label
                }
            })
            farm_habitat_ids.append(habitat_id)
            habitat_id += 1

        # Add farm_habitat_ids to farms data
        farm['properties']['nb_hab_ids'] = farm_habitat_ids

    # Create habitat GeoJSON
    habitat_geojson = {
        "type": "FeatureCollection",
        "features": habitat_geojson_features
    }

    # Save habitat GeoJSON
    habitat_geojson_path = os.path.join(cfg.data_dir, "crop_inventory", "habitats_m_s.geojson")
    with open(habitat_geojson_path, 'w') as f:
        json.dump(habitat_geojson, f)

    # Save updated farms GeoJSON
    with open(farms_geojson_path, 'w') as f:
        json.dump(farms_data, f)


def get_farm_geojson(farm_id, data):
    if data is None:
        # Load farms.geojson
        farms_geojson_path = os.path.join(cfg.data_dir, "crop_inventory", "farms_m.geojson")
        with open(farms_geojson_path, 'r') as f:
            farms_data = json.load(f)

        # Load plots.geojson
        plots_geojson_path = os.path.join(cfg.data_dir, "crop_inventory", "plots_m.geojson")
        with open(plots_geojson_path, 'r') as f:
            plots_data = json.load(f)

        # Load habitats.geojson
        habitats_geojson_path = os.path.join(cfg.data_dir, "crop_inventory", "habitats_m.geojson")
        with open(habitats_geojson_path, 'r') as f:
            habitats_data = json.load(f)
    else:
        farms_data = data["farms"]
        plots_data = data["plots"]
        habitats_data = data["habitats"]

    # Find the farm with the given farm_id
    farm = next((f for f in farms_data['features'] if f['properties']['farm_id'] == farm_id), None)
    if not farm:
        raise ValueError(f"Farm with id {farm_id} not found.")

    # Extract farm polygon
    farm_polygon = farm['geometry']

    # Extract agricultural plots
    plot_ids = farm['properties']['plot_ids']
    farm_plots = [
        {**feature, "properties": {**feature["properties"], "type": "ag_plot"}}
        for i, feature in enumerate(plots_data['features']) if i+1 in plot_ids
    ]

    # Extract habitat polygons
    habitat_ids = farm['properties'].get('nb_hab_ids', [])
    farm_habitats = [
        {**feature, "properties": {**feature["properties"], "type": "hab_plots"}}
        for feature in habitats_data['features'] if feature['properties']['habitat_id'] in habitat_ids
    ]

    # Create the farm-level GeoJSON structure
    farm_geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": farm_polygon, "properties": {"type": "farm"}}
        ] + farm_plots + farm_habitats
    }

    # Save the new GeoJSON file
    path = os.path.join(cfg.data_dir, "crop_inventory", "farms_s", "farm_" + str(farm_id))
    output_path = os.path.join(path, f"farm_{farm_id}.geojson")
    with open(output_path, 'w') as f:
        json.dump(farm_geojson, f)


def create_combined_farms_gif():
    images = []

    for farm_id in range(1, 2404):
        # Load the created farm geojson as a GeoDataFrame
        path = os.path.join(cfg.data_dir, "crop_inventory", "farms", "farm_" + str(farm_id))
        farm_geojson_path = os.path.join(path, f"farm_{farm_id}.geojson")
        try:
            gdf = gpd.read_file(farm_geojson_path)
        except Exception as e:
            continue

        # Plot the farm and save it as an image
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(ax=ax, column="label", legend=True, aspect=1)
        plt.axis('off')
        plt.title(str(farm_id))
        leg = ax.get_legend()
        if leg is not None:
            leg.set_bbox_to_anchor((1.1, 1.1))
        img_path = os.path.join(path, f"farm_{farm_id}.png")
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()

        # Open the image and append to the list of images
        images.append(Image.open(img_path))

    # Create a GIF from the collected images
    gif_path = os.path.join(cfg.data_dir, "crop_inventory", "combined_farms.gif")
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, disposal=2, loop=0)


def calculate_yield_for_farm(farm_id, biomass_gdf):
    # Load the farm GeoJSON
    path = os.path.join(cfg.data_dir, "crop_inventory", "farms_s", "farm_" + str(farm_id))
    farm_geojson_path = os.path.join(path, f"farm_{farm_id}.geojson")
    farm_gdf = gpd.read_file(farm_geojson_path)

    # Yield mapping for specific crops
    crop_yield_mapping = {
        "Barley": "CROP_BARLEY_YLD",
        "Canola/rapeseed": "CROP_CANOLA_YLD",
        "Corn": "CROP_CORN_YLD",
        "Flaxseed": "CROP_FLAX_YLD",
        "Oats": "CROP_OAT_YLD",
        "Soybeans": "CROP_SOYBEAN_YLD",
        "Spring wheat": "CROP_WHEAT_YLD"
    }

    # Iterate through each agricultural plot and assign yield values
    for idx, plot in farm_gdf.iterrows():
        if plot['type'] != 'ag_plot':
            continue

        label = plot['label']
        plot_polygon = plot['geometry']

        # Find the biomass polygon that contains the plot polygon
        containing_biomass = biomass_gdf[biomass_gdf.contains(plot_polygon)]

        yield_value = 0
        if not containing_biomass.empty:
            biomass_row = containing_biomass.iloc[0]
            if label in crop_yield_mapping:
                yield_column = crop_yield_mapping[label]
                yield_value = biomass_row.get(yield_column, 0) or 0
            else:
                # Calculate average yield for other crops
                yield_values = [biomass_row.get(val, 0) or 0 for val in crop_yield_mapping.values()]
                yield_value = sum(yield_values) / len(yield_values)

        # Convert yield to Tonnes/Hectare
        yield_value /= 1000
        farm_gdf.at[idx, 'yield'] = yield_value

    # Save the updated farm GeoJSON
    farm_gdf.to_file(farm_geojson_path, driver='GeoJSON')


def convert(geojson_data):
    if geojson_data is None:
        farm_geojson_path = os.path.join(cfg.data_dir, "crop_inventory", "plots_init.geojson")
        with open(farm_geojson_path, 'r') as f:
            geojson_data = json.load(f)

    proj_string = """
    +proj=aea
    +lat_1=44.75
    +lat_2=55.75
    +lat_0=40
    +lon_0=-96
    +x_0=0
    +y_0=0
    +datum=WGS84
    +units=m
    +no_defs
    """

    source_crs = CRS.from_epsg(4326)
    target_crs = CRS.from_proj4(proj_string)

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    def convert_meters_to_lat_long(geojson):
        if geojson['type'] == 'FeatureCollection':
            for feature in geojson['features']:
                if feature['geometry']['type'] == 'Polygon':
                    coordinates = feature['geometry']['coordinates']
                    # Transform coordinates to lat-long format
                    lat_long_coordinates = [[transformer.transform(point[0], point[1]) for point in polygon] for polygon
                                            in coordinates]
                    feature['geometry']['coordinates'] = lat_long_coordinates
                elif feature['geometry']['type'] == 'MultiPolygon':
                    coordinates = feature['geometry']['coordinates']
                    lat_long_coordinates = [
                        [transformer.transform(point[0], point[1]) for point in polygon] for polygon in coordinates[0]
                    ]
                    feature['geometry']['coordinates'] = lat_long_coordinates
        return geojson

    # Convert the GeoJSON from meters to latitude-longitude format
    converted_geojson_meters = convert_meters_to_lat_long(geojson_data)

    if geojson_data is None:
        # Save the updated GeoJSON
        output_path_meters = os.path.join(cfg.data_dir, "crop_inventory", "plots_ll.geojson")
        with open(output_path_meters, 'w') as f:
            json.dump(converted_geojson_meters, f)
    else:
        return geojson_data


def filter_small(cfg, farm_id):
    # Load GeoJSON
    path = os.path.join(cfg.data_dir, "crop_inventory", "farms_s", "farm_" + str(farm_id))

    geojson_path = os.path.join(path, "farm_" + str(farm_id) + ".geojson")
    with open(geojson_path) as f:
        geojson_data = json.load(f)

    # Extract polygons, calculate areas, and retain properties
    areas = []
    polygons = []
    properties = []
    for feature in geojson_data['features']:
        geom = shape(feature['geometry'])
        if geom.is_valid and geom.geom_type == 'Polygon':
            polygons.append(geom)
            areas.append(geom.area)
            properties.append(feature['properties'])

    # Convert areas to numpy array for easy manipulation
    areas_np = np.array(areas)

    # Calculate 5th percentile
    threshold = np.percentile(areas_np, 25)

    # Filter polygons that are larger than the 5th percentile
    filtered_data = [(poly, prop) for poly, prop, area in zip(polygons, properties, areas) if area > threshold]

    # Separate each property into its own column and create GeoDataFrame
    filtered_properties = {key: [] for key in properties[0].keys()}
    for _, prop in filtered_data:
        for key, value in prop.items():
            filtered_properties[key].append(value)

    filtered_gdf = gpd.GeoDataFrame({**filtered_properties, 'geometry': [item[0] for item in filtered_data]})

    try:
        # Plot the filtered polygons
        fig, ax = plt.subplots(figsize=(10, 10))
        filtered_gdf.plot(ax=ax, column="label", legend=True, aspect=1)
        plt.axis('off')
        leg = ax.get_legend()
        if leg is not None:
            leg.set_bbox_to_anchor((1.1, 1.1))
        plt.savefig(os.path.join(path, "farm_" + str(farm_id) + "_red.png"))
    except Exception as e:
        pass

    # Save the filtered GeoJSON
    filtered_geojson = {
        'type': 'FeatureCollection',
        'features': []
    }
    for poly, prop in filtered_data:
        filtered_geojson['features'].append({
            'type': 'Feature',
            'geometry': json.loads(gpd.GeoSeries([poly]).to_json())['features'][0]['geometry'],
            'properties': prop
        })

    with open(os.path.join(path, "farm_" + str(farm_id) + "_red.geojson"), 'w') as f:
        json.dump(filtered_geojson, f)


def fill_polygon_gaps(cfg, farm_id):
    path = os.path.join(cfg.data_dir, "crop_inventory", "farms_s", "farm_" + str(farm_id))
    filtered_geojson_path = os.path.join(path, "farm_" + str(farm_id) + "_red.geojson")

    # Load filtered GeoJSON
    with open(filtered_geojson_path, 'r') as f:
        geojson_data = json.load(f)

    # Extract polygons and their properties
    polygons = []
    properties = []
    for feature in geojson_data['features']:
        geom = shape(feature['geometry'])
        if geom.is_valid and geom.geom_type == 'Polygon':
            polygons.append(geom)
            properties.append(feature['properties'])

    # Create a spatial index for the polygons
    #polygon_tree = STRtree(polygons)

    # Find gaps between polygons and fill them
    all_polygons_union = unary_union(polygons)
    filled_polygons = []
    filled_properties = []

    if isinstance(all_polygons_union, MultiPolygon):
        # Find holes in the MultiPolygon that need to be filled
        for poly in all_polygons_union.geoms:

            for interior in poly.interiors:
                interior_poly = Polygon(interior)

                # Ensure there is no significant overlap with existing polygons
                overlaps = [existing_poly for existing_poly in polygons if interior_poly.intersects(existing_poly)]
                for overlap in overlaps:
                    interior_poly = interior_poly.difference(overlap)

                if interior_poly.is_empty:
                    continue

                potential_neighbors = [polygons[idx] for idx in range(len(polygons)) if
                                       polygons[idx].touches(interior_poly)]
                neighbor_labels = [properties[idx]['label'] for idx in range(len(polygons)) if
                                   polygons[idx] in potential_neighbors]
                majority_label = Counter(neighbor_labels).most_common(1)[0][0]

                # Assign the majority label and merge with touching polygons of the same label
                touching_polygons = [poly for poly in potential_neighbors if
                                     properties[polygons.index(poly)]['label'] == majority_label]
                merged_polygon = cascaded_union([interior_poly] + touching_polygons)
                new_properties = properties[polygons.index(touching_polygons[0])].copy()
                filled_polygons.append(merged_polygon)
                filled_properties.append(new_properties)
    else:
        #filled_polygons.append(all_polygons_union)
        #filled_properties.append({'label': properties[0]['label'] if properties else 'Unknown'})
        #print("here")
        pass

    # Add original polygons
    for poly, prop in zip(polygons, properties):
        filled_polygons.append(poly)
        filled_properties.append(prop.copy())

    # Save the filled GeoJSON
    filled_geojson = {
        'type': 'FeatureCollection',
        'features': []
    }
    for poly, prop in zip(filled_polygons, filled_properties):
        filled_geojson['features'].append({
            'type': 'Feature',
            'geometry': json.loads(gpd.GeoSeries([poly]).to_json())['features'][0]['geometry'],
            'properties': prop
        })

    output_geojson_path = os.path.join(path, "farm_" + str(farm_id) + "_filled.geojson")

    with open(output_geojson_path, 'w') as f:
        json.dump(filled_geojson, f)

    filled_geojson = gp.read_file(output_geojson_path)

    try:
        # Plot the farm and save it as an image
        fig, ax = plt.subplots(figsize=(10, 10))
        filled_geojson.plot(ax=ax, column="label", legend=True, aspect=1)
        plt.axis('off')
        leg = ax.get_legend()
        if leg is not None:
            leg.set_bbox_to_anchor((1.1, 1.1))
        img_path = os.path.join(path, "farm_" + str(farm_id) + "_filled.png")
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        pass


def geojson_concise(path, geojson_path, drop_geometry=False):
    def load_geojson(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    gdf = gp.read_file(geojson_path)

    # Add a unique ID to each feature
    if "id" not in gdf.columns:
        gdf["id"] = range(len(gdf))
        gdf.to_file(geojson_path)

    farm_geojson = load_geojson(geojson_path)

    # Calculate neighbors for each feature
    neighbors = []
    for idx, feature in gdf.iterrows():
        feature_neighbors = gdf[gdf.geometry.touches(feature.geometry)]["id"].tolist()
        neighbors.append(feature_neighbors)
    gdf["nbs"] = neighbors

    # Remove geometry from each feature
    try:
        if "habitat_id" in gdf.columns:
            gdf.drop(columns=["habitat_id"], inplace=True)

        if "geometry" in gdf.columns:
            gdf.drop(columns=["geometry"], inplace=True)

        # Convert back to GeoJSON
        geojson_result = {
            "type": "FeatureCollection",
            "features": []
        }

        for _, row in gdf.iterrows():
            if row["label"] is None:
                continue
            id = row["id"]

            if not drop_geometry:
                geom = [feat["geometry"] for feat in farm_geojson["features"] if feat["properties"]["id"] == id][0]
            else:
                geom = None

            feature = {
                "type": "Feature",
                "geometry": geom,
                "properties": row.to_dict()
            }
            geojson_result["features"].append(feature)

        output_geojson_path = os.path.join(path, "input.geojson")
        with open(output_geojson_path, 'w') as f:
            json.dump(geojson_result, f)
        if len(geojson_result["features"]) > 0:
            return 0
        else:
            return 1
    except Exception as e:
        print(e)
        return 1


def get_data():
    # Load the biomass inventory GeoJSON
    #biomass_geojson_path = os.path.join(cfg.data_dir, "biomass", "BIOMASS_INV_CT_GEOJSON.geojson")
    #with open(biomass_geojson_path, 'r') as f:
    #    biomass_data = json.load(f)

    #biomass_data = convert(biomass_data)
    converted_path = os.path.join(cfg.data_dir, "biomass", "BIOMASS_INV_CT_GEOJSON_aea.geojson")
    #with open(converted_path, 'w') as f:
    #    json.dump(biomass_data, f)

    biomass_gdf = gpd.read_file(converted_path)

    # Load farms.geojson
    farms_geojson_path = os.path.join(cfg.data_dir, "crop_inventory", "farms_m_s.geojson")
    with open(farms_geojson_path, 'r') as f:
        farms_data = json.load(f)

    # Load plots.geojson
    plots_geojson_path = os.path.join(cfg.data_dir, "crop_inventory", "plots_m.geojson")
    with open(plots_geojson_path, 'r') as f:
        plots_data = json.load(f)

    # Load habitats.geojson
    habitats_geojson_path = os.path.join(cfg.data_dir, "crop_inventory", "habitats_m_s.geojson")
    with open(habitats_geojson_path, 'r') as f:
        habitats_data = json.load(f)

    data = {}
    data["farms"] = farms_data
    data["plots"] = plots_data
    data["habitats"] = habitats_data
    return data, biomass_gdf


def run_pipeline(data, biomass_gdf):
    for farm_id in range(1, len(data["farms"]["features"]) + 1):
        if farm_id<3841:
            continue
        print("Running farm: {}".format(farm_id))

        farm_id_path = os.path.join(cfg.data_dir, "crop_inventory", "farms_s", "farm_" + str(farm_id))
        if not os.path.isdir(farm_id_path):
            os.makedirs(farm_id_path)

        get_farm_geojson(farm_id, data=data)

        calculate_yield_for_farm(farm_id, biomass_gdf)

        filter_small(cfg, farm_id)

        fill_polygon_gaps(cfg, farm_id)

        geojson_path = os.path.join(farm_id_path, "farm_" + str(farm_id) + "_filled.geojson")
        status = geojson_concise(farm_id_path, geojson_path, drop_geometry=True)
        if status:
            shutil.rmtree(farm_id_path)


def plot_nbs_geojson(farm_id):
    path = os.path.join(cfg.data_dir, "crop_inventory", "farms", "farm_" + str(farm_id))

    def load_geojson(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    def save_geojson(data, output_path):
        with open(output_path, 'w') as file:
            json.dump(data, file)

    input_geojson_path = os.path.join(path, "input.geojson")
    farm_geojson_path = os.path.join(path, "farm_" + str(farm_id) + "_filled.geojson")
    output_gt_geojson_path = os.path.join(path, "output_gt.geojson")
    output_directory = os.path.join(path, "plots_geojsons")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    input_geojson = load_geojson(input_geojson_path)
    farm_geojson = load_geojson(farm_geojson_path)
    output_gt_geojson = load_geojson(output_gt_geojson_path)

    # Create a dictionary from output_gt.geojson for quick lookup
    output_gt_lookup = {
        feature['properties']['id']: feature['properties']
        for feature in output_gt_geojson['features']
    }

    # Process each feature in input.geojson
    for feature in input_geojson['features']:
        if feature['properties']['type'] == "ag_plot":
            plot_id = feature['properties']['id']

            # Find the corresponding plot and its neighbors in farm.geojson
            farm_plot = None
            neighbors = []

            for farm_feature in farm_geojson['features']:
                if farm_feature['properties']['id'] == plot_id:
                    farm_plot = farm_feature
                elif farm_feature['properties']['id'] in feature['properties']['nbs']:
                    neighbors.append(farm_feature)

            if not farm_plot:
                continue  # Skip if central plot not found

            # Prepare central plot properties
            central_properties = farm_plot['properties'].copy()
            central_properties['class'] = "central"

            interventions = {}
            if plot_id in output_gt_lookup:
                interventions['margin_intervention'] = output_gt_lookup[plot_id].get('margin_intervention', 0)
                interventions['habitat_conversion'] = output_gt_lookup[plot_id].get('habitat_conversion', 0)
            else:
                interventions['margin_intervention'] = 0
                interventions['habitat_conversion'] = 0

            # Prepare features for output
            output_features = [
                {
                    "type": "Feature",
                    "properties": central_properties,
                    "geometry": farm_plot['geometry']
                }
            ]

            for neighbor in neighbors:
                neighbor_properties = neighbor['properties'].copy()
                neighbor_properties['class'] = "nbs"
                output_features.append({
                    "type": "Feature",
                    "properties": neighbor_properties,
                    "geometry": neighbor['geometry']
                })

            # Save to a separate GeoJSON file
            output_geojson = {
                "type": "FeatureCollection",
                "features": output_features
            }

            plot_dir = os.path.join(output_directory, f"plot_{plot_id}")
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            output_path = os.path.join(plot_dir, "input.geojson")
            intervention_path = os.path.join(plot_dir, "interventions.json")
            save_geojson(output_geojson, output_path)
            save_geojson(interventions, intervention_path)


if __name__ == "__main__":
    cfg = Config()
    #farm_ids = np.arange(9, 11)

    #create_plots_from_tif()

    #create_farm_polygons()
    #combine_habitat_polygons()

    #data, biomass_gdf = get_data()
    #run_pipeline(data, biomass_gdf)

    # convert(geojson_data=None)
    # create_combined_farms_gif()

    #plot_nbs_geojson(farm_id)

    """
    for farm_id in farm_ids:
        print(f"Running farm: {farm_id}")
        farm_id_path = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms", "farm_" + str(farm_id), "connectivity")
        geojson_path = os.path.join(farm_id_path, "input.geojson")
        geojson_concise(farm_id_path, geojson_path, drop_geometry=False)
    """



