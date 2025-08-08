import json
import math
import itertools
import os.path
from shapely.geometry import shape, Polygon, Point, LineString, MultiPolygon, MultiPoint
from shapely import voronoi_polygons
import networkx as nx
import pyomo.environ as pyo
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches # For legend handles
import matplotlib.lines as mlines
from shapely import wkt as shapely_wkt
import numpy as np
from config import Config
from scipy.spatial import KDTree
from pyomo.opt import SolverStatus, TerminationCondition


def parse_geojson(geojson_path):
    """
    Reads the interventions GeoJSON file and returns a structured list of plots.
    """
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    plots = []
    for feature in data['features']:
        props = feature['properties']
        geom = shape(feature['geometry'])  # shapely Polygon

        # Build record
        plot_record = {
            'farm_id': props.get('farm_id'),
            'plot_id': props.get('id'),
            'plot_type': props.get('type'),
            'label': props.get('label', ""),
            'yield': props.get('yield', 0.0),
            'geometry': geom,
            'margin_frac': props.get("margin_intervention", 0.0),
            'habitat_frac': props.get("habitat_conversion", 0.0)
        }
        plots.append(plot_record)

    return plots


def precompute_polygon_data(plots, n_points=5):
    """
    Precompute data needed for margin effects:
    - Polygon boundaries
    - Random points within polygons (for margin calculations)
    Storing them in dictionaries for quick access.
    """
    polygon_data = {}
    for idx, p in enumerate(plots):
        poly = p["geometry"]
        boundary = poly.boundary
        pts = poly.centroid
        """
        min_x, min_y, max_x, max_y = poly.bounds
        pts = []
        # Generate a fixed set of random points once
        while len(pts) < n_points:
            rx = np.random.uniform(min_x, max_x, n_points * 2)
            ry = np.random.uniform(min_y, max_y, n_points * 2)
            candidate_points = [Point(xy) for xy in zip(rx, ry)]
            for cpt in candidate_points:
                if len(pts) >= n_points:
                    break
                if poly.contains(cpt):
                    pts.append(cpt)
        """
        polygon_data[idx] = (boundary, pts)
    return polygon_data


def calculate_distances(centroids):
    tree = KDTree(centroids)
    distances = tree.sparse_distance_matrix(tree, max_distance=10000)
    distances = distances.toarray()
    return distances


def discretize_polygon_boundary(poly, num_segments=100):
    """
    Returns a list of boundary arcs (each an approximate linestring)
    of equal length along 'poly's boundary.
    Also returns an array of arc lengths.
    """
    boundary = poly.exterior
    total_len = boundary.length
    segment_length = total_len / num_segments

    arcs = []
    lengths = []
    for i in range(num_segments):
        start_dist = i * segment_length
        end_dist = (i + 1) * segment_length
        if end_dist > total_len:
            end_dist = total_len
        arc_coords = []
        steps = 5  # subdivide each arc for an approximate linestring
        step_len = (end_dist - start_dist) / steps
        for s in range(steps + 1):
            d = start_dist + s * step_len
            pt = boundary.interpolate(d)
            arc_coords.append(pt.coords[0])
        arc = LineString(arc_coords)
        arcs.append(arc)
        lengths.append(arc.length)
    return arcs, lengths


def discretize_polygon_interior(poly, total_cells=100):
    """
    Create ~total_cells Voronoi cells inside 'poly'.
    Steps:
      1) sample total_cells random points inside bounding box, keep those in poly
      2) build Voronoi from these points (shapely.ops.voronoi_polygons)
      3) intersect each Voronoi cell with 'poly'
      4) keep non-empty intersections as our discrete cells
    Returns:
      cells (list of Polygon),
      areas (list of float)
    """
    minx, miny, maxx, maxy = poly.bounds

    inside_points = []
    max_tries = total_cells * 10
    tries = 0
    while len(inside_points) < total_cells and tries < max_tries:
        tries += 1
        rx = random.uniform(minx, maxx)
        ry = random.uniform(miny, maxy)
        p = Point(rx, ry)
        if poly.contains(p):
            inside_points.append(p)

    if not inside_points:
        return [], []

    mp = MultiPoint(inside_points)
    try:
        vor = voronoi_polygons(mp)  # returns a MultiPolygon (Shapely >= 2.0)
    except:
        return [], []

    cells = []
    areas = []
    for cell in vor.geoms:
        clipped = cell.intersection(poly)
        if not clipped.is_empty:
            if clipped.geom_type == 'Polygon':
                cells.append(clipped)
                areas.append(clipped.area)
            elif clipped.geom_type == 'MultiPolygon':
                for subp in clipped.geoms:
                    if not subp.is_empty:
                        cells.append(subp)
                        areas.append(subp.area)

    return cells, areas


def build_reposition_ilp(plots, params, boundary_seg_count, interior_cell_count, adjacency_dist, connectivity_metric, al_factor, neib_dist):
    """
    Builds a Pyomo model (previously PuLP) that decides how to assign margin arcs
    and habitat cells for each ag_plot so that the fraction constraints are
    satisfied and the chosen layout maximizes a connectivity measure.
    Returns a tuple (model, piece_list).
    """
    polygon_data = precompute_polygon_data(plots, n_points=5)
    centroids = [p["geometry"].centroid.coords[0] for p in plots]
    distances = calculate_distances(tuple(centroids))

    # Create the model
    prob = pyo.ConcreteModel("ConnectivityRepositioning")

    piece_list = []
    adjacency_list = []

    # We'll need a dynamic counter for naming variables
    var_counter = 0

    # Step 1: build the piece_list
    for pidx, p in enumerate(plots):
        if p['plot_type'] == 'hab_plots':
            # This is a "full habitat" -> always 100% selected
            poly = p['geometry']
            area = poly.area
            piece_list.append({
                'plot_index': pidx,
                'plot_id': (p['farm_id'], p['plot_id']),
                'geom': poly,
                'type': 'full_habitat',
                'length': 0.0,
                'area': area/10000,
                'var': None  # always selected
            })
        else:
            poly = p['geometry']
            m_frac = p['margin_frac']
            h_frac = p['habitat_frac']

            if m_frac > 0:
                arcs, arc_lengths = discretize_polygon_boundary(poly, num_segments=boundary_seg_count)
                for arc, arc_len in zip(arcs, arc_lengths):
                    var_name = f"x_margin_{pidx}_{var_counter}"
                    # Create a binary var in Pyomo
                    this_var = pyo.Var(domain=pyo.Binary)
                    setattr(prob, var_name, this_var)

                    piece_list.append({
                        'plot_index': pidx,
                        'plot_id': (p['farm_id'], p['plot_id']),
                        'geom': arc,
                        'type': 'margin',
                        'length': arc_len/1000,
                        'area': 0.0,
                        'var': this_var
                    })
                    var_counter += 1

            if h_frac > 0:
                cells, cell_areas = discretize_polygon_interior(poly, total_cells=interior_cell_count)
                for cgeom, carea in zip(cells, cell_areas):
                    var_name = f"x_habitat_{pidx}_{var_counter}"
                    this_var = pyo.Var(domain=pyo.Binary)
                    setattr(prob, var_name, this_var)

                    piece_list.append({
                        'plot_index': pidx,
                        'plot_id': (p['farm_id'], p['plot_id']),
                        'geom': cgeom,
                        'type': 'habitat_patch',
                        'length': 0.0,
                        'area': carea/10000,
                        'var': this_var
                    })
                    var_counter += 1

    plot_baseline_npv = {}
    plot_types = [p["plot_type"] for p in plots]
    # Need all habitat fractions for the neighbour effect calculation in compute_plot_npv
    habitat_fracs_all = [p.get("habitat_frac", 0.0) for p in plots]

    for idx, p in enumerate(plots):
        if p['plot_type'] == 'ag_plot':
            m_f = p.get('margin_frac', 0.0)
            h_f = p.get('habitat_frac', 0.0)
            # Calculate NPV using the function, passing all required params
            npv_p = compute_plot_npv(p, m_f, h_f, idx, polygon_data, distances, plot_types, habitat_fracs_all, params,
                                     neib_dist)
            plot_baseline_npv[idx] = npv_p

    # Step 2: Add fraction constraints plot by plot
    # We'll group by plot, type='margin' and type='habitat_patch'
    from collections import defaultdict
    plot_margin_arcs = defaultdict(list)
    plot_habitat_cells = defaultdict(list)

    for i, piece in enumerate(piece_list):
        pidx = piece['plot_index']
        if piece['type'] == 'margin':
            plot_margin_arcs[pidx].append(i)
        elif piece['type'] == 'habitat_patch':
            plot_habitat_cells[pidx].append(i)

    # We'll store constraints in the model:
    # margin_frac constraints => we approximate with +/- some tolerance
    # habitat_frac constraints => likewise
    prob.constraints = pyo.ConstraintList()

    for pidx, p in enumerate(plots):
        if p['plot_type'] == 'ag_plot':
            poly = p['geometry']
            perim = poly.length/1000
            area = poly.area/10000
            m_frac = p['margin_frac']
            h_frac = p['habitat_frac']

            if m_frac > 0 and pidx in plot_margin_arcs:
                arc_indices = plot_margin_arcs[pidx]

                # Lower bound ( >= )
                c_name_low = f"MarginFracDown_{pidx}"
                expr_low = sum(piece_list[ii]['length'] * piece_list[ii]['var']
                               for ii in arc_indices) \
                           >= ((m_frac * perim) - ((1 / boundary_seg_count) * perim))
                prob.constraints.add(expr_low)

                # Upper bound ( <= )
                c_name_up = f"MarginFracUp_{pidx}"
                expr_up = sum(piece_list[ii]['length'] * piece_list[ii]['var']
                              for ii in arc_indices) \
                          <= ((m_frac * perim) + ((1 / boundary_seg_count) * perim))
                prob.constraints.add(expr_up)

            if h_frac > 0 and pidx in plot_habitat_cells:
                cell_indices = plot_habitat_cells[pidx]

                expr_low_hab = sum(piece_list[ii]['area'] * piece_list[ii]['var']
                                   for ii in cell_indices) \
                               >= ((h_frac * area) - ((1 / interior_cell_count) * area))
                prob.constraints.add(expr_low_hab)

                expr_up_hab = sum(piece_list[ii]['area'] * piece_list[ii]['var']
                                  for ii in cell_indices) \
                              <= ((h_frac * area) + ((1 / interior_cell_count) * area))
                prob.constraints.add(expr_up_hab)

    # Step 3: Build adjacency among pieces
    for i, j in itertools.combinations(range(len(piece_list)), 2):
        pi = piece_list[i]
        pj = piece_list[j]
        # check distance
        if pi['geom'].distance(pj['geom']) <= adjacency_dist:
            dist = pi['geom'].distance(pj['geom'])
            adjacency_list.append((i, j, dist))

    # Step 4: Define the objective for the chosen connectivity_metric
    # We'll do a linear combination: sum_i( x_i * piece_score[i] ) + sum_(i,j)( y_ij * adjacency_weight(i,j) )
    # We'll need y_ij in the model
    prob.y_vars = {}
    prob.constraints_adjacency = pyo.ConstraintList()
    adjacency_weight = {}

    # piece_score array
    piece_score = [0.0] * len(piece_list)
    for i, piece in enumerate(piece_list):
        if piece['type'] == 'full_habitat':
            piece_score[i] = piece['area']
        elif piece['type'] == 'habitat_patch':
            piece_score[i] = piece['area']
        elif piece['type'] == 'margin':
            piece_score[i] =  piece['length']
        else:
            piece_score[i] = 0.0

    for (i, j, dist) in adjacency_list:
        y_name = f"y_{i}_{j}"
        y_ij = pyo.Var(domain=pyo.Binary)
        setattr(prob, y_name, y_ij)
        prob.y_vars[(i, j)] = y_ij

        xi = piece_list[i]['var']
        xj = piece_list[j]['var']

        # adjacency_weight depends on connectivity_metric
        if dist < 1e-9:
            dist = 1e-9

        if connectivity_metric == 'LCC':
            adjacency_weight[(i, j)] = 5.0
        elif connectivity_metric == 'IIC':
            ai = piece_list[i]['area']
            li = piece_list[i]['length']
            aj = piece_list[j]['area']
            lj = piece_list[j]['length']
            if piece_list[i]['type'] == "full_habitat" or piece_list[j]['type'] == "full_habitat":
                adjacency_weight[(i, j)] = (li * lj) + (ai * aj) + al_factor * (ai * lj + aj * li)
            else:
                adjacency_weight[(i, j)] = (li * lj) + (ai * aj) + (ai * lj) + (aj * li)
        elif connectivity_metric == 'PC':
            alpha = 0.05
            ai = piece_list[i]['area'] + piece_list[i]['length']
            aj = piece_list[j]['area'] + piece_list[j]['length']
            w_ij = ai * aj * pyo.exp(-alpha * dist)
            adjacency_weight[(i, j)] = w_ij
        else:
            adjacency_weight[(i, j)] = 5.0

        # Now set constraints for y_ij
        # if both var are not None
        if xi is not None and xj is not None:
            prob.constraints_adjacency.add(y_ij <= xi)
            prob.constraints_adjacency.add(y_ij <= xj)
            prob.constraints_adjacency.add(y_ij >= xi + xj - 1)
        else:
            # handle cases with full_habitat
            if piece_list[i]['type'] == 'full_habitat' and xj is not None:
                # y_ij == xj
                prob.constraints_adjacency.add(y_ij <= xj)
                prob.constraints_adjacency.add(y_ij >= xj)
            elif piece_list[j]['type'] == 'full_habitat' and xi is not None:
                prob.constraints_adjacency.add(y_ij <= xi)
                prob.constraints_adjacency.add(y_ij >= xi)
            elif piece_list[i]['type'] == 'full_habitat' and piece_list[j]['type'] == 'full_habitat':
                # both always 1
                prob.constraints_adjacency.add(y_ij == 1)

    # Build objective expression
    obj_expr = []
    for i, piece in enumerate(piece_list):
        xi = piece['var']
        sc = piece_score[i]
        if piece['type'] == 'full_habitat':
            # always selected
            obj_expr.append(sc)
        else:
            obj_expr.append(sc * xi)

    for (i, j, dist) in adjacency_list:
        y_ij = prob.y_vars[(i, j)]
        w_ij = adjacency_weight[(i, j)]
        # handle full_habitat logic
        if piece_list[i]['type'] == 'full_habitat' and piece_list[j]['var'] is not None:
            # effectively y_ij = x_j
            pass
        elif piece_list[j]['type'] == 'full_habitat' and piece_list[i]['var'] is not None:
            pass
        obj_expr.append(w_ij * y_ij)

    prob.obj = pyo.Objective(expr=sum(obj_expr), sense=pyo.maximize)

    return prob, piece_list, plot_baseline_npv


def solve_reposition_ilp(plots, params, adjacency_dist, boundary_seg_count, interior_cell_count, connectivity_metric, al_factor, neib_dist):
    """
    Build and solve the Pyomo model that repositions the existing margin/habitat
    fractions for each plot to maximize connectivity.
    Return the chosen geometry (which arcs/cells are selected).
    """

    prob, piece_list, plot_repos_npv = build_reposition_ilp(plots, params, boundary_seg_count, interior_cell_count, adjacency_dist,
                                            connectivity_metric, al_factor, neib_dist)

    solver = pyo.SolverFactory('cbc')
    solver.options['ratio'] = 0.001
    results = solver.solve(prob, tee=False)

    chosen_pieces = []
    if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:
        for piece in piece_list:
            var = piece['var']
            if piece['type'] == 'full_habitat':
                chosen = True
            else:
                if var is not None:
                    var_val = pyo.value(var)
                    chosen = (var_val > 0.5)
                else:
                    chosen = False

            if chosen:
                chosen_pieces.append(piece)

        G = build_connectivity_graph_from_chosen_pieces(chosen_pieces, adjacency_dist=adjacency_dist)
        conn_val = compute_connectivity_metric(G, connectivity_metric)
        val = pyo.value(prob.obj)
    else:
        val, conn_val = 0, 0
    return chosen_pieces, val, conn_val, plot_repos_npv


def compute_time_factors(params):
    """Returns array [1/(1+r)^1, 1/(1+r)^2, ..., 1/(1+r)^T]."""
    t_arr = np.arange(1, params['t'] + 1)
    df = (1 + params['r']) ** (-t_arr)

    gamma_values = set()
    for cr in params['crops'].values():
        gamma_values.add(cr['margin']['gamma'])
        gamma_values.add(cr['habitat']['gamma'])
        gamma_values.add(cr['margin']['zeta'])
        gamma_values.add(cr['habitat']['zeta'])

    time_factor = {}
    for g in gamma_values:
        time_factor[g] = (1 - np.exp(-g * t_arr))
    return df, time_factor


def combined_yield_factor(margin_f, crop_params, idx, polygon_data, distances, time_factor, plot_types,
                          habitat_fracs, neib_dist):

    def calculate_margin_effect(idx_m, alpha, beta, gamma, fraction, neib_dist):
        indices_to_consider = np.where(distances[idx_m, :] < neib_dist)[0]
        effects = []
        indices_to_consider = [idx_m] + list(set(indices_to_consider) - set([idx_m]))
        for n_idx in indices_to_consider:
            if n_idx == idx_m:
                boundary, centroid = polygon_data[n_idx]
            else:
                _, centroid = polygon_data[n_idx]
            dist = centroid.distance(boundary)
            effect = fraction * alpha * np.exp(-beta * dist) * time_factor[gamma]
            effects.append(effect)
        return np.sum(effects, axis=0)

    # margin pollination/pest from the same plot:
    pollination_services_list = [
        calculate_margin_effect(idx,
                            crop_params['margin']['alpha'],
                            crop_params['margin']['beta'],
                            crop_params['margin']['gamma'], margin_f, neib_dist)
    ]
    pest_control_services_list = [
        calculate_margin_effect(idx,
                             crop_params['margin']['delta'],
                             crop_params['margin']['epsilon'],
                             crop_params['margin']['zeta'], margin_f, neib_dist)
    ]

    # habitat pollination/pest from other plots
    crop_habitat_params = crop_params['habitat']
    other_indices = np.where(distances[idx, :] < neib_dist)[0]
    dists = np.array([distances[idx, o] for o in other_indices])
    oth_types = [pt for i_pt, pt in enumerate(plot_types) if i_pt in other_indices]

    habitat_values = []
    for idd, tp in enumerate(oth_types):
        if tp == 'hab_plots':
            habitat_values.append(1.0)
        elif tp == 'ag_plot':
            habitat_values.append(habitat_fracs[idd])
        else:
            habitat_values.append(0.0)

    habitat_values = np.array(habitat_values)
    mask = habitat_values > 0
    if np.any(mask):
        hv = habitat_values[mask]
        dd = dists[mask]
        for hv_i, dd_i in zip(hv, dd):
            pollination_services_list.append(
                hv_i * crop_habitat_params['alpha'] * np.exp(
                    -crop_habitat_params['beta'] * dd_i) * time_factor[crop_habitat_params['gamma']]
            )
            pest_control_services_list.append(
                hv_i * crop_habitat_params['delta'] * np.exp(
                    -crop_habitat_params['epsilon'] * dd_i) * time_factor[
                    crop_habitat_params['zeta']]
            )

    total_pollination = np.sum(pollination_services_list, axis=0)
    total_pest = np.sum(pest_control_services_list, axis=0)
    yield_factor = 1 + total_pollination + total_pest
    return yield_factor


def compute_plot_npv(plot_dict, margin_f, habitat_f, idx, polygon_data, distances, plot_types, habitat_fracs,
                     params, neib_dist):
    """
    Very simplified example that computes NPV for one plot:
      - Implementation cost for margin/hab (one-time).
      - Annual yield: yield_t * price * area.
      - Annual maintenance cost: baseline + margin/hab portion.
    """
    # Extract discount
    crop_label = plot_dict.get('label')
    crop_params = params['crops'][crop_label]
    area = plot_dict["geometry"].area/10000

    df, time_factor = compute_time_factors(params)

    impl = (margin_f * params['costs']['margin']['implementation']
            + habitat_f * params['costs']['habitat']['implementation']) * area

    base_yield = plot_dict.get("yield", 0.0)
    y_mult = combined_yield_factor(margin_f, crop_params, idx, polygon_data, distances, time_factor,
                                   plot_types, habitat_fracs, neib_dist)
    final_yield = base_yield * y_mult

    # The line that caused issues in PuLP is just a normal arithmetic expression here:
    revenue_per_year = final_yield * crop_params['p_c'] * area * (1 - habitat_f)

    habitat_loss_yield = habitat_f * crop_params['p_c'] * area * base_yield

    maintenance = (params['costs']['agriculture']['maintenance'] * (1 - habitat_f)
                   + params['costs']['agriculture']['maintenance'] * margin_f
                   + params['costs']['habitat']['maintenance'] * habitat_f) * area

    annual_cashflow = revenue_per_year - maintenance - habitat_loss_yield
    discounted_flow = np.sum(annual_cashflow * df)
    return discounted_flow - impl


def compute_farm_baseline_npvs(plots, polygon_data, distances, params, neib_dist):
    """
    Sums up the baseline NPV for each farm separately,
    returning a dict: {farm_id: NPV_baseline}.
    """
    from collections import defaultdict
    farm_npv_map = defaultdict(float)
    plot_baseline_npv = defaultdict(float)
    plot_types = [p["plot_type"] for p in plots]
    habitat_fracs = [p["habitat_frac"] for p in plots]

    for idx, p in enumerate(plots):
        farm_id = p['farm_id']
        if p['plot_type'] == 'ag_plot':
            m_f = p.get('margin_frac', 0.0)
            h_f = p.get('habitat_frac', 0.0)
            npv_p = compute_plot_npv(p, m_f, h_f, idx, polygon_data, distances, plot_types, habitat_fracs, params, neib_dist)
            plot_baseline_npv[idx] = npv_p
            farm_npv_map[farm_id] += npv_p
        else:
            pass
    return dict(farm_npv_map), dict(plot_baseline_npv)


def calculate_margin_effect_expr(i, piece_list, dist_matrix, alpha, beta, gamma, time_factor, neib_dist):
    """
    Returns a symbolic (Pyomo) expression that represents the pollination/pest effect
    from all 'margin' pieces that are within neib_dist of piece i,
    approximating the approach from combined_yield_factor.
    """
    expr_list = []
    N = len(piece_list)

    for j in range(N):
        if i == j:
            continue
        if piece_list[j]['type'] == 'margin':
            d_ij = dist_matrix[i, j]
            if d_ij < neib_dist:
                x_j = piece_list[j]['x_var']
                frac = piece_list[j]['fraction']
                if x_j is not None:
                    expr_list.append(alpha * x_j * frac * pyo.exp(-beta * d_ij) * time_factor[gamma])

    if len(expr_list) == 0:
        return 0.0
    return sum(expr_list)


def calculate_habitat_effect_expr(i, piece_list, dist_matrix, alpha, beta, gamma, time_factor, neib_dist):
    """
    Similar approach for 'habitat' effect.
    We do not average for habitat in combined_yield_factor.
    """
    N = len(piece_list)
    #tf_sum = np.sum(time_factor[gamma])
    expr_list = []

    for j in range(N):
        if i == j:
            continue
        d_ij = dist_matrix[i, j]
        if d_ij < neib_dist:
            pj_type = piece_list[j]['type']
            if pj_type == 'full_habitat':
                # fraction_j = 1
                expr_list.append(alpha * pyo.exp(-beta * d_ij) * time_factor[gamma])
            elif pj_type == 'habitat_patch':
                x_j = piece_list[j]['x_var']
                frac = piece_list[j]['fraction']
                expr_list.append(x_j * frac * alpha * pyo.exp(-beta * d_ij) * time_factor[gamma])
    if len(expr_list) == 0:
        return 0.0
    return sum(expr_list)


def build_connectivity_ilp(plots, farm_baseline_npv, params, boundary_seg_count, interior_cell_count,
                           max_loss_ratio, adjacency_dist, al_factor, neib_dist, margin_weight):
    """
    1) Discretize each ag_plot into margin arcs + habitat cells => piece_list.
    2) For adjacency-based connectivity, define x_i for each piece (1=chosen).
    3) Summation of chosen arcs => margin fraction, chosen cells => habitat fraction.
    4) Compute a (linear or approximate) expression for each farm's new NPV
       and require newNPV_farm >= baselineNPV_farm*(1 - max_loss_ratio).
    5) Objective: maximize connectivity (IIC, LCC, PC, etc.).
    """
    prob = pyo.ConcreteModel("ConnectivityWithFarmNPV")
    piece_list = []

    var_counter = 0

    for pidx, p in enumerate(plots):
        if p['plot_type'] == 'hab_plots':
            piece_list.append({
                'plot_index': pidx,
                'farm_id': p['farm_id'],
                'geom': p['geometry'],
                'type': 'full_habitat',
                'x_var': None,
                'length': 0.0,
                'area': p['geometry'].area/10000,
                'fraction': 1
            })
        elif p['plot_type'] == 'ag_plot':
            poly = p['geometry']
            # interior:
            piece_list.append({
                'plot_index': pidx,
                'farm_id': p['farm_id'],
                'geom': poly.centroid,
                'type': 'ag_interior',
                'x_var': None,
                'length': poly.length/1000,
                'area': poly.area/10000,
                'fraction': 1
            })
            var_counter += 1

            # margin arcs
            arcs, arc_lengths = discretize_polygon_boundary(poly, boundary_seg_count)
            for arc, arc_len in zip(arcs, arc_lengths):
                var_name = f"x_margin_{pidx}_{var_counter}"
                this_var = pyo.Var(domain=pyo.Binary)
                #this_var = pyo.Var(bounds = (0, 1))
                setattr(prob, var_name, this_var)

                piece_list.append({
                    'plot_index': pidx,
                    'farm_id': p['farm_id'],
                    'geom': arc,
                    'type': 'margin',
                    'x_var': this_var,
                    'length': arc_len/1000,
                    'area': 0.0,
                    'fraction': (arc_len/1000) / (poly.length/1000)
                })
                var_counter += 1

            # habitat cells
            cells, cell_areas = discretize_polygon_interior(poly, interior_cell_count)
            for cell, area_c in zip(cells, cell_areas):
                var_name = f"x_hab_{pidx}_{var_counter}"
                this_var = pyo.Var(domain=pyo.Binary)
                #this_var = pyo.Var(bounds=(0, 1))
                setattr(prob, var_name, this_var)

                piece_list.append({
                    'plot_index': pidx,
                    'farm_id': p['farm_id'],
                    'geom': cell,
                    'type': 'habitat_patch',
                    'x_var': this_var,
                    'length': 0.0,
                    'area': area_c/10000,
                    'fraction': (area_c/10000) / (poly.area/10000)
                })
                var_counter += 1
        else:
            pass

    N = len(piece_list)
    dist_matrix = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                dist_matrix[i, j] = piece_list[i]['geom'].distance(piece_list[j]['geom'])

    from collections import defaultdict
    margin_pieces_by_plot = defaultdict(list)
    habitat_pieces_by_plot = defaultdict(list)
    ag_interior_piece_by_plot = {}

    for i, pc in enumerate(piece_list):
        pidx = pc['plot_index']
        if pc['type'] == 'margin':
            margin_pieces_by_plot[pidx].append(i)
        elif pc['type'] == 'habitat_patch':
            habitat_pieces_by_plot[pidx].append(i)
        elif pc['type'] == 'ag_interior':
            ag_interior_piece_by_plot[pidx] = i

    plot_perimeters = {}
    plot_areas = {}

    for pidx, p in enumerate(plots):
        if p['plot_type'] == 'ag_plot':
            plot_perimeters[pidx] = p['geometry'].length/1000
            plot_areas[pidx] = p['geometry'].area/10000

    prob.constraints = pyo.ConstraintList()

    df_array, time_factor = compute_time_factors(params)

    # Build expressions for each plot's NPV
    plot_npv_expr = {}

    for pidx, p in enumerate(plots):
        if p['plot_type'] != 'ag_plot':
            continue

        perimeter_p = plot_perimeters[pidx]
        area_p = plot_areas[pidx]

        margin_indices = margin_pieces_by_plot[pidx]
        if perimeter_p > 0:
            margin_frac_expr = sum(piece_list[i]['length'] * piece_list[i]['x_var']
                                   for i in margin_indices) / perimeter_p
        else:
            margin_frac_expr = 0.0

        habitat_indices = habitat_pieces_by_plot[pidx]
        if area_p > 0:
            habitat_frac_expr = sum(piece_list[i]['area'] * piece_list[i]['x_var']
                                    for i in habitat_indices) / area_p
        else:
            habitat_frac_expr = 0.0

        # Implementation cost
        margin_impl = params['costs']['margin']['implementation']
        habitat_impl = params['costs']['habitat']['implementation']
        impl_cost_expr = area_p * (margin_frac_expr * margin_impl + habitat_frac_expr * habitat_impl)

        ag_maint = params['costs']['agriculture']['maintenance']
        mar_maint = params['costs']['margin']['maintenance']
        hab_maint = params['costs']['habitat']['maintenance']

        maintenance_per_year_expr = area_p * (
                ag_maint * (1 - habitat_frac_expr)
                + mar_maint * margin_frac_expr
                + hab_maint * habitat_frac_expr
        )

        base_yield = p.get('yield', 0.0)
        crop_label = p.get('label')
        crop_params = params['crops'][crop_label]
        price = crop_params['p_c']

        if pidx not in ag_interior_piece_by_plot:
            continue

        interior_id = ag_interior_piece_by_plot[pidx]

        # margin pollination
        pollination_services_list = [
            calculate_margin_effect_expr(
                interior_id, piece_list, dist_matrix,
                alpha=crop_params['margin']['alpha'],
                beta=crop_params['margin']['beta'],
                gamma=crop_params['margin']['gamma'],
                time_factor=time_factor,
                neib_dist=neib_dist
            )
        ]
        pest_control_services_list = [
            calculate_margin_effect_expr(
                interior_id, piece_list, dist_matrix,
                alpha=crop_params['margin']['delta'],
                beta=crop_params['margin']['epsilon'],
                gamma=crop_params['margin']['zeta'],
                time_factor=time_factor,
                neib_dist=neib_dist
            )
        ]

        # habitat pollination
        crop_habitat_params = crop_params['habitat']
        pollination_services_list.append(
            calculate_habitat_effect_expr(
                interior_id, piece_list, dist_matrix,
                alpha=crop_habitat_params['alpha'],
                beta=crop_habitat_params['beta'],
                gamma=crop_habitat_params['gamma'],
                time_factor=time_factor,
                neib_dist=neib_dist
            )
        )
        pest_control_services_list.append(
            calculate_habitat_effect_expr(
                interior_id, piece_list, dist_matrix,
                alpha=crop_habitat_params['delta'],
                beta=crop_habitat_params['epsilon'],
                gamma=crop_habitat_params['zeta'],
                time_factor=time_factor,
                neib_dist=neib_dist
            )
        )

        total_pollination_expr = sum(pollination_services_list)
        total_pest_expr = sum(pest_control_services_list)
        yield_factor_expr = 1 + total_pollination_expr + total_pest_expr

        annual_revenue_expr = base_yield * yield_factor_expr * price * area_p * (1 - habitat_frac_expr)
        habitat_loss_yield_expr = habitat_frac_expr * price * area_p * base_yield

        annual_cashflow = annual_revenue_expr - maintenance_per_year_expr - habitat_loss_yield_expr

        # discounted flow sum
        # in Pyomo, we can do sum(...) for each year discount, but here let's do a direct multiplication
        discounted_flow_expr = sum(cf_i * df_i for cf_i, df_i in zip(annual_cashflow, df_array))

        npv_expr = discounted_flow_expr - impl_cost_expr
        plot_npv_expr[pidx] = npv_expr

    # Now sum up farm-level NPV, add constraints
    from collections import defaultdict
    farm_plot_map = defaultdict(list)
    for pidx, p in enumerate(plots):
        farm_plot_map[p['farm_id']].append(pidx)

    for f_id, pindices in farm_plot_map.items():
        farm_expr = sum(plot_npv_expr[pi] for pi in pindices if pi in plot_npv_expr)
        baseline_npv_farm = farm_baseline_npv.get(f_id, 0.0)
        min_allowed = (1 - max_loss_ratio) * baseline_npv_farm
        constraint_comparison = farm_expr >= min_allowed
        if constraint_comparison is True:
            continue
        else:
            prob.constraints.add(farm_expr >= min_allowed)

    # Step C: adjacency for connectivity
    adjacency_list = []
    for i, j in itertools.combinations(range(N), 2):
        if piece_list[i]['type'] == 'ag_interior' or piece_list[j]['type'] == 'ag_interior':
            continue
        dist_ij = dist_matrix[i, j]
        if dist_ij <= adjacency_dist:
            adjacency_list.append((i, j, dist_ij))

    prob.y_vars = {}
    prob.adjacency_constraints = pyo.ConstraintList()
    adjacency_weight = {}

    for (i, j, dist_ij) in adjacency_list:
        y_name = f"y_{i}_{j}"
        y_ij = pyo.Var(domain=pyo.Binary)
        #y_ij = pyo.Var(bounds=(0, 1))
        setattr(prob, y_name, y_ij)
        prob.y_vars[(i, j)] = y_ij

        pc_i = piece_list[i]
        pc_j = piece_list[j]
        xi = pc_i['x_var']
        xj = pc_j['x_var']

        ai = pc_i["geom"].area/10000
        aj = pc_j["geom"].area/10000
        li = pc_i["geom"].length/1000
        lj = pc_j["geom"].length/1000
        if pc_i["type"] == "full_habitat" or pc_j["type"] == "full_habitat":
            adjacency_weight[(i, j)] = (li * lj) + (ai * aj) + al_factor * (ai * lj + aj * li)
        else:
            adjacency_weight[(i, j)] = (li * lj) + (ai * aj) + (ai * lj) + (aj * li)

        # adjacency constraints
        if pc_i['type'] == 'full_habitat' and xj is not None:
            prob.adjacency_constraints.add(y_ij <= xj)
            prob.adjacency_constraints.add(y_ij >= xj)
        elif pc_j['type'] == 'full_habitat' and xi is not None:
            prob.adjacency_constraints.add(y_ij <= xi)
            prob.adjacency_constraints.add(y_ij >= xi)
        elif pc_i['type'] == 'full_habitat' and pc_j['type'] == 'full_habitat':
            prob.adjacency_constraints.add(y_ij == 1)
        else:
            if xi is not None and xj is not None:
                prob.adjacency_constraints.add(y_ij <= xi)
                prob.adjacency_constraints.add(y_ij <= xj)
                prob.adjacency_constraints.add(y_ij >= xi + xj - 1)

    # Step D: define objective
    obj_terms = []
    for i, piece in enumerate(piece_list):
        if piece['type'] == 'full_habitat':
            obj_terms.append(piece['area'])
        elif piece['type'] in ('habitat_patch', 'margin'):
            xi = piece['x_var']
            if piece['type'] == 'habitat_patch':
                obj_terms.append(piece['area'] * xi)
            else:
                obj_terms.append(margin_weight * piece['length'] * xi)

    for (i, j, dist_ij) in adjacency_list:
        w_ij = adjacency_weight[(i, j)]
        y_ij = prob.y_vars[(i, j)]
        obj_terms.append(w_ij * y_ij)

    prob.obj = pyo.Objective(expr=sum(obj_terms), sense=pyo.maximize)
    return prob, piece_list, plot_npv_expr


def solve_connectivity_ilp(plots, al_factor, neib_dist, exit_tol, params, connectivity_metric, max_loss_ratio,
                           adjacency_dist, boundary_seg_count, interior_cell_count, margin_weight):
    polygon_data = precompute_polygon_data(plots, n_points=5)
    centroids = [p["geometry"].centroid.coords[0] for p in plots]
    distances = calculate_distances(tuple(centroids))

    farm_baseline_npv, plot_baseline_npv = compute_farm_baseline_npvs(plots, polygon_data, distances, params, neib_dist)

    prob, piece_list, plot_npv_expr = build_connectivity_ilp(plots, farm_baseline_npv, params, boundary_seg_count, interior_cell_count,
                           max_loss_ratio, adjacency_dist, al_factor, neib_dist, margin_weight)
    solver = pyo.SolverFactory('ipopt')
    solver.options['acceptable_tol'] = exit_tol
    results = solver.solve(prob, tee=False)

    chosen_pieces = []
    if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:
        for pc in piece_list:
            if pc["type"] == "full_habitat":
                chosen = True
            else:
                xvar = pc["x_var"]
                if xvar is not None:
                    valx = pyo.value(xvar)
                    chosen = (valx >= 0.5)
                else:
                    chosen = False
            if chosen:
                chosen_pieces.append(pc)

        G = build_connectivity_graph_from_chosen_pieces(chosen_pieces, adjacency_dist=adjacency_dist)
        conn_val = compute_connectivity_metric(G, connectivity_metric)
        val = pyo.value(prob.obj)

        optimized_plot_npvs = {}
        for pidx, p in enumerate(plots):
            if p['plot_type'] == 'ag_plot':
                if pidx in plot_npv_expr:
                    optimized_plot_npvs[pidx] = pyo.value(plot_npv_expr[pidx])
                else:
                    optimized_plot_npvs[pidx] = 0
    else:
        optimized_plot_npvs = {}
        val, conn_val = 0, 0

    return chosen_pieces, val, conn_val, plot_baseline_npv, optimized_plot_npvs


def build_connectivity_graph_from_chosen_pieces(chosen_pieces, adjacency_dist=0.0):
    G = nx.Graph()
    for i, piece in enumerate(chosen_pieces):
        G.add_node(i,
                   geometry=piece['geom'],
                   node_type=piece['type'],
                   area=piece['area'],
                   length=piece['length'])
    for i, j in itertools.combinations(range(len(chosen_pieces)), 2):
        pi = chosen_pieces[i]
        pj = chosen_pieces[j]
        dist = pi['geom'].distance(pj['geom'])
        if dist <= adjacency_dist:
            G.add_edge(i, j)
    return G


def compute_connectivity_metric(G, connectivity_metric):
    if connectivity_metric == 'IIC':
        comps = list(nx.connected_components(G))
        total_val = 0.0
        for comp in comps:
            comp_list = list(comp)
            areas = []
            lengths = []
            for n in comp_list:
                d = G.nodes[n]
                areas.append(d['area'])
                lengths.append(d['length'])
            sum_area = 0.0
            for a_i, l_i in zip(areas, lengths):
                for a_j, l_j in zip(areas, lengths):
                    # if node is 'full_habitat' in original code, we used a factor
                    # but for brevity we'll keep the same approach:
                    # (matching the IIC adjacency weighting done earlier)
                    sum_area += (l_i * l_j) + (a_i * a_j) + (a_i * l_j) + (a_j * l_i)
            total_val += sum_area
        return total_val
    else:
        # For other metrics, or default
        return 0.0


def plot_farms(ax, plots, chosen_pieces=None, title="Farm Plots"):
    """
    Plots the farm polygons from 'plots' on the given matplotlib Axes 'ax'.
    If chosen_pieces is provided, highlights margin arcs (in red)
    and habitat polygons (in green).
    """
    patches = []
    for p in plots:
        geom = p['geometry']
        if geom.is_empty:
            continue
        if geom.geom_type == 'Polygon':
            coords = list(geom.exterior.coords)
            poly_patch = MplPolygon(coords, closed=True,
                                    facecolor='lightgray' if p['plot_type'] == 'ag_plot' else 'forestgreen',
                                    edgecolor='black', alpha=0.5)
            patches.append(poly_patch)
        elif geom.geom_type == 'MultiPolygon':
            for subg in geom.geoms:
                coords = list(subg.exterior.coords)
                poly_patch = MplPolygon(coords, closed=True,
                                        facecolor='lightgray' if p['plot_type'] == 'ag_plot' else 'forestgreen',
                                        edgecolor='black', alpha=0.5)
                patches.append(poly_patch)

    pc = PatchCollection(patches, match_original=True)
    ax.add_collection(pc)

    # highlight chosen pieces
    if chosen_pieces:
        for c in chosen_pieces:
            g = c['geom']
            if c['type'] == 'margin':
                x, y = g.xy if hasattr(g, 'xy') else ([], [])
                ax.plot(x, y, color='red', linewidth=2)
            elif c['type'] == 'habitat_patch':
                if g.geom_type == 'Polygon':
                    coords = list(g.exterior.coords)
                    poly_patch = MplPolygon(coords, closed=True,
                                            facecolor='lime', edgecolor='green', alpha=0.5)
                    ax.add_patch(poly_patch)
                elif g.geom_type == 'MultiPolygon':
                    for subp in g.geoms:
                        coords = list(subp.exterior.coords)
                        poly_patch = MplPolygon(coords, closed=True,
                                                facecolor='lime', edgecolor='green', alpha=0.5)
                        ax.add_patch(poly_patch)
            elif c['type'] == 'full_habitat':
                pass

    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.autoscale_view()

    legend_handles = []
    legend_handles.append(
            mpatches.Patch(facecolor='lightgray', edgecolor='black', alpha=0.5, label='Agricultural Plots'))
    legend_handles.append(mpatches.Patch(facecolor='forestgreen', edgecolor='black', alpha=0.5,
                                             label='Existing Habitats'))  # Adjust label if needed
    legend_handles.append(mlines.Line2D([0], [0], color='red', linewidth=2, label='Margin Interventions'))
    legend_handles.append(
            mpatches.Patch(facecolor='lime', edgecolor='green', alpha=0.5, label='Habitat Conversions'))

    # Add legend to plot if there are handles to show
    if legend_handles:
        ax.legend(handles=legend_handles, loc='best')


def save_plots_and_chosen(plots, chosen_pieces, filename):
    data = {"plots": [], "chosen_pieces": []}

    for p in plots:
        p_copy = {k: v for k, v in p.items() if k != "geometry"}
        geom_wkt = shapely_wkt.dumps(p["geometry"])
        p_copy["geometry_wkt"] = geom_wkt
        data["plots"].append(p_copy)

    for cp in chosen_pieces:
        cp_copy = {k: v for k, v in cp.items() if k not in ["geom", "var", "x_var"]}
        geom_wkt = shapely_wkt.dumps(cp["geom"])
        cp_copy["geom_wkt"] = geom_wkt
        data["chosen_pieces"].append(cp_copy)

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def optimize_landscape_connectivity(geojson_path, boundary_seg_count,
                                        interior_cell_count, adjacency_dist, connectivity_metric, al_factor, max_loss_ratio,
                                        neib_dist, exit_tol, reposition, params, conn_dir, margin_weight, mode, plot):
    plots = parse_geojson(geojson_path)

    conn_val_repos = None
    chosen_pieces_repos = None
    if reposition:
        chosen_pieces_repos, optim_val_repos, conn_val_repos, _ = solve_reposition_ilp(plots, params, adjacency_dist, boundary_seg_count,
                                                                  interior_cell_count, connectivity_metric, al_factor, neib_dist)

        print(
            f"Baseline connectivity ({connectivity_metric}) after repositioning: {conn_val_repos:.4f}, Optimization score:{optim_val_repos:.4f}")

        save_plots_and_chosen(plots, chosen_pieces_repos, os.path.join(conn_dir, f"repositioning_{mode}.json"))
        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_farms(ax, plots, chosen_pieces_repos,"")
            plt.savefig(os.path.join(conn_dir, f"repositioning_{mode}.svg"))
            plt.close()

    chosen_final, optim_val_final, conn_val_final, plot_baseline_npv, optimized_plot_npvs = solve_connectivity_ilp(plots, al_factor, neib_dist, exit_tol, params,
                                                                           connectivity_metric, max_loss_ratio, adjacency_dist,
                                                                           boundary_seg_count, interior_cell_count, margin_weight)
    print(f"New connectivity ({connectivity_metric}) after solving connectivity ilp and allowing for "
          f"max_loss_ratio of {max_loss_ratio}: {conn_val_final:.4f}, Optimization score:{optim_val_final:.4f}")

    save_plots_and_chosen(plots, chosen_final, os.path.join(conn_dir, f"connectivity_interventions_{mode}.json"))
    if plot:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        plot_farms(ax2, plots, chosen_final, "")
        plt.savefig(os.path.join(conn_dir, f"connectivity_interventions_{mode}.svg"))
        plt.close()


    return chosen_final, optim_val_final, conn_val_final, conn_val_repos, plots, plot_baseline_npv, optimized_plot_npvs, chosen_pieces_repos


if __name__ == '__main__':
    cfg = Config()
    mode = "pred"
    syn_farm_dir = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms")
    conn_dir = os.path.join(syn_farm_dir, "connectivity", "run_3")
    if not os.path.exists(conn_dir):
        os.makedirs(conn_dir)

    boundary_seg_count = 10
    interior_cell_count = 10
    adjacency_dist = 0.0
    connectivity_metric = 'IIC'
    al_factor = 1 #1e-9
    max_loss_ratio = 0.1
    params = cfg.params
    neib_dist = 1500
    exit_tol = 1e-6
    reposition = False
    margin_weight = 0.5
    plot = True

    all_plots_geojson = os.path.join(conn_dir, f"all_plots_interventions_{mode}.geojson")
    optimize_landscape_connectivity(all_plots_geojson, boundary_seg_count,
                                        interior_cell_count, adjacency_dist, connectivity_metric, al_factor, max_loss_ratio,
                                        neib_dist, exit_tol, reposition, params, conn_dir, margin_weight, mode, plot)
    print("done")
