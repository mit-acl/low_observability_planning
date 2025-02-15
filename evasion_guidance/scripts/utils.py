import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import qmc
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
from scipy.sparse import csgraph

from numba import jit

from scipy.spatial import ConvexHull

from matplotlib.collections import LineCollection
from matplotlib import pyplot as plot
import itertools

from evasion_guidance.scripts.laguerre_voronoi_2d import get_power_triangulation, get_voronoi_cells

def generate_radar_config(num_radar_min, num_radar_max, separation_radius=30.0, map_range=500, radar_minimal_separatin_dist=20.0):
    num_radars = np.random.randint(num_radar_min, high=num_radar_max)
    '''
    Poisson disk sampling.
    '''
    rng = np.random.default_rng()
    radius = separation_radius/map_range
    engine = qmc.PoissonDisk(d=2, radius=radius, seed=rng)
    radar_locs = map_range*engine.random(num_radars)

    radar_orientations = (2*np.pi)*(np.random.rand(num_radars))
    valid_radar_locs = []
    valid_radar_orientations = []
    for i in range(radar_locs.shape[0]):
        if np.linalg.norm(radar_locs[i, :]) > radar_minimal_separatin_dist:
            valid_radar_locs.append(radar_locs[i, :])
            valid_radar_orientations.append(radar_orientations[i])
    radar_locs = np.array(valid_radar_locs)
    radar_orientations = np.array(valid_radar_orientations)

    return radar_locs, radar_orientations


def visualiza_radar_config(radar_locs, radius=30, xlim=None, ylim=None):
    plt.scatter(radar_locs[:, 0], radar_locs[:, 1], c='b', label="Radar Locations")
    for i in range(radar_locs.shape[0]):
        # plt.scatter(radar_locs[i, 0], radar_locs[i, 1])
        plt.scatter([radar_locs[i, 0] + radius*np.cos(theta) for theta in np.linspace(0, np.pi*2)], [radar_locs[i, 1] + radius*np.sin(theta) for theta in np.linspace(0, np.pi*2)], s=0.5, c='b', alpha=0.5)
    
    if not xlim is None:
        plt.xlim(xlim)
    if not ylim is None:
        plt.ylim(ylim)

@jit(nopython=True)
def get_radar_heat_map(state, radar_locs, img_size, aircraft_detection_range, grid_size):
    '''
    state: [x, y, theta]
    '''
    radars_encoding = np.zeros((img_size, img_size))
    theta = state[2]
    loc_to_glob = np.array([[np.cos(theta), -np.sin(theta), state[0]],
                            [np.sin(theta), np.cos(theta), state[1]],
                            [0., 0., 1.]])
    
    glob_to_loc = np.linalg.inv(loc_to_glob)
    # print(glob_to_loc)
    for radar_loc in radar_locs:
        if abs(state[0] - radar_loc[0]) < aircraft_detection_range or abs(state[1] - radar_loc[1]) < aircraft_detection_range:
            glob_loc_hom = np.array([radar_loc[0], radar_loc[1], 1])
            local_loc_hom = np.dot(glob_to_loc, glob_loc_hom)
            radars_loc_coord = local_loc_hom[:2]

            y_grid = np.rint((radars_loc_coord[1]) / grid_size) 
            x_grid = np.rint((radars_loc_coord[0]) / grid_size) 

            for i in range(-int(img_size/2), int(img_size/2)):
                for j in range(-int(img_size/2), int(img_size/2)):
                    radars_encoding[int(i + img_size/2), int(j + img_size/2)] += np.exp(( -(x_grid - i)**2 / (10.0)**2 - (y_grid - j)**2 / (10.0)**2 ))*1e3

    ### Transpose so that x <---> row, y <---> column
    radars_encoding = radars_encoding.T

    ### Make the magnitude correct.
    if np.max(radars_encoding) > 0:
        formatted = (radars_encoding * 255.0 / np.max(radars_encoding)).astype('float32')
    else:
        formatted = radars_encoding.astype('float32')
    
    ### Add one more dimension (batch)
    formatted = formatted[np.newaxis, :, :]

    return formatted

def center_state(state_inert, loc):
    '''
    Canter loc with respect to state_inert.
    '''
    mat = np.array([
            [np.cos(state_inert[2]), -np.sin(state_inert[2]), state_inert[0]],
            [np.sin(state_inert[2]), np.cos(state_inert[2]), state_inert[1]],
            [0, 0, 1]
        ])
    mat_inv = np.linalg.inv(mat)
    loc_hom = np.array([loc[0], loc[1], 1])
    return np.dot(mat_inv, loc_hom)[:2]

'''
From: https://stackoverflow.com/questions/55816902/finding-the-intersection-of-two-circles
'''
def get_intersections(radar_loc_1, radar_loc_2, radius):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1
    x0 = radar_loc_1[0]
    y0 = radar_loc_1[1]

    x1 = radar_loc_2[0]
    y1 = radar_loc_2[1]

    r0 = r1 = radius

    d = math.sqrt((x1-x0)**2 + (y1-y0)**2)
    
    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        
        return (x3, y3, x4, y4)
    
def get_intersection_points_dict(radar_locations, radar_radius, num_boundary_sample, bloat_radius):
    intersection_points = []
    for i in range(radar_locations.shape[0]):
        for dtheta in range(num_boundary_sample):
            intersection_points.append([radar_locations[i][0] + (radar_radius + bloat_radius)*np.cos(dtheta*(2*np.pi)/num_boundary_sample), radar_locations[i][1] + (radar_radius + bloat_radius)*np.sin(dtheta*(2*np.pi)/num_boundary_sample)])
        
        for j in range(i+1, radar_locations.shape[0]):
            res = get_intersections(radar_locations[i], radar_locations[j], radar_radius)
            if res is None:
                continue
            p1x, p1y, p2x, p2y = res
            intersection_points.append([p1x, p1y])
            intersection_points.append([p2x, p2y])

            # Insert middle point
            mid_point = [(p1x + p2x) / 2.0 , (p1y + p2y) / 2.0]
            intersection_points.append(mid_point)

    return np.asarray(intersection_points)



def generate_search_points(radar_locs, radar_radius, map_range, num_boundary_sample, bloat_radius):
    '''
    Modified based on
    https://gist.github.com/marmakoide/45d5389252683ae09c2df49d0548a627#file-laguerre-voronoi-2d-py
    '''
    intersection_points = get_intersection_points_dict(radar_locs, radar_radius, num_boundary_sample, bloat_radius)
    S = radar_locs
    R = np.asarray(radar_locs.shape[0]*[radar_radius])
    tri_list, V = get_power_triangulation(S, R)

    # Compute the Voronoi cells
    voronoi_cell_map = get_voronoi_cells(S, V, tri_list)
    # Setup
    fig, ax = plot.subplots()
    
    plot.axis('equal')
    plot.axis('off')	

	# Set min/max display size, as Matplotlib does it wrong
    min_corner = np.amin(S, axis = 0) - np.max(R)
    max_corner = np.amax(S, axis = 0) + np.max(R)
    plot.xlim((min_corner[0], max_corner[0]))
    plot.ylim((min_corner[1], max_corner[1]))

	# Plot the samples
    for Si, Ri in zip(S, R):
        ax.add_artist(plot.Circle(Si, Ri, fill = True, alpha = .4, lw = 0., color = '#8080f0', zorder = 1))

	# Plot the power triangulation
    edge_set = frozenset(tuple(sorted(edge)) for tri in tri_list for edge in itertools.combinations(tri, 2))
    line_list = LineCollection([(S[i], S[j]) for i, j in edge_set], lw = 1., colors = '.9')
    line_list.set_zorder(0)
    ax.add_collection(line_list)


    # Plot the Voronoi cells
    edge_map = { }
    for segment_list in voronoi_cell_map.values():
        for edge, (A, U, tmin, tmax) in segment_list:
            edge = tuple(sorted(edge))
            if edge not in edge_map:
                if tmax is None:
                    tmax = 10
                if tmin is None:
                    tmin = -10

                edge_map[edge] = (A + tmin * U, A + tmax * U)


    line_list = LineCollection(edge_map.values(), lw = 1., colors = 'k')
    line_list.set_zorder(0)
    ax.add_collection(line_list)
    ax.scatter(radar_locs[:, 0], radar_locs[:, 1], c='r', label='Radar Locations')

        

    point_count = 0
    search_centers = []
    for p1, p2 in edge_map.values():
        mid_point = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        if p1[0] < map_range and p1[0] > -map_range/50 and p1[1] < map_range and p1[1] > -map_range/50:
            point_count += 1
            search_centers.append(p1)
        if p2[0] < map_range and p2[0] > -map_range/50 and p2[1] < map_range and p2[1] > -map_range/50:
            point_count += 1 
            search_centers.append(p2)
        if mid_point[0] < map_range and mid_point[0] > -map_range/50 and mid_point[1] < map_range and mid_point[1] > -map_range/50:
            point_count += 1 
            search_centers.append(mid_point)

    search_centers = np.asarray(search_centers)

    if search_centers.shape[0] != 0:
        all_search_points = np.concatenate((intersection_points, search_centers))
    else:
        all_search_points = intersection_points

    search_points_merged = []

    for point in all_search_points:
        valid_flag = True
        for point2 in search_points_merged:
            if math.hypot(point[0]-point2[0], point[1]-point2[1]) < 5.0:
                valid_flag = False
        if valid_flag:
            search_points_merged.append(point)

    search_points_merged = np.asarray(search_points_merged)

    search_centers_probabilities = np.ones(search_points_merged.shape[0])
    for i, [px, py] in enumerate(search_points_merged):
        for rx, ry in radar_locs:
            d = math.hypot(px-rx, py-ry)
            if d < radar_radius+1e-3:
                search_centers_probabilities[i] *= np.exp(-0.5*(radar_radius**2/d**2))
    search_centers_probabilities /= np.sum(search_centers_probabilities)


	# Job done
    ax.scatter(search_points_merged[:, 0], search_points_merged[:, 1], c='g', alpha=0.8, label="Sampling Distribution")
    ax.legend()
    plot.show()
    return search_points_merged, search_centers_probabilities