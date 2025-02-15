'''
Collect rrt paths.
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import os
import argparse
from evasion_guidance.scripts.utils import generate_radar_config, visualiza_radar_config
from evasion_guidance.scripts.laguerre_voronoi_2d import get_power_triangulation, get_voronoi_cells
from evasion_guidance.scripts.cl_rrt import ClosedLoopRRTStar

import yaml

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
    


def main(config):
    output_path = config['data_collection']['output_path']
    os.mkdir(output_path)

    map_range = config['env']['map_range']
    radar_radius = config['env']['radar_radius']
    min_num_radar = config['env']['min_num_radar']
    max_num_radar = config['env']['max_num_radar']
    V = config['planner']['V']
    L1 = config['planner']['L1']
    planning_delta_t = config['planner']['delta_t']
    num_boundary_sample = config['planner']['serach_center_parameters']['num_boundary_sample']
    bloat_radius = config['planner']['serach_center_parameters']['bloat_radius']
    max_iter = config['planner']['max_iter']
    random_sampling = config['data_collection']['random_sample']
    search_radius= config['planner']['search_radius']
    risk_interval= config['planner']['risk_buffer_length']
    connect_circle_dist=config['planner']['connect_circle_dist']
    min_dist_to_risk_radius_ratio=config['planner']['min_dist_to_risk_radius_ratio']
    min_search_center_num=config['planner']['min_search_center_num']
    max_collection_loop_num= config['data_collection']['collection_loop_num']

    #########################################################################################################################
    ########################################            Utilities           #################################################
    #########################################################################################################################
    def get_intersection_points_dict(radar_locations, radius):
        intersection_points = []
        for i in range(radar_locations.shape[0]):
            for dtheta in range(num_boundary_sample):
                intersection_points.append([radar_locations[i][0] + (radar_radius + bloat_radius)*np.cos(dtheta*(2*np.pi)/num_boundary_sample), radar_locations[i][1] + (radar_radius + bloat_radius)*np.sin(dtheta*(2*np.pi)/num_boundary_sample)])
            
            for j in range(i+1, radar_locations.shape[0]):
                res = get_intersections(radar_locations[i], radar_locations[j], radius)
                if res is None:
                    continue
                p1x, p1y, p2x, p2y = res
                intersection_points.append([p1x, p1y])
                intersection_points.append([p2x, p2y])

                # Insert middle point
                mid_point = [(p1x + p2x) / 2.0 , (p1y + p2y) / 2.0]
                intersection_points.append(mid_point)

        return np.asarray(intersection_points)

    def generate_search_points(radar_locs):
        intersection_points = get_intersection_points_dict(radar_locs, radar_radius)
        S = radar_locs
        R = np.asarray(radar_locs.shape[0]*[radar_radius])
        tri_list, V = get_power_triangulation(S, R)

        # Compute the Voronoi cells
        voronoi_cell_map = get_voronoi_cells(S, V, tri_list)

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

        return search_points_merged, search_centers_probabilities

    def verify_goal(radar_locs, start, goal):
        if math.hypot(start[0] - goal[0], start[1] - goal[1]) < radar_radius:
            return False
        goal_pos = np.asarray([goal[0], goal[1]])
        for i in range(radar_locs.shape[0]):
            if np.linalg.norm(goal_pos - radar_locs[i]) < radar_radius / min_dist_to_risk_radius_ratio:
                return False
        return True

    #########################################################################################################################
    ########################################            Utilities           #################################################
    #########################################################################################################################

    num_data = 0
    gyaw=np.deg2rad(45.0) # doesn't matter

    seperation_radius=np.random.uniform(config['env']['seperation_radius_bounds'][0], config['env']['seperation_radius_bounds'][1])
    radar_minimal_separatin_dist=np.random.uniform(config['env']['radar_minimal_separatin_dist_bounds'][0], config['env']['radar_minimal_separatin_dist_bounds'][1])
    radar_locs, radar_orientations = generate_radar_config(min_num_radar, max_num_radar, separation_radius=seperation_radius, radar_minimal_separatin_dist=radar_minimal_separatin_dist, map_range=map_range)
    fig, ax = plt.subplots()
    visualiza_radar_config(radar_locs, radius=config['env']['radar_radius'], xlim=[0, config['env']['map_range']], ylim=[0, config['env']['map_range']])
    plt.show()
    try:
        search_points_merged, search_centers_probabilities = generate_search_points(radar_locs)
    except:
        print("Seach center ill defined.")

    num_repeat_goal = config['data_collection']['repeat_each_goal']
    while num_data < max_collection_loop_num:
        print("Number of data collected: ", num_data)
        # Set Initial parameters
        start = [map_range/2, map_range/2, np.deg2rad(0.0)]
        gx=np.random.uniform(0, map_range)
        gy=np.random.uniform(0, map_range)
        goal = [gx, gy, gyaw]
        while not verify_goal(radar_locs, start, goal):
            gx=np.random.uniform(0, map_range)
            gy=np.random.uniform(0, map_range)
            goal = [gx, gy, gyaw]

        repeat_idx = 0
        while repeat_idx < num_repeat_goal:
            ###############################################################
            ###                 Perform RRT*                            ###
            ###############################################################
            # fig, ax = plt.subplots()
            # fig.ion()
            ax = fig = None
            # print(min_search_center_num)
            closed_loop_rrt_star = ClosedLoopRRTStar(start, goal, 
                                            speed=V, L1=L1, planning_delta_t=planning_delta_t,
                                            search_centers=search_points_merged,
                                            search_centers_probabilities=search_centers_probabilities, 
                                            search_radius= search_radius,
                                            radar_locations=radar_locs,
                                            risk_radius=radar_radius,
                                            risk_interval= risk_interval,
                                            max_iter=max_iter,
                                            expand_dist=3*radar_radius,
                                            connect_circle_dist=connect_circle_dist,
                                            min_dist_to_risk_radius_ratio=min_dist_to_risk_radius_ratio,
                                            tracking_tol=planning_delta_t*V,
                                            min_search_center_num=min_search_center_num,
                                            map_bound_x_min=0.0,
                                            map_bound_y_min=0.0,
                                            map_bound_x_max=map_range,
                                            map_bound_y_max=map_range,
                                            random_sample=random_sampling)
            try:
                paths, costs, node_sequences, input_histories, risk_histories = closed_loop_rrt_star.do_planning(ax, fig, animation=False)
            except:
                continue
            
            # plt.show(block=False)
            # paths, costs, node_sequences, input_histories, risk_histories = closed_loop_rrt_star.do_planning(ax, fig, animation=False)
            # print("Path shape: ", len(paths))
            # print("Input History shape: ", input_histories[0].shape)
            # print("Risk History Shape: ", risk_histories[0].shape)
            if len(costs) > 0:
                idx=np.argmin(costs)

                node_sequence = node_sequences[idx]
                node_sequence_vis = []
                for node in node_sequence:
                    node_sequence_vis.append([node.x, node.y])
                node_sequence_vis = np.asarray(node_sequence_vis)

                episode_dict = {'radar_locations': radar_locs, 
                                'start_state': np.asarray(start),
                                'goal_location': np.array([gx, gy]),
                                'state_history': paths[idx],
                                'input_history': input_histories[idx],
                                'risk_history': risk_histories[idx],
                                'node_sequence': node_sequence_vis}
                np.save(config['data_collection']['output_path'] + f'episode_{num_data}.npy', episode_dict)
                num_data += 1
                repeat_idx += 1


        ###############################################################
        ###                 Visualize Best Trajectory               ###
        ###############################################################
        # try:
        #     idx=np.argmin(costs)

        #     node_sequence = node_sequences[idx]
        #     node_sequence_vis = []
        #     for node in node_sequence:
        #         node_sequence_vis.append([node.x, node.y])
        #     node_sequence_vis = np.asarray(node_sequence_vis)


        #     plt.close()
        #     fig, ax = plt.subplots()
        #     visualiza_radar_config(radar_locs, radius=radar_radius, xlim=[0, map_range], ylim=[0, map_range])
        #     # for path in paths:
        #     #     ax.scatter(path[:, 0], path[:, 1], s=5)
        #     ax.scatter(paths[idx][:, 0], paths[idx][:, 1], s=10, c='r')
        #     ax.scatter(node_sequence_vis[:, 0], node_sequence_vis[:, 1], s=100, c='r')
        #     for i in range(node_sequence_vis.shape[0]):
        #         ax.annotate(str(i), (node_sequence_vis[i, 0], node_sequence_vis[i, 1]))
        #     # print(risk_histories[idx].shape)
        #     for i in range(risk_histories[idx].shape[0]):
        #         if i % 10 == 0:
        #             ax.annotate(int(100*risk_histories[idx][i])/ 100.0, (paths[idx][i, 0], paths[idx][i, 1]))

        #     ax.set_xlim(0, map_range)
        #     ax.set_ylim(0.0, map_range)
        #     plt.show()
        # except:
        #     continue

    ### Save the config
    with open(config['data_collection']['output_path'] + 'config.yaml', 'w') as outfile:
        yaml.dump(config, outfile)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rrt_config', type=str, help='Path to rrt config file', required=True)
    args = parser.parse_args()
    with open(args.rrt_config,"r") as file_object:
        rrt_config = yaml.load(file_object,Loader=yaml.SafeLoader)
    main(rrt_config)
