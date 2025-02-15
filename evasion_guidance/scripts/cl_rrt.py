import matplotlib.pyplot as plt
from IPython import display

import numpy as np
import random
import time
import math
import copy

from evasion_guidance.scripts.aircraft_tracking import AirCraftPathGeneration, plot_arrow

'''
Template based on:
https://github.com/AtsushiSakai/PythonRobotics
'''

class ClosedLoopRRTStar():
    """
    Class for Closed loop RRT star planning
    """
    class Node:
        """
        RRT Node
        """
        def __init__(self, x, y, yaw, u=0):
            self.x = x
            self.y = y
            self.yaw = yaw

            self.u = u

            ### Used for collision checking
            self.path_x = []
            self.path_y = []

            self.parent = None

            self.path_length_from_parent = math.inf
            self.total_cost = 0.0
            self.cost_from_parent = 0.0


    def __init__(self, start, goal, 
                 speed,
                 L1,
                 search_centers,
                 search_centers_probabilities,
                 search_radius,
                 radar_locations,
                 risk_radius,
                 risk_interval=20,
                 max_iter=200,
                 connect_circle_dist=500.0,
                 planning_delta_t =0.5,
                 goal_threshold=25.0,
                 start_threshold=10.0,
                 expand_dist=150,
                 min_dist_to_risk_radius_ratio=5.0,
                 map_bound_x_min=-50,
                 map_bound_x_max=550,
                 map_bound_y_min=-50,
                 map_bound_y_max=550,
                 tracking_tol=10.0,
                 min_search_center_num=10,
                 random_sample=False
                 ):
        '''
        search_centers (Nx2 np.array): random nodes are sampled around search_centers 
        '''
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2]) # End yaw doesn't matter for now.

        self.max_iter = max_iter
        self.planning_delta_t = planning_delta_t

        ### Used for computing the radius to search for nearby nodes.
        self.connect_circle_dist = connect_circle_dist
        self.expand_dist = expand_dist

        self.goal_threshold = goal_threshold
        self.start_threshold = start_threshold

        self.tracking_tol = tracking_tol

        ### Aircraft speed
        self.speed = speed

        # Samplings
        self.random_sample = random_sample
        self.search_centers = search_centers
        self.search_radius = search_radius # std 
        self.search_centers_probabilities = search_centers_probabilities

        self.radar_locations = radar_locations
        self.radar_locations_search_probability = (1/radar_locations.shape[0])*np.ones(radar_locations.shape[0])
        # self.radar_orientations = radar_orientations

        self.path_generator = AirCraftPathGeneration(planning_delta_t, speed, radar_locations, 
                                                     risk_interval, risk_radius, expand_dist, L1, tracking_err_tol=tracking_tol)
        self.risk_radius = risk_radius
        self.min_search_center_num = min_search_center_num

        self.has_valid_path = False
        ### This is the minimum distance that the aircraft should be away from any radar.
        self.node_min_dist_threshold = risk_radius / min_dist_to_risk_radius_ratio

        ### Map bounds.
        self.map_bound_x_min=map_bound_x_min
        self.map_bound_x_max=map_bound_x_max
        self.map_bound_y_min=map_bound_y_min
        self.map_bound_y_max=map_bound_y_max

        ### Timer 
        self.time_spent_on_sampling = 0
        self.time_spent_on_steering = 0
        self.steering_count = 0
        self.time_spent_on_validation = 0
        self.validation_count = 0
        self.time_spent_on_rewiring = 0
        self.rewire_count = 0
        self.time_spent_on_computing_cost = 0
        self.computing_cost_count = 0
        self.steer_for_parent_count = 0
        self.steer_for_new_node_count = 0
        self.steer_for_rewire_count = 0
        self.steer_for_goal_count = 0
        

    def get_random_node(self):
        start = time.time()

        if self.random_sample or np.random.binomial(1, 0.5, 1)[0]:
            # print("Uniform Sampling...")
            rnd = self.Node(
                np.random.uniform(self.map_bound_x_min, self.map_bound_x_max),
                np.random.uniform(self.map_bound_y_min, self.map_bound_y_max),
                0.0) # Note yaw doesn't matter
            idx = -1

            return rnd, idx
        
        idx_center = np.random.choice(self.search_centers.shape[0], p=self.search_centers_probabilities)
        rnd = self.Node(
            np.random.normal(self.search_centers[idx_center][0], self.search_radius),
            np.random.normal(self.search_centers[idx_center][1], self.search_radius),
            0.0) # Note yaw doesn't matter
        idx = idx_center


        end = time.time()
        self.time_spent_on_sampling += end - start

        return rnd, idx
    
    def steer(self, from_node: Node, to_node: Node, tracking=False):
        start = time.time()
        ### Remark: steering distance should be > risk checking horizon
        ### Disable tracking for now.
        xs, ys, yaws, us, path_length, risk, _ =  self.path_generator.generate_path(from_node, to_node, False)

        if xs is None or xs.shape[0] == 1:
            return None
        
        new_node = self.Node(xs[-1], ys[-1], yaws[-1], us[-1])

        new_node.path_x = xs
        new_node.path_y = ys

        new_node.total_cost = from_node.total_cost + risk
        new_node.cost_from_parent = risk
        new_node.parent = from_node

        new_node.path_length_from_parent = path_length

        end = time.time()
        self.steering_count += 1
        self.time_spent_on_steering += end - start

        return new_node
    
    def validate_node_path(self, node):
        '''
        Validate the new node by checking if any waypoint it passes is too close to any radar.
        '''
        self.validation_count += 1
        start = time.time()
        for x, y in zip(node.path_x, node.path_y):
            if x < self.map_bound_x_min or x > self.map_bound_x_max or y < self.map_bound_y_min or y > self.map_bound_y_max:
                end = time.time()
                self.time_spent_on_validation += end - start
                return False
            dist_list = [math.hypot(radar_loc[0] - x, radar_loc[1] - y) for radar_loc in self.radar_locations]
            if any(d < self.node_min_dist_threshold for d in dist_list):
                end = time.time()
                self.time_spent_on_validation += end - start
                return False
        end = time.time()
        self.time_spent_on_validation += end - start
        return True
    
    def validate_new_node(self, new_node):
        '''
        Validate the new node by checking if any waypoint it passes is too close to any radar.
        '''
        self.validation_count += 1
        start = time.time()
        dist_list = [math.hypot(radar_loc[0] - new_node.x, radar_loc[1] - new_node.y) for radar_loc in self.radar_locations]
        if any(d < self.node_min_dist_threshold for d in dist_list):
            end = time.time()
            self.time_spent_on_validation += end - start
            return False
        end = time.time()
        self.time_spent_on_validation += end - start
        return True
    

    def do_planning(self, ax, fig, animation=True):

        self.rrt_star(ax, fig, animation=animation)

        path_indexs = self.get_goal_indices()
        paths, costs, node_sequences, input_histories, risk_histories = self.generate_paths_from_indices(path_indexs)

        ###############################################################
        ###                 Print statistics                        ###
        ###############################################################

        # print("Time spent on sampling: ", self.time_spent_on_sampling)

        # print("Time spent on validation: ", self.time_spent_on_validation, "Number count: ", self.validation_count)
        # if self.validation_count > 0:
        #     print("Average run time: ", self.time_spent_on_validation / self.validation_count)

        # print("Time spent on computing cost: ", self.time_spent_on_computing_cost, "Number count: ", self.computing_cost_count)
        # if self.computing_cost_count > 0:
        #     print("Average run time: ", self.time_spent_on_computing_cost / self.computing_cost_count)


        # print("Time spent on rewiring: ", self.time_spent_on_rewiring, "Number count: ", self.rewire_count)
        # if self.rewire_count > 0:
        #     print("Average run time: ", self.time_spent_on_rewiring / self.rewire_count)

        # print("Time spent on steering: ", self.time_spent_on_steering, "Number count: ", self.steering_count)
        # if self.steering_count > 0:
        #     print("Average run time: ", self.time_spent_on_steering / self.steering_count)

        # print("Steer count for new node: ", self.steer_for_new_node_count)
        # print("Steer count for parent: ", self.steer_for_parent_count)
        # print("Steer count for rewire: ", self.steer_for_rewire_count)
        # print("Steer count for goal: ", self.steer_for_goal_count)

        # print("Time spent on generating control inputs: ", self.path_generator.time_spent_on_tracking)
        # print("Time spent on evaluating risk: ", self.path_generator.time_spent_on_evaluating_risk)
        # print("Number of calls of path generation: ", self.path_generator.path_generation_call_num)

        return paths, costs, node_sequences, input_histories, risk_histories
    
    def chooose_best_path(self, paths):
        '''
        Choose the best path from paths
        ''' 
        return 

    def rrt_star(self, ax, fig, animation=True, search_until_max_iter=True):
        self.node_list = [self.start]

        for i in range(self.max_iter):
            # print("Iter:", i, ", number of nodes:", len(self.node_list))

            if animation:
                if i == 0:
                    self.plot_start_goal_arrow(ax, fig)
                    self.draw_graph(ax, fig)
                    time.sleep(0.1)

            ### Line 3 - 5
            rnd, _ = self.get_random_node()
            nearest_idx = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_idx], rnd)
            self.steer_for_new_node_count += 1
            ###

            ### Line 6
            if not new_node or not self.validate_node_path(new_node):
                continue
            ###
            

            ### Line 7 - 13
            near_indices = self.find_near_nodes(new_node)
            new_node = self.choose_parent(new_node, near_indices)
            ###

            ### Line 14 - 16
            if not new_node or not self.validate_node_path(new_node):
                continue

            self.node_list.append(new_node)
            self.rewire(new_node, near_indices)
            self.try_goal_path(new_node)

            ### TODO: Lower the probability of selected search center?
            # if idx >= 0:
            #     self.search_centers_probabilities[idx] *= 1e-6
            #     self.search_centers_probabilities /= np.sum(self.search_centers_probabilities)
            ###

            if animation:
                self.plot_start_goal_arrow(ax, fig)
                self.draw_graph(ax, fig, rnd, new_node)

        # print("Reached max iteration")
        return
    
    
    def generate_paths_from_indices(self, path_indices):
        '''
        Generate paths that lead to nodes indexed by path_indices.
        '''
        # print("Start search feasible path")
        paths = []
        input_histories = []
        risk_histories = []
        costs = []
        node_sequences = []
        for idx in path_indices:
            path, inputs, risks, cost, node_sequence = self.generate_final_course(idx)
            if not path:
                continue
            paths.append(np.asarray(path))
            input_histories.append(np.asarray(inputs))
            risk_histories.append(np.asarray(risks))
            ### 'cost' is the cost of the whole path, while 'risks' collect risk at each state.
            costs.append(cost)
            node_sequences.append(node_sequence)

        return paths, costs, node_sequences, input_histories, risk_histories

    def generate_final_course(self, goal_index):
        # print("Generating Final Course...")
        path = []
        cost = 0

        node = self.node_list[goal_index]
        node_path = []

        node_loop_idx = 0
        while node: 
            node_path.append(node)
            node = node.parent
            node_loop_idx += 1
            if node_loop_idx > 100:
                # TODO: Need to fix this.
                print('Recursive Node Path.....')
                return None, None, None, None, None
        node_path = list(reversed(node_path))

        if math.hypot(self.start.x - node_path[0].x, self.start.y - node_path[0].y) > self.start_threshold:
            return None, None, None, None, None

        xs, ys, yaws, us, path_length, risk, risk_history =  self.path_generator.generate_full_path(node_path)
        cost = risk + 0.05*path_length
        for (ix, iy, iyaw) in zip(xs, ys, yaws):
            path.append([ix, iy, iyaw])

        return path, us, risk_history, cost, node_path

    def get_goal_indices(self):
        goal_indices = []
        for (i, node) in enumerate(self.node_list):
            if self.calc_dist_to_goal(node.x, node.y) <= self.goal_threshold:
                goal_indices.append(i)
        # print("Goal indices number: ")
        # print(len(goal_indices))

        return goal_indices
    
    def try_goal_path(self, node):
        steer_dir_x = self.end.x - node.x
        steer_dir_y = self.end.y - node.y
        steer_norm = math.hypot(steer_dir_x, steer_dir_y)

        ### TODO: think more about this
        drive_dist = min(self.expand_dist, steer_norm)
        goal = self.Node(node.x + drive_dist*steer_dir_x/steer_norm, node.y + drive_dist*steer_dir_y/steer_norm, self.end.yaw)

        new_node = self.steer(node, goal)
        self.steer_for_goal_count += 1

        if new_node is None or not self.validate_node_path(new_node):
            return

        if self.check_reached(new_node, self.end):
            self.has_valid_path = True
        self.node_list.append(new_node)


    def compute_cost(self, from_node, to_node):
        self.computing_cost_count += 1
        start = time.time()
        xs, ys, yaws, us, path_length, risk, _ = self.path_generator.generate_path(from_node, to_node)
        end = time.time()
        self.time_spent_on_computing_cost += end - start
        return risk, xs, ys, yaws, us, path_length
    
    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
        # if expand_dist exists, search vertices in a range no more than expand_dist
        if hasattr(self, 'expand_dist'):
            r = min(r, self.expand_dist)
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def choose_parent(self, new_node: Node, near_inds):
        if not near_inds:
            return None

        # search nearest cost in near_inds
        best_new_node = new_node

        for i in near_inds:
            near_node = self.node_list[i]
            temporary_node = self.steer(near_node, new_node)

            self.steer_for_parent_count += 1

            if temporary_node and self.validate_node_path(temporary_node) and self.check_reached(temporary_node, new_node):
                if temporary_node.total_cost < best_new_node.total_cost:
                    best_new_node = temporary_node
                elif abs(temporary_node.cost_from_parent - best_new_node.cost_from_parent) < 0.001 and (temporary_node.path_length_from_parent < best_new_node.path_length_from_parent):
                    best_new_node = temporary_node
        return best_new_node
    
    def check_reached(self, from_node: Node, to_node: Node):
        if math.hypot(from_node.x - to_node.x, from_node.y - to_node.y) <= self.tracking_tol:
            return True
        return False

    def rewire(self, new_node: Node, near_inds):
        start = time.time()
        for i in near_inds:
            near_node = self.node_list[i]

            if near_node == new_node.parent:
                continue

            edge_node = self.steer(new_node, near_node, True)
            self.steer_for_rewire_count += 1

            if not edge_node or not self.validate_node_path(edge_node) or not self.check_reached(edge_node, near_node):
                continue
            
            try:
                improved_cost = (near_node.total_cost > edge_node.total_cost) or (abs(near_node.cost_from_parent - edge_node.cost_from_parent) < 0.001 and (1.1*edge_node.path_length_from_parent < near_node.path_length_from_parent))
                # improved_cost = near_node.total_cost > edge_node.total_cost
                if improved_cost:
                    for node in self.node_list:
                        if node.parent == near_node:
                            node.parent = edge_node

                            cost, xs, ys, _, _, path_length = self.compute_cost(edge_node, node)
                            
                            node.path_x = xs
                            node.path_y = ys

                            node.cost_from_parent = cost
                            node.total_cost = edge_node.total_cost + cost

                            node.path_length_from_parent = path_length
                    
                    self.node_list[i] = edge_node
                    self.propagate_cost_to_leaves(self.node_list[i], top_level=True)
            except Exception:
                improved_cost = near_node.total_cost > edge_node.total_cost
                if improved_cost:
                    for node in self.node_list:
                        if node.parent == near_node:
                            node.parent = edge_node

                            cost, xs, ys, _, _, path_length = self.compute_cost(edge_node, node)
                            
                            node.path_x = xs
                            node.path_y = ys

                            node.cost_from_parent = cost
                            node.total_cost = edge_node.total_cost + cost

                            node.path_length_from_parent = path_length
                    
                    self.node_list[i] = edge_node
                    self.propagate_cost_to_leaves(self.node_list[i], top_level=True)

        end = time.time()
        self.time_spent_on_rewiring += end - start
        self.rewire_count += 1

    def propagate_cost_to_leaves(self, parent_node : Node , top_level=False):
        # for node in self.node_list:
        for node in self.node_list:
            if node.parent == parent_node:
                if top_level:
                    self.propagate_cost_to_leaves(node)
                else:
                    node.total_cost = parent_node.total_cost + node.cost_from_parent
                    self.propagate_cost_to_leaves(node)


    def plot_start_goal_arrow(self, ax, fig):
        plot_arrow(self.start.x, self.start.y, self.start.yaw, ax, fig)
        plot_arrow(self.end.x, self.end.y, self.end.yaw, ax, fig)
        

    def draw_graph(self, ax, fig, rnd=None, new_node=None):
        ax.clear()
        # for stopping simulation with the esc key.
        fig.canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            ax.plot(rnd.x, rnd.y, "^b", markersize=20, label='Sampled Node')
        if new_node is not None:
            ax.plot(new_node.x, new_node.y, "^g", markersize=20, label='New Node')
        for node in self.node_list:
            ax.plot(node.x, node.y, "*g", markersize=10)
            if node.parent:
                ax.plot(node.path_x, node.path_y, "--g", alpha=0.7)

        ax.plot(self.search_centers[:, 0], self.search_centers[:, 1], '.b', markersize=5, label='Search Center', alpha=0.3)
            
        ax.plot(self.radar_locations[:, 0], self.radar_locations[:, 1], 'Xr')
        for i in range(self.radar_locations.shape[0]):
            # plt.arrow(self.radar_locations[i, 0], self.radar_locations[i, 1], 10*np.cos(self.radar_orientations[i]), 10*np.sin(self.radar_orientations[i]))
            ax.plot([self.radar_locations[i, 0] + self.risk_radius*np.cos(theta) for theta in np.linspace(0, np.pi*2)], 
                    [self.radar_locations[i, 1] + self.risk_radius*np.sin(theta) for theta in np.linspace(0, np.pi*2)], 
                    'Xr', alpha=0.3, markersize=0.3)
    
        ax.plot(self.start.x, self.start.y, "xr")
        ax.plot(self.end.x, self.end.y, "xr", markersize=10.0)
        ax.axis([self.map_bound_x_min, self.map_bound_x_max, self.map_bound_y_min, self.map_bound_y_max])
        ax.grid(True)
        ax.legend()
        self.plot_start_goal_arrow(ax, fig)

        display.display(plt.gcf())
        # time.sleep(0.2)
        plt.cla()
        display.clear_output(wait =True)

    def set_random_seed(self, seed):
        random.seed(seed)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        
        minind = dlist.index(min(dlist))

        return minind 

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)
    