import math
import time
from skspatial.objects import Circle
from skspatial.objects import Line, Point
import matplotlib.pyplot as plt
import numpy as np
from evasion_guidance.scripts.aircraft_model import AirCraftModel
from evasion_guidance.scripts.evasion_risk import EvasionRisk

def plot_arrow(x, y, yaw, ax, fig, length=1.0, width=0.5, fc="r", ec="k"):
    if isinstance(x, list):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw, ax, fig)
    else:
        ax.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw), fc=fc,
                  ec=ec, head_width=width, head_length=width)
        ax.plot(x, y)

class AirCraftPathGeneration():
    def __init__(self, delta_t, V, 
                 radar_locations, 
                 risk_interval, risk_radius, 
                 expand_dist,
                 L1,
                 tracking_err_tol=10.0):
        self.delta_t = delta_t
        self.V = V
        self.radar_locations = radar_locations
        self.tracking_controller = AirCraftModel(V, delta_t, L1)
        self.risk_interval = risk_interval
        self.risk_model = EvasionRisk(radar_locations, risk_interval, risk_radius)
        self.radar_locations = radar_locations
        self.expand_dist = expand_dist
        self.radar_detection_radius = risk_radius
        self.tracking_err_tol=tracking_err_tol
        self.L1 = L1
        self.tracking_controller = AirCraftModel(self.V, self.delta_t, self.L1)

        ### Timers
        self.time_spent_on_tracking = 0
        self.time_spent_on_evaluating_risk = 0
        self.path_generation_call_num = 0
    

    def compute_ref_point_reflection(self, x, radar_loc):
        xd = np.asarray([np.cos(x[2]), np.sin(x[2])])
        dir_rel = -(radar_loc - x[:2])

        inner_prod = np.dot(dir_rel, xd)
        if inner_prod > 0:
            return np.zeros(2)
        else:
            dir_rel_reflected = -2*inner_prod*xd + dir_rel
        dir_rel_reflected *= self.radar_detection_radius

        return dir_rel_reflected

    def get_closest_radar(self, x):
        dists = np.asarray([np.linalg.norm(x[:2] - radar_loc[:2]) for radar_loc in self.radar_locations])
        # print(dists)
        idx = np.argmin(dists)
        if (dists[idx] > self.radar_detection_radius).any():
            return -1
        return idx

    def get_reference(self, x_cur, path_init_point, path_end_point):
        circle = Circle([x_cur[0], x_cur[1]], self.L1)
        line = Line([path_init_point[0], path_init_point[1]], [path_end_point[0], path_end_point[1]])

        try:
            intersections = circle.intersect_line(line)
        except ValueError:
            point_projected = line.project_point(Point([x_cur[0], x_cur[1]]))
            dir = np.array([point_projected[0] - x_cur[0], point_projected[1] - x_cur[1], 0])
            dir_vec = dir / np.linalg.norm(dir)
            return np.array([x_cur[0], x_cur[1], 0]) + self.L1*dir_vec


        
        if type(intersections) is tuple:
            x_ref = None
            min_dist = math.inf
            for x_intersect in intersections:
                dist = math.hypot(x_intersect[0] - path_end_point[0], x_intersect[1] - path_end_point[1])
                if dist < min_dist:
                    min_dist = dist
                    x_ref = x_intersect
            # print(p_intersect, x_ref)
            return np.array([x_ref[0], x_ref[1], 0])
        
        else:
            # print("One intersection")
            # print(intersections)
            return np.array([intersections[0], intersections[1], 0])


    def generate_path(self, from_node, to_node, track=False, Timer=False):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y

        x_cur = np.array([from_node.x, from_node.y, from_node.yaw])
        path_length = 0
        
        xs = [from_node.x]
        ys = [from_node.y]
        yaws = [from_node.yaw]
        # us = [from_node.u]
        us = []

        traj = [x_cur]
        x_ref = np.array([to_node.x, to_node.y, to_node.yaw])

        # try:
        loop_idx = 0
        max_iter = min(math.ceil(math.hypot(dx, dy)/(self.delta_t*self.V)), math.ceil(self.expand_dist/(self.delta_t*self.V)))
        
        # while loop_idx <= max_iter:
        if Timer:
            start = time.time()
        
        track_failed = False
        while math.hypot(x_cur[0] - to_node.x, x_cur[1] - to_node.y) > self.tracking_err_tol:
        # for _ in range(math.ceil(math.hypot(dx, dy)/(self.delta_t*self.V))):
            if not track:
                if loop_idx > max_iter:
                    break
                
            if track and loop_idx > 1e3:
                track_failed = True
                print("Tracking takes too long.")
                break

            rel_displacement = np.array([x_ref[0] - x_cur[0], x_ref[1] - x_cur[1]])
            dir_vec = rel_displacement / np.linalg.norm(rel_displacement)
            virtual_ref = np.array([x_cur[0] + self.L1*dir_vec[0], x_cur[1] + self.L1*dir_vec[1], 0])
            u = self.tracking_controller.compute_input(x_cur, virtual_ref)
            
            ### TODO: Revisit
            # if to_node.parent is None:
            #     rel_displacement = np.array([x_ref[0] - x_cur[0], x_ref[1] - x_cur[1]])
            #     dir_vec = rel_displacement / np.linalg.norm(rel_displacement)
            #     virtual_ref = np.array([x_cur[0] + self.L1*dir_vec[0], x_cur[1] + self.L1*dir_vec[1], 0])
            #     u = self.tracking_controller.compute_input(x_cur, virtual_ref)
            # else:
            #     print("Parent not None.")
            #     virtual_reference = self.get_reference(x_cur, np.array([to_node.parent.x, to_node.parent.y, to_node.parent.yaw]), x_ref)
            #     u = self.tracking_controller.compute_input(x_cur, virtual_reference)
            ### 


            x_cur = self.tracking_controller.f_discrete(x_cur, u)

            xs.append(x_cur[0])
            ys.append(x_cur[1])
            yaws.append(x_cur[2])
            traj.append(x_cur)
            us.append(u)
            
            loop_idx += 1

        if Timer:
            end = time.time()
            self.time_spent_on_tracking += end - start

        xs = np.asarray(xs)
        ys = np.asarray(ys)
        yaws = np.asarray(yaws)
        us = np.asarray(us)
        traj = np.asarray(traj)

        if Timer:
            start = time.time()

        if traj.shape[0] > 1:
            path_total_risk, risk_history = self.risk_model.evalute_risk_coarse(traj[:-1, :], us, return_history=True)
        else:
            path_total_risk = 0
            risk_history = [0]

        if Timer:
            end = time.time()
            self.time_spent_on_evaluating_risk += end - start

        self.path_generation_call_num += 1
        path_length = loop_idx * self.V
        # if track_failed:
        #     return None, ys, yaws, us, path_length, path_total_risk, risk_history
        return xs, ys, yaws, us, path_length, path_total_risk, risk_history

    def generate_full_path(self, node_sequences):
        x_cur = np.array([node_sequences[0].x, node_sequences[0].y, node_sequences[0].yaw])
        path_length = 0
        
        xs = [node_sequences[0].x]
        ys = [node_sequences[0].y]
        yaws = [node_sequences[0].yaw]
        us = []

        traj = [x_cur]

        x_ref = np.array([node_sequences[1].x, node_sequences[1].y, node_sequences[1].yaw])

        # try:
        loop_idx = 0
        cur_node_idx = 1
        while True:
            if cur_node_idx >= len(node_sequences):
                break
            if loop_idx > 1e3:
                print("Tracking takes too long.")
                break

            if math.hypot(x_ref[0] - x_cur[0], x_ref[1] - x_cur[1]) < self.tracking_err_tol:
                cur_node_idx += 1
                if cur_node_idx >= len(node_sequences):
                    break
                x_ref = np.array([node_sequences[cur_node_idx].x, node_sequences[cur_node_idx].y, node_sequences[cur_node_idx].yaw])

            
            rel_displacement = np.array([x_ref[0] - x_cur[0], x_ref[1] - x_cur[1]])
            dir_vec = rel_displacement / np.linalg.norm(rel_displacement)
            virtual_ref = np.array([x_cur[0] + self.L1*dir_vec[0], x_cur[1] + self.L1*dir_vec[1], 0])
            u = self.tracking_controller.compute_input(x_cur, virtual_ref)

            x_cur = self.tracking_controller.f_discrete(x_cur, u)

            xs.append(x_cur[0])
            ys.append(x_cur[1])
            yaws.append(x_cur[2])
            traj.append(x_cur)
            us.append(u)
            
            loop_idx += 1

        xs = np.asarray(xs)
        ys = np.asarray(ys)
        yaws = np.asarray(yaws)
        us = np.asarray(us)
        traj = np.asarray(traj)

        if traj.shape[0] > 1:
            path_total_risk, risk_history = self.risk_model.evalute_risk_coarse(traj[:-1, :], us, return_history=True)
        else:
            path_total_risk = 0
            risk_history = [0]

        self.path_generation_call_num += 1
        path_length = loop_idx * self.V
        return xs, ys, yaws, us, path_length, path_total_risk, risk_history