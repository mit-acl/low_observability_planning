import numpy as np
from scipy.stats import multivariate_normal
import os
from multiprocessing import Process, Queue

import time
import argparse
from evasion_guidance.scripts.pycubicspline import *
from evasion_guidance.scripts.evasion_risk import EvasionRisk

import yaml


def main(rrt_config, mcmc_config):
    rrt_data_path = rrt_config['data_collection']['output_path']

    data = np.load(os.path.join(rrt_data_path, 'episode_0.npy'),allow_pickle='TRUE').item()
    radar_locs = data['radar_locations']
    risk_radius = rrt_config['env']['radar_radius']
    risk_buffer_length = rrt_config['planner']['risk_buffer_length']
    risk_evaluator = EvasionRisk(radar_locs, risk_buffer_length, risk_radius)
    V = rrt_config['planner']['V']

    def P(risk, path_length, curvature):
        if np.any(abs(np.asarray(curvature)) > 0.5):
            return 0.0

        return np.exp(-100*risk)*np.exp(-path_length/50.0)


    num_data = mcmc_config['num_data']
    mcmc_data_path = mcmc_config['output_path']
    os.mkdir(mcmc_data_path)

    for data_idx in range(num_data):
        # print("Processing data: ", data_idx)
        data = np.load(os.path.join(rrt_data_path, f'episode_{data_idx}.npy'),allow_pickle='TRUE').item()
        path = data['state_history']
        print("Path length: ", len(path))
        node_sequence = data['node_sequence']
        x, y, yaw, k, travel, u = calc_2d_spline_interpolation(path[:, 0], path[:, 1], v=V, num=2*len(path))

        # APPEND TWICE FOR ZERO VELOCITY
        x0 = []
        x0.append(np.copy(path[0, :-1]))
        x0[0][0] -= 1e-6 
        x0.append(np.copy(path[0, :-1]))
        x0.extend([[p[0], p[1]] for p in zip(x[1::10], y[1::10])])
        x0.append(path[-1, :-1])
        xt = np.asarray(x0).flatten()
        print("Xt shape: ", xt.shape)

        cov = 20*np.eye(2*len(x0))
        cov[:2, :2] = np.zeros((2, 2))
        cov[2:4, 2:4] = np.zeros((2, 2))
        cov[-2:, -2:] = np.zeros((2, 2))

        samples = []
        risks = []
        acceptance_counts = 0
        for i in range(30_000):            
            cov_i = np.exp(-i/3000)*cov
            xt_candidate = np.random.multivariate_normal(xt, cov_i)

            x, y, yaw, k_candidate, travel_candidate, u_candidate = calc_2d_spline_interpolation(xt_candidate.reshape(-1, 2)[:, 0], xt_candidate.reshape(-1, 2)[:, 1], v=V, num=100)
            knots_candidate = np.asarray([np.asarray([x[i], y[i], yaw[i]]) for i in range(len(x))])
            risk_candidate, _ = risk_evaluator.evalute_trajectory_risk(knots_candidate, u_candidate) 

            x, y, yaw, k_xt, travel_xt, u_xt = calc_2d_spline_interpolation(xt.reshape(-1, 2)[:, 0], xt.reshape(-1, 2)[:, 1], v=V, num=100)
            knots_t = np.asarray([[x[i], y[i], yaw[i]] for i in range(len(x))])
            risk_xt, _ = risk_evaluator.evalute_trajectory_risk(knots_t, u_xt) 

            rv = multivariate_normal(mean=xt_candidate, cov=cov_i, allow_singular=True)
            Q1 = rv.pdf(xt)

            rv = multivariate_normal(mean=xt, cov=cov_i, allow_singular=True)
            Q2 = rv.pdf(xt_candidate)

            accept_prob = np.divide(P(risk_candidate, travel_candidate[-1], k_candidate) * Q1, P(risk_xt, travel_xt[-1], k_xt) * Q2)

            if np.random.uniform(0, 1) < accept_prob:
                acceptance_counts += 1
                xt = xt_candidate
                risk_xt = risk_candidate

            samples.append(xt)
            risks.append(risk_xt)
            # print()
        print("Acceptance rate: ", acceptance_counts / 30_000)
        best_idx = np.argmin(risks[10000:]) + 10000
        print("Best index: ", best_idx)
        np.save(os.path.join(mcmc_data_path, f'episode_{data_idx}.npy'), samples[best_idx].reshape(-1, 2)) 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rrt_config', type=str, help='Path to rrt config file', required=True)
    parser.add_argument('--mcmc_config', type=str, help='Path to mcmc config file', required=True)
    args = parser.parse_args()
    with open(args.rrt_config,"r") as file_object:
        rrt_config = yaml.load(file_object,Loader=yaml.SafeLoader)
    with open(args.mcmc_config,"r") as file_object:
        mcmc_config = yaml.load(file_object,Loader=yaml.SafeLoader)
    main(rrt_config, mcmc_config)
