import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
import numpy as np
from evasion_guidance.scripts.evasion_risk import EvasionRisk
from scipy.spatial.transform import Rotation as R
from evasion_guidance.scripts.pycubicspline import *
from evasion_guidance.scripts.utils import *

class EvasionDataset(Dataset):
    def __init__(self, mcmc_data_path, rrt_data_path, params_path, num_trajectories, preload=False, device='cuda'):
        self.mcmc_data_path = mcmc_data_path
        self.rrt_data_path = rrt_data_path

        self.device = device
        self.input_limits = {"heading": 100}         # insert keys of values that should be normalized in features
        self.output_limits = {"desired_path": 50}     

        self.traj_steps = 10
        self.traj_delta_step = 1 
        self.lookahead = 15
        
        with open(os.path.join(params_path, "bc_data.yaml"),"r") as file_object:
            bc_data_config = yaml.load(file_object,Loader=yaml.SafeLoader)

        self.img_size = bc_data_config['env']['img_size']
        self.radar_radius = bc_data_config['env']['radar_radius']
        self.aircraft_detection_range = bc_data_config['env']['aircraft_detection_range']
        self.grid_size = 2 * self.aircraft_detection_range/self.img_size

        with open(os.path.join(params_path, "rrt_data_collection.yaml"),"r") as file_object:
            config = yaml.load(file_object,Loader=yaml.SafeLoader)

        self.risk_radius = config['env']['radar_radius']

        self.path_files = sorted([f for f in os.listdir(self.mcmc_data_path) if f.startswith('episode_')])
        self.num_trajectories = num_trajectories

        self.dataframe = self._create_dataframe()
        print('Finished creating dataset.')

    def __len__(self):
        return self.num_data
    
    def _create_dataframe(self):
        data = []

        for i in tqdm(range(self.num_trajectories), desc="Loading demonstrations"):

            episode_dict = np.load(os.path.join(self.rrt_data_path, f'episode_{i}.npy'),allow_pickle='TRUE').item()
            goal_location = episode_dict['goal_location']
            radar_locs = episode_dict['radar_locations']
            risk_evaluator = EvasionRisk(radar_locs, 5, self.risk_radius)

            knot_points = np.load(os.path.join(self.mcmc_data_path, f"episode_{i}.npy"))
            x_upsampled, y_upsampled, _, _, traveled_distances, _ = calc_2d_spline_interpolation(knot_points[:, 0], knot_points[:, 1], v=30.0, num=100)
            path_length = traveled_distances[-1]

            x, y, yaw, _, _, u = calc_2d_spline_interpolation(x_upsampled, y_upsampled, v=30.0, num=int(np.floor(path_length/5)))
            _, risks = risk_evaluator.evalute_trajectory_risk(np.asarray([[x[i], y[i], yaw[i]] for i in range(len(x))]), u, True)
            
            assert len(x) == len(risks), "Path risk should have the same size as the path itself."

            for j in range(len(x)):
                heat_map = get_radar_heat_map(np.asarray([x[j], y[j], yaw[j]]), radar_locs, self.img_size, self.aircraft_detection_range, self.grid_size) / 255.0
                
                ### Only load the first 100 demonstrations.
                # if demonstration > 5:
                #    continue

                data.append({
                    'demonstration_id': int(i),
                    'timestep': int(j),
                    'x': float(x[j]),
                    'y': float(y[j]),
                    'yaw': float(yaw[j]),
                    'heat_map': heat_map.astype(float),
                    'risk': float(risks[j]),
                    'goal': goal_location.astype(float),
                })
                        
        df = pd.DataFrame(data)
        
        # Sort by demonstration_id and then by timestep
        df = df.sort_values(['demonstration_id', 'timestep'], ignore_index=True)
        
        # Compute heading vectors and desired trajectories
        df[['hx', 'hy']] = 0.0
        df['desired_path'] = None
        df['desired_path_risk'] = None

        for demonstration_id in df['demonstration_id'].unique():
            demo_df = df[df['demonstration_id'] == demonstration_id]
            positions = demo_df[['x', 'y']].values
            risks = demo_df[['risk']].values

            headings = self._compute_heading_vectors(positions)
            desired_path, desired_path_risk = self._compute_desired_path_and_risk(positions, risks)

            df.loc[df['demonstration_id'] == demonstration_id, ['hx', 'hy']] = headings

            condition = df['demonstration_id'] == demonstration_id
            indices = df[condition].index

            for idx, path in zip(indices, desired_path):
                df.at[idx, 'desired_path'] = path

            for idx, path_risk in zip(indices, desired_path_risk):
                df.at[idx, 'desired_path_risk'] = path_risk

        self.num_data = len(data)
        print("Number of data: ", self.num_data)
        return df

    def _preprocess_data(self, record):
        heading = torch.tensor([record['hx'], record['hy']], dtype=torch.float32).to(self.device)
        
        features = {
            'heat_map': torch.tensor(record['heat_map'], dtype=torch.float32).to(self.device),
            'heading': heading,
            }

        desired_path = torch.tensor(record['desired_path'], dtype=torch.float32).to(self.device)
        desired_path_risk = torch.tensor(record['desired_path_risk'], dtype=torch.float32).to(self.device)

        labels = {
            'desired_path': desired_path,
            'desired_path_risk': desired_path_risk
            }

        return features, labels

    def __getitem__(self, idx):
        record = self.dataframe.iloc[idx]
        features, labels = self._preprocess_data(record)

        return self._normalize_features(features), self._normalize_labels(labels)
    
    
    def _normalize_features(self, features):
        return {
            key: (value / self.input_limits[key] if key in self.input_limits else value)
            for key, value in features.items()
        }

    def _normalize_labels(self, labels):
        return {
            key: (value / self.output_limits[key] if key in self.output_limits else value)
            for key, value in labels.items()
        }

    def _compute_heading_vectors(self, positions):
        headings = np.zeros_like(positions)
        for i in range(len(positions)):
            if i + self.lookahead < len(positions):
                diff = positions[i + self.lookahead] - positions[i]
                headings[i] = diff
            else:
                diff = positions[-1] - positions[i]
                headings[i] = diff
        return headings

    def _compute_desired_path_and_risk(self, positions, risks):
        desired_trajectories = []
        desired_trajectories_risks = []
        for i in range(len(positions)):
            # i: current position
            trajectory = np.zeros((self.traj_steps, 2))
            trajectory_risk = np.zeros(self.traj_steps)
            for j in range(self.traj_steps):
                #j : trajectory starting from i 
                target_idx = i + (j+1) * self.traj_delta_step
                
                if target_idx < len(positions):
                    trajectory[j] = positions[target_idx] - positions[i]    # Relative to current position
                    trajectory_risk[j] = risks[target_idx]
                else:
                    trajectory[j] = positions[-1] - positions[i]            # Relative to current position
                    trajectory_risk[j] = risks[-1]
            desired_trajectories.append(trajectory)
            desired_trajectories_risks.append(trajectory_risk)
        
        return desired_trajectories, desired_trajectories_risks
