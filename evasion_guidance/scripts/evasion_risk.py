import numpy as np
import math

def moving_average(x, N):
    '''
    https://stackoverflow.com/a/27681394
    '''
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N


def moving_geometric_mean(risk_array, horizon):
    geo_mean = []
    for i in range(0, risk_array.shape[0]-horizon+1):
        subarr = risk_array[i:i+horizon]
        geo_mean.append(subarr.prod()**(1.0/horizon))
    return np.asarray(geo_mean)


class EvasionRisk():
    def __init__(self, radar_locs, risk_interval, risk_radius, z=10.0, g=9.81):
        self.radar_locs = radar_locs
        self.z = z
        self.g = g
        self.risk_interval = risk_interval
        self.risk_radius = risk_radius
    
    def RCS(self, lamb, phi, mu, a=0.3172, b=0.1784, c=1.003):
        '''
        lamb: Aspect angle.
        phi: Elevation angle.
        mu: Bank angle.
        '''
        lamb_e = np.arccos(np.cos(phi)*np.cos(lamb))
        mu_e = mu - np.arctan2(np.tan(phi), np.sin(lamb))
        sin2_lamb_e = np.sin(lamb_e)**2
        cos2_lamb_e = np.cos(lamb_e)**2
        sin2_mu_e = np.sin(mu_e)**2
        cos2_mu_e = np.cos(mu_e)**2
        return (np.pi*(a**2)*(b**2)*(c**2)) / ((a**2)*sin2_lamb_e*cos2_mu_e + (b**2)*sin2_lamb_e*sin2_mu_e + (c**2)*cos2_lamb_e)**2

    def RCS_F117(self, alpha, beta,  a=1.01, b=0.99, c=10.0):
        '''
        lamb: Aspect angle.
        phi: Elevation angle.
        mu: Bank angle.
        '''
        sigma = a + b*np.sin(alpha) + c*abs(np.sin(alpha)*np.sin(2*beta + np.pi/2))
        return sigma
    

    def Pt(self, x, u, loc_radar, c2=(0.02)**4,  c1=1/2):
        '''
        x: state of aircraft [x, y, yaw]
        u: input
        Note: Center the aircraft with respect to radar.
        '''

        theta = np.arctan2(x[1] - loc_radar[1], x[0] - loc_radar[0])
        psi = x[2]
        lamb = theta - psi + np.pi
        phi = np.arctan2(self.z, math.hypot(x[1] - loc_radar[1], x[0] - loc_radar[0]))
        mu = np.arctan2(u, self.g)
        R = np.sqrt(math.hypot(x[1] - loc_radar[1], x[0] - loc_radar[0])**2 + self.z**2)
        sig = self.RCS(lamb, phi, mu)
        # print("State: ", x, "RCS: ", sig)
        return 1 / (1 + (c2*(R**4)/sig)**c1)



    def evalute_risk_coarse(self, trajectory, input_hist, return_history=False):
        '''
        Given a trajectory as well as the input history associated with the trajectory, evaluate the risk.
        '''
        ### If return_history is True, collect the risk at each state along the trajectory.
        risk_history = np.zeros(trajectory.shape[0])
        if return_history:
            for t in range(trajectory.shape[0]):
                for i in range(self.radar_locs.shape[0]):
                    radar_loc = self.radar_locs[i]
                    dist = math.hypot(trajectory[t, 0] - radar_loc[0], trajectory[t, 1] - radar_loc[1])
                    if dist < self.risk_radius:
                        risk_history[t] += self.Pt(trajectory[t] , input_hist[t], radar_loc)
                        
        ### Check the first and final state first. And prune far away radars.
        valid_radars = []
        for i in range(self.radar_locs.shape[0]):
            radar_loc = self.radar_locs[i]
            dist_init = math.hypot(trajectory[0, 0] - radar_loc[0], trajectory[0, 1] - radar_loc[1])
            dist_mid = math.hypot(trajectory[math.floor((0 + len(trajectory) / 2)), 0] - radar_loc[0], trajectory[math.floor((0 + len(trajectory) / 2)), 1] - radar_loc[1])
            dist_end = math.hypot(trajectory[-1, 0] - radar_loc[0], trajectory[-1, 1] - radar_loc[1])
            if dist_init <= self.risk_radius or dist_end <= self.risk_radius or dist_mid <= self.risk_radius:
                valid_radars.append(radar_loc)

        if len(valid_radars) == 0:
            return 0, risk_history
        
        risk_cost = np.zeros(len(valid_radars))
        
        for i in range(len(valid_radars)):
            risk_list = []
            for t in range(trajectory.shape[0]):
                dist = math.hypot(trajectory[t, 0] - valid_radars[i][0], trajectory[t, 1] - valid_radars[i][1])
                if dist < self.risk_radius:
                    risk_list.append(self.Pt(trajectory[t] , input_hist[t], valid_radars[i]))
                else:
                    risk_list.append(0)

            window_length = min(self.risk_interval, len(risk_list))
            # risk_cost[i] = np.max(moving_geometric_mean(np.asarray(risk_list), window_length))
            risk_cost[i] = np.max(moving_average(risk_list, window_length))

        cost = max(risk_cost)
        # print(cost, risk_history)
        # print("Cost:", cost)
        # print("Risk history: ", risk_history)
        return cost, risk_history
    

    def evalute_risk(self, state, u, return_list=False):
        num_radars = len(self.radar_locs)    
        risk_list = []
        for i in range(num_radars):
            dist = math.hypot(state[0] - self.radar_locs[i, 0], state[1] - self.radar_locs[i, 1])
            if dist < self.risk_radius:
                risk_list.append(self.Pt(state , u, self.radar_locs[i]))
            else:
                risk_list.append(0)
        risk = max(risk_list)
        if not return_list:
            return risk
        return risk, risk_list
    
    def evalute_trajectory_risk(self, trajectory, input_hist, return_history=False):
        '''
        Given a trajectory as well as the input history associated with the trajectory, evaluate the risk.
        '''
        ### If return_history is True, collect the risk at each state along the trajectory.
        risk_history = np.zeros(trajectory.shape[0])
        if return_history:
            # Hack the last time step so that we have a risk measure
            for t in range(trajectory.shape[0]):
                for i in range(self.radar_locs.shape[0]):
                    radar_loc = self.radar_locs[i]
                    dist = math.hypot(trajectory[t, 0] - radar_loc[0], trajectory[t, 1] - radar_loc[1])
                    if dist < self.risk_radius:
                        if t == trajectory.shape[0] - 1:
                            risk_history[t] += self.Pt(trajectory[t] , 0.0, radar_loc)
                        else:
                            risk_history[t] += self.Pt(trajectory[t] , input_hist[t], radar_loc)
        risk_cost = np.zeros(self.radar_locs.shape[0])
        
        for i in range(self.radar_locs.shape[0]):
            risk_list = []
            for t in range(trajectory.shape[0]-1):
                dist = math.hypot(trajectory[t, 0] - self.radar_locs[i][0], trajectory[t, 1] - self.radar_locs[i][1])
                if dist < self.risk_radius:
                    risk_list.append(self.Pt(trajectory[t] , input_hist[t], self.radar_locs[i]))
                else:
                    risk_list.append(0)

            window_length = min(self.risk_interval, len(risk_list))
            # risk_cost[i] = np.max(moving_geometric_mean(np.asarray(risk_list), window_length))
            risk_cost[i] = np.max(moving_average(risk_list, window_length))

        cost = max(risk_cost)
        # print(cost, risk_history)
        # print("Cost:", cost)
        # print("Risk history: ", risk_history)
        return cost, risk_history