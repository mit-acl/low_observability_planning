
import numpy as np
import math

class AirCraftModel():
    def __init__(self, V, delta_t, L1):
        self.V = V
        self.delta_t = delta_t
        self.L1 = L1
    
    def f(self, x, u):
        '''
        Continuous Dynamics
        x_d = Vcos(phi)
        y_d = Vsin(phi)
        phi_d = u/V
        '''
        return np.array([self.V*np.cos(x[2]), self.V*np.sin(x[2]), u/self.V])
    
    def f_discrete(self, x, u):
        '''
        Discrete Dynamics
        x_next = x + Vcos(phi)delta_t
        y_next = y + Vsin(phi)delta_t
        phi_next = phi + delta_t(u/V)
        '''
        return x + self.delta_t*self.f(x, u)
    
    def compute_input(self, x, x_reference):
        '''
        Compute lateral acceleration based on x and x_reference
        '''
        assert np.abs(np.linalg.norm(x[:2] - x_reference[:2]) - self.L1) < 1e-3
        # L1 = np.linalg.norm(x[:2] - x_reference[:2])
        # if L1 < 1e-3:
        #     return 0.0
        rel_vec = x_reference[:2] - x[:2]
        # asin returns angle in [-pi/2, pi/2], which is what we wnat
        eta = math.asin( ( (self.V*np.cos(x[2]))*rel_vec[1] - (self.V*np.sin(x[2]))*rel_vec[0] ) / (self.V*self.L1) )
        
        return 2*np.sin(eta)*(self.V**2)/self.L1

