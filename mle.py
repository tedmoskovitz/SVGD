import numpy as np
from utils import *
from tqdm import tqdm

class MLE(object):
    
    def __init__(self):
        pass
        
    def run(self, theta0, p, optimizer=Adagrad, num_iters=500, lr: float = 1e-3):
        n, d = theta0.shape
        if n == 1: theta0 = theta0[0]; 
        opt = optimizer(lr)
        theta = np.copy(theta0)
        theta_hist = [theta]
        
        for i in tqdm(range(num_iters)):
            
            # get grad
            grad = p.grad_logli(theta)
            # compute update using optimizer
            delta_theta = opt.update(grad)
            theta = theta + delta_theta
            
            theta_hist.append(theta)
            
        return theta_hist