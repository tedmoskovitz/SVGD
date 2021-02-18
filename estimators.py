from tqdm import tqdm
import numpy as np
from utils import * 
from typing import Union, Callable
from distributions import GaussMixture1D, LatentModel

Optimizer = Union[Adagrad, Adam]
Distribution = Union[GaussMixture1D, LatentModel]


def run_SVGD(
    x0: np.ndarray,
    p: Distribution,
    kernel: Callable = rbf_kernel,
    optimizer: Optimizer = Adagrad,
    num_iters: int = 500,
    lr: float = 1e-3,
    verbose: bool = True) -> list:
    """Run Stein Variational Gradient Descent. 

    Args:
        x0 (np.ndarray): initial particle set. 
        p (Distribution): target distribution. 
        kernel (Callable, optional): kernel function. Defaults to rbf_kernel.
        optimizer (Optimizer, optional): the inner optimizer. Defaults to Adagrad.
        num_iters (int, optional): number of iterations to run Defaults to 500.
        lr (float, optional): learning rate. Defaults to 1e-3.
        verbose (bool, optional): whether to display a progres bar. Defaults to True.

    Returns:
        list: the set of particles at each time step. 
    """
    # init optimizer, etc. 
    n, d = x0.shape
    opt = optimizer(lr)
    x = np.copy(x0)
    x_hist = [x]

    iterator = tqdm(range(num_iters)) if verbose else range(num_iters)
    
    for i in iterator:
        
        # get kernel matrix, necessary grads
        K, grad_K = kernel(x)
        score_p = p.score_func(x)
        # compute svgd grad
        grad = (K @ score_p + grad_K) / n
        # compute update using optimizer
        delta_x = opt.update(grad)
        x = x + delta_x
        
        x_hist.append(x)
        
    return x_hist


def run_MLE(
    theta0: np.ndarray,
    p: Distribution,
    optimizer: Optimizer = Adagrad,
    num_iters: int = 500,
    lr: float = 1e-3,
    verbose: bool = True) -> list:
    """Perform maximum likelihood estimation. 

    Args:
        theta0 (np.ndarray): initial parameter estimate. 
        p (Distribution): the target distribution. 
        optimizer (Optimizer, optional): the inner optimizer. Defaults to Adagrad.
        num_iters (int, optional): number of iterations. Defaults to 500.
        lr (float, optional): learning rate. Defaults to 1e-3.
        verbose (bool, optional): whether to display a progress bar. Defaults to True. 

    Returns:
        list: MLE estimate at each time step. 
    """

    # initialize
    n, d = theta0.shape
    if n == 1: theta0 = theta0[0]; 
    opt = optimizer(lr)
    theta = np.copy(theta0)
    theta_hist = [theta]

    iterator = tqdm(range(num_iters)) if verbose else range(num_iters)
    
    for i in iterator:
        
        # get grad
        grad = p.grad_logli(theta)
        # compute update using optimizer
        delta_theta = opt.update(grad)
        theta = theta + delta_theta
        
        theta_hist.append(theta)
        
    return theta_hist