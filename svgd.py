from tqdm import tqdm
import numpy as np
from utils import * 

#class SVGD(object):
    
#def __init__(self):
#pass
        
def run_SVGD(
    #self,
    x0,
    p,
    kernel=rbf_kernel,
    optimizer=Adagrad,
    num_iters=500,
    lr: float = 1e-3,
    verbose: bool = True):

    n, d = x0.shape
    opt = optimizer(lr)
    x = np.copy(x0)
    x_hist = [x]
    #kernel = kernel()

    iterator = tqdm(range(num_iters)) if verbose else range(num_iters)
    
    for i in iterator:

        #if i == num_iters // 2: opt.lr /= 5; 
        
        # get necessary gradients
        K, grad_K = kernel(x)
        #(K, _), grad_K = kernel(x, x), kernel.grad(x, x)
        score_p = p.score_func(x)
        # compute svgd grad
        grad = (K @ score_p + grad_K) / n
        # compute update using optimizer
        delta_x = opt.update(grad)
        x = x + delta_x
        
        x_hist.append(x)
        
    return x_hist