import numpy as np

def sq_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert x.shape[1] == y.shape[1], "need same feature dim"
    n, d = x.shape
    m, d = y.shape
    # form distance from expansion ||x - y||^2 = x^2 - 2xy + y^2 
    dist = -2*np.einsum('mr,nr->mn', x, y) 
    dist += np.tile(np.sum(x**2, -1)[..., None], (1, n)) 
    dist += np.tile(np.sum(y**2, -1)[None, ...], (m, 1))
    return dist    


def rbf_kernel(X: np.ndarray) -> [np.ndarray, np.ndarray]:
    """RBF kernel function. 

    Args:
        X (np.ndarray): data set of n d-dimensional observations 

    Returns:
        [np.ndarray, np.ndarray]: the kernel gram matrix, the derivative wrt X
    """

    n, d = X.shape
    # get pairwise squared distances ||x_i - x_j||^2
    squared_dists = sq_dist(X, X)
    # use median to set bandwidth
    h = np.median(squared_dists) ** 2 / np.log(n)
    # compute kernel
    K = np.exp(-squared_dists / h)
    # compute dKdx
    dKdx = -K @ X + X * np.sum(K, axis=1)[:, None]
    dKdx = 2 * dKdx / h

    return K, dKdx

    

class Adagrad(object):
    
    def __init__(self, learning_rate: float, r: float = 0.0) -> None:
        """Initialize an Adagrad optimizer. 

        Args:
            learning_rate (float): the learning rate. 
            r (float, optional): the initial value of the squared gradients. Defaults to 0.0.
        """
        self.r = r
        self.t = 0 # iteration counter
        self.lr = learning_rate
        self.alpha = 0.99
        
    def update(self, g: np.ndarray) -> np.ndarray:
        """Compute the Adagrad update with momentum. 

        Args:
            g (np.ndarray): input gradient. 

        Returns:
            np.ndarray: Adagrad update. same shape as g. 
        """
        eps = 1e-6 # for numerical stability
        
        #if self.t == 0: self.r = g * g; 
        #else:
        self.r = self.alpha * self.r + (1 - self.alpha) * (g * g)
        
        delta = self.lr * g / (eps + np.sqrt(self.r))
        self.t += 1
        return delta


class Adam(object):

    def __init__(
        self,
        learning_rate: float,
        eps: float = 1e-8,
        beta1: float =0.9,
        beta2: float = 0.999) -> None:
        """Initialize an Adam optimizer. 

        Args:
            learning_rate (float): the learning rate.
            eps (float, optional): small constant for stability. Defaults to 1e-8.
            beta1 (float, optional): decay rate for first moment. Defaults to 0.9.
            beta2 (float, optional): decay rate for second moment. Defaults to 0.999.
        """

        self.t, self.m, self.v = 1, 0, 0
        self.eps, self.beta1, self.beta2 = eps, beta1, beta2
        self.lr = learning_rate

    def update(self, g: np.ndarray) -> np.ndarray:
        """Compute the Adam update. 

        Args:
            g (np.ndarray): the input gradient.

        Returns:
            np.ndarray: the update. same shape as g. 
        """
        t, m, v = self.t, self.m, self.v
        eps, beta1, beta2 = self.eps, self.beta1, self.beta2

        m = beta1 * m + (1 - beta1) * g
        mt = m / (1 - beta1 ** t + eps)
        v = beta2 * v + (1-beta2) * (g ** 2)
        vt = v / (1 - beta2 ** t + eps)
        update = self.lr * mt / (np.sqrt(vt) + eps)

        self.m, self.v = m, v
        self.t += 1
        return update


def monte_carlo_estimate(f, p, n_samples=100, **kwargs):
    # draw n_samples samples from p
    x = p.sample(size=n_samples)
    # return empirical average of f(x)
    return np.mean(f(x, **kwargs))

def mse(x, y):
    return np.mean((x - y)**2)

def h1(x): 
    return x; 

def h2(x): 
    return x**2; 

def h3(x, omega=1, b=0):
    return np.cos(omega * x + b)