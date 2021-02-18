import numpy as np
import matplotlib.pyplot as plt
import matplotlib

AXIS = type(plt.subplots()[1])
plt.clf()

class GaussMixture1D(object):
    
    def __init__(
        self,
        params: list,
        mixture_wts: list
        ) -> None:
        """A Gaussian mixture.

        Args:
            params (list): list of (mu, cov) for each component of the mixture. 
            mixture_wts (list): list of weights w (prior probs) assigned to each mixture. 
        """
        assert len(params) == len(mixture_wts), "need a weight for each Gaussian"
        assert np.sum(mixture_wts) == 1, "weights must sum to one"
        assert np.array(mixture_wts).all() >= 0, "weights must be non-negative"
        self.N = len(params)
        self.params = params
        self.mixture_wts = mixture_wts

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """A simple (naive) sampling scheme for the mixture. 

        Args:
            n_samples (int, optional): number of samples to draw. Defaults to 1.

        Returns:
            np.ndarray: the samples
        """
        
        samples = []
        for _ in range(n_samples):
            # draw a cluster
            cluster = np.random.choice(self.N, p=self.mixture_wts)
            # draw from the corresponding Gaussian
            mu, var = self.params[cluster]
            x = np.random.normal(loc=mu, scale=np.sqrt(var))
            samples.append(x)
        
        return np.array(samples)
        
    def _gauss_pdf(self, x: np.ndarray, mu: float, var: float) -> np.ndarray:
        """Compute 1D Gaussian pdf 

        Args:
            x (np.ndarray): the points to evaluate
            mu (float): mean
            var (float): variance

        Returns:
            np.ndarray: probability of each point under the distribution
        """
        
        pdf = np.exp(-(x - mu)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)
        return pdf
    
    def evaluate_density(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the probability of a set of points under the mixture distribution.

        Args:
            x (np.ndarray): the points to evaluate

        Returns:
            np.ndarray: the density for each point
        """
        density = 0
        for w, (mu, var) in zip(self.mixture_wts, self.params): 
            density += w * self._gauss_pdf(x, mu, var)
        return density
    
    def score_func(self, theta: np.ndarray) -> np.ndarray:
        """Compute the score. 

        Args:
            theta (np.ndarray): parameter/particles

        Returns:
            np.ndarray: grad_theta log p(theta, D)
        """
        score = 0
        for w, (mu, var) in zip(self.mixture_wts, self.params): 
            score += w * self._gauss_pdf(theta, mu, var) * (-(theta - mu) / var)
        score = score / self.evaluate_density(theta)
        return score

 
    def plot(
        self,
        x: np.ndarray,
        ax: AXIS = None,
        lw: float = 2,
        ls: str = '--',
        c: str = 'C0',
        label: str = None) -> None:
        """Plot the mixture distribution.

        Args:
            x (np.ndarray): the points on which to evaluate the density. 
            ax (AXIS, optional): the axis on which to plot. Defaults to None.
            lw (float, optional): the plot line width. Defaults to 2.
            ls (str, optional): the line style. Defaults to '--'.
            c (str, optional): the color. Defaults to 'C0'.
            label (str, optional): the label. Defaults to None.
        """
        density = self.evaluate_density(x)
        
        ax = ax if ax is not None else plt.gca()
        ax.plot(x, density, c=c, ls=ls, lw=lw, label=label)
    


class LatentModel(object):

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        theta0: np.ndarray,
        var: float = 1.0,
        prior_mean: np.ndarray = np.zeros(3),
        prior_cov: np.ndarray = np.eye(3),
        seed: int = 0) -> None:
        """Initialize a model of the following generative process: 

        (a, b, c) = theta ~ p(theta) = N(theta; mu, cov)
        f = ax^c + b
        epsilon ~ N(0, sigma^2)
        y = f + epsilon

        Args:
            X (np.ndarray): inputs, 1d
            y (np.ndarray): model outputs
            theta0 (np.ndarray): initial estimate of (a, b, c) (or population of estimates)
            var (float, optional): sigma^2. Defaults to 1.0.
            prior_mean (np.ndarray, optional): mean of p(theta). Defaults to np.zeros(3).
            prior_cov (np.ndarray, optional): covariance of p(theta). Defaults to np.eye(3).
            seed (int, optional): random seed. Defaults to 0.
        """
        np.random.seed(seed)
        self.m = X.shape[0] # size of data set
        self.X = X
        self.y = y
        self.var = var
        self.n = theta0.shape[0] # number of particles/samples
        self.theta = np.copy(theta0)
        self.mu_theta = prior_mean
        self.var_theta = prior_cov
        
    def grad_logli(self, theta: np.ndarray) -> np.ndarray:
        """Compute the MLE gradient

        Args:
            theta (np.ndarray): current parameter/latent estimate.

        Returns:
            np.ndarray: grad_theta log p(D | theta)
        """
        a, b, c = theta 
        xc = self.X ** c # m x 1
        
        f = a * xc + b
        # Gaussian derivative
        dgauss = (self.y - f) / self.var # m x 1
        
        da = dgauss * xc
        db = dgauss
        dc = dgauss * xc * np.log(self.X)
        dtheta = np.hstack([da[:, None], db[:, None], dc[:, None]]) # m x 3
        
        return np.mean(dtheta, axis=0) # average over data

    def score_func(self, theta: np.ndarray) -> np.ndarray:
        """Compute the score function

        Args:
            theta (np.ndarray): current set of theta particles (n x 3)

        Returns:
            np.ndarray: grad_theta log p(theta, D)
        """
        
        a, b, c = theta[:, 0], theta[:, 1], theta[:, 2] # each n x 1
        
        xc = self.X[None, :] ** c[:, None] # n x m
        f = a[:, None] * xc + b[:, None] # n x m

        # Gaussian derivative
        dgauss = (self.y - f) / self.var 
        # grad_theta log p(theta)
        dprior = -(theta - self.mu_theta) @ np.linalg.inv(self.var_theta) 
        # grad_theta log p(y | theta) + grad_theta log p(theta)
        da = dgauss * xc + dprior[:, 0, None] # n x m
        db = dgauss + dprior[:, 1, None] # n x m
        dc = dgauss * xc * np.log(self.X[None, :]) + dprior[:, 2, None] # n x m
        
        # stack gradients
        dtheta = np.stack([da, db, dc], axis=-1) # n x m x 3
        # average across data
        dtheta = np.mean(dtheta, axis=1) # n x 3
        
        return dtheta
        