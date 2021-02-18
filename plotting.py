import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from distributions import *
Distribution = Union[GaussMixture1D, LatentModel]


def plot_svgd_fig1(xs: list, p_target: Distribution, iter_list: list = [0, -1]) -> None:
    """Reproduce Figure 1 of the SVGD figure"""
    inputs = np.linspace(-12, 12, num=200)
    n = len(iter_list)
    fig, axs = plt.subplots(1, n, figsize=(n * 3, 3), sharey=True)
    
    for i, t in enumerate(iter_list):
        labels = ('Ground Truth', 'SVGD') if i == 0 else (None, None)
        # plot target distribution
        p_target.plot(inputs, ax=axs[i], ls='--', lw=2, c='C1', label=labels[0])
        # plot SVGD estimate (using KDE)
        sns.distplot(xs[t], hist=False, bins=20, kde=True,
             kde_kws = {'shade': False, 'linewidth': 2},
             label = labels[1], color='C2', ax=axs[i], norm_hist=True)
        
        axs[i].set_title(f"{iter_list[i]}th Iteration", fontsize=15)
        axs[i].set_xlim([-12, 12])
        axs[i].set_xticks([-10, 0, 10])
    
    axs[0].set_ylim([-0.01, 0.4])
    axs[0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    axs[0].legend(fontsize=11)
    
    plt.show()

def plot_svgd_fig2(mc_list: list, svgd_list: list) -> None:
    """Reproduce Figure 2 of the SVGD figure"""
    
    assert len(mc_list) == 3 and len(svgd_list) == 3, "incorrect lengths"
    mse_mc_x, mse_mc_x2, mse_mc_cos = mc_list
    mse_svgd_x, mse_svgd_x2, mse_svgd_cos = svgd_list
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # plot E[x]
    axs[0].plot(n_list, mse_mc_x, '-o', label='Monte Carlo')
    axs[0].plot(n_list, mse_svgd_x, '-o', label='SVGD')
    axs[0].legend(fontsize=14)
    axs[0].set_ylabel("$\log_{10}$ MSE", fontsize=16)
    axs[0].set_xlabel("Sample size ($n$)", fontsize=16)
    axs[0].set_title("Estimating $E[x]$", fontsize=18)

    # plot E[x^2]
    axs[1].plot(n_list, mse_mc_x2, '-o')
    axs[1].plot(n_list, mse_svgd_x2, '-o')
    axs[1].set_xlabel("Sample size ($n$)", fontsize=16)
    axs[1].set_title("Estimating $E[x^2]$", fontsize=18)

    # plot E[cos(omega * w + b)]
    axs[2].plot(n_list, mse_mc_cos, '-o')
    axs[2].plot(n_list, mse_svgd_cos, '-o')
    axs[2].set_xlabel("Sample size ($n$)", fontsize=16)
    axs[2].set_title("Estimating $E[cos(\omega x + b)]$", fontsize=18)

    plt.show()


def plot_dists_p2(theta_hist: list, labels: list) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    theta_initial = theta_hist[0]
    theta_final = theta_hist[-1]
    d = theta_final.shape[-1]
    
    # plot initial distributions
    axs[0].set_title("Initial Posteriors", fontsize=15)
    for i, label in enumerate(labels):
        sns.distplot(theta_initial[:, i], hist=False, bins=20, kde=True,
             kde_kws = {'shade': False, 'linewidth': 2},
             label=label, ax=axs[0], color=f'C{i}', norm_hist=True)
    
    # plot final distributions
    axs[1].set_title("Final Posteriors", fontsize=15)
    for i, label in enumerate(labels):
        sns.distplot(theta_final[:, i], hist=False, bins=20, kde=True,
             kde_kws = {'shade': False, 'linewidth': 2},
             label=None, ax=axs[1], color=f'C{i}', norm_hist=True)
    
    axs[0].legend(fontsize=14)
    plt.show()


def plot_posterior_vs_mle_p3(posterior_hist: list, mle_hist: list, labels: list) -> None:
    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    
    posterior_final = posterior_hist[-1] # m x d
    mle_final = posterior_hist[-1] # 1 x d
    d = posterior_final.shape[-1]

    # plot final distributions
    axs.set_title("Posterior vs MLE", fontsize=15)
    for i, label in enumerate(labels):
        sns.distplot(posterior_final[:, i], hist=False, bins=20, kde=True,
             kde_kws = {'shade': False, 'linewidth': 2},
             label=f"$p({label} | D)$", ax=axs, color=f'C{i}', norm_hist=True)
        axs.axvline(x=mle_final[0, i], color=f'C{i}', ls='--', label=f"${label}$ (MLE)")
    
    axs.legend(fontsize=11)
    plt.show()