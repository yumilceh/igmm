"""
Created on June,2017

@author: Andreas Gerken

This class extends the gmm class by Juan Manuel Acevedo Valle by the ability
to sample conditionally from the gaussians. pypr is used for this.
"""

from igmm import IGMM
import pypr.clustering.gmm as pypr_gmm
import numpy as np

class IGMM_COND(IGMM):
    def __init__(self, min_components=3,
                 max_step_components=30,
                 max_components=60,
                 a_split=0.8,
                 forgetting_factor=0.05,
                 plot=False, plot_dims=[0, 1]):
        IGMM.__init__(self, min_components,
                 max_step_components,
                 max_components,
                 a_split, forgetting_factor,
                 plot, plot_dims)

    def cond_dist(self, Y):
        return pypr_gmm.cond_dist(np.array(Y), self.means_, self.covariances_, self.weights_)

    def sample_cond_dist(self, Y, n_samples):

        # get the conditional distribution
        (con_means, con_covariances, con_weights) = self.cond_dist(Y)

        #sample from the conditional distribution
        samples = pypr_gmm.sample_gaussian_mixture(con_means, con_covariances, con_weights, n_samples)

        # find the columns where the nans are
        nan_cols = np.where(np.isnan(Y))[0]

        # extend the input to the length of the samples
        full_samples = np.tile(Y, (n_samples, 1))

        #copy the sample columns
        full_samples[:,nan_cols] = samples

        return full_samples
