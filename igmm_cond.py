"""
Created on June,2017

@author: Andreas Gerken

This class extends the gmm class by Juan Manuel Acevedo Valle by the ability
to sample conditionally from the gaussians. pypr is used for this.
"""

from igmm import IGMM
from igmm import DynamicParameter
import pypr.clustering.gmm as pypr_gmm
import numpy as np

class IGMM_COND(IGMM):

    def cond_dist(self, Y):
        """ Returns the conditional distribution with some fixed dimensions.

            Keyword arguments:
            Y -- A numpy vector of the same length as the input data samples with either
                values or np.nan. A two dimensional input array could be
                np.array([3, np.nan]). The gmm would be sampled with a fixed first
                dimension of 3.
        """
        return pypr_gmm.cond_dist(np.array(Y), self.means_, self.covariances_, self.weights_)

    def sample_cond_dist(self, Y, n_samples):
        """ Returns conditional samples from the gaussian mixture model.

        Keyword arguments:
        Y -- A numpy vector of the same length as the input data samples with either
            values or np.nan. A two dimensional input array could be
            np.array([3, np.nan]). The gmm would be sampled with a fixed first
            dimension of 3.
        n_samples -- Number of requested samples
        """


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
