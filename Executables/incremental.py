'''
Created on Sep 13, 2016

@author: Juan Manuel Acevedo Valle
'''

if __name__ == '__main__':
    import os, sys
    import itertools

    sys.path.append("../../")

    import numpy as np
    import matplotlib.pyplot as plt

    from igmm import IGMM

    # Number of samples per component
    n_samples = 500
    colors = ['k', 'r', 'g', 'b', 'c', 'm', 'y']
    color_iter = itertools.cycle(colors)

    # Generate random sample, two components
    np.random.seed(0)
    C = np.array([[0., -0.1], [1.7, .4]])
    X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
              .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

    model = IGMM(min_components=2, max_components=5, plot=True, plot_dims=[0, 1])
    model.train(X)

    #  Y_  = model.model.predict(X)

    # Model computed with three Gaussians
    n_samples2 = 150
    C2 = np.array([[0., -0.1], [0.2, 0.4]])
    X2 = np.r_[np.dot(np.random.randn(n_samples2, 2), 0.5 * C),
               .7 * np.random.randn(n_samples, 2) + np.array([-5, 4]),
               .2 * np.random.randn(n_samples2, 2) + np.array([-2, 1]),
               .5 * np.random.randn(n_samples2, 2) + np.array([1, 3]),
               .5 * np.random.randn(n_samples2, 2) + np.array([1, 3]),
               .4 * np.random.randn(n_samples, 2) + np.array([-2, -0.5])]
    model.train(X2)

    # Model merging similar Gaussians
    X3 = np.r_[np.dot(np.random.randn(n_samples2, 2), 0.1 * C),
               .1 * np.random.randn(n_samples, 2) + np.array([-5, 4]),
               .1 * np.random.randn(n_samples2, 2) + np.array([-2, 1]),
               .2 * np.random.randn(n_samples2, 2) + np.array([1, 3]),
               .3 * np.random.randn(n_samples2, 2) + np.array([1, 3]),
               .4 * np.random.randn(n_samples, 2) + np.array([-2, 3]),
               .4 * np.random.randn(n_samples, 2) + np.array([-6, 0]),
               .4 * np.random.randn(n_samples, 2) + np.array([4, 5])]

    model.train(X3)

    plt.draw()
    plt.show()
