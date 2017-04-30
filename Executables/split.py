'''
Created on Sep 13, 2016

@author: Juan Manuel Acevedo Valle
'''

if __name__ == '__main__':
    import os,sys
    sys.path.append(os.getcwd())
    
    import numpy as np
    from matplotlib import pyplot as plt

    from igmm import IGMM

    # Number of samples per component
    n_samples = 500
    
    # Generate random sample, two components
    np.random.seed(0)
    C = np.array([[0., -0.1], [1.7, .4]])
    X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
              .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

    model = IGMM(min_components=2,a_split=0.5)
    model.train(X)
    
    #Model computed with three Gaussians
    fig1, ax1 = plt.subplots(1,1)
    ax1 = model.plot_gmm_projection(0, 1, axes=ax1)
    
    model.splitGaussian(1)
    
    #Model merging similar Gaussians
    fig2, ax2 = plt.subplots(1,1)
    ax2 = model.plot_gmm_projection(0, 1, axes=ax2)
    
    #===========================================================================
    # #Model computed with two Gaussians
    # model = ILGMM(min_components=2)
    # model.train(X)
    # fig3, ax3 = initializeFigure()
    # fig3, ax3 = model.plot_gmm_projection(fig3, ax3, 0, 1)
    #===========================================================================
    
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    #===========================================================================
    # ax3.relim()
    # ax3.autoscale_view()
    #===========================================================================
    plt.draw()
    plt.show()
    
    