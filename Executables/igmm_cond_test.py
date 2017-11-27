from igmm_cond import IGMM_COND
import numpy as np
import matplotlib.pyplot as plt

# Number of samples per component
n_samples_1 = 150
n_samples_2 = 600

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0.1, -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples_1, 2), C),
          .7 * np.random.randn(n_samples_2 , 2) + np.array([-1, 3])]

# create and train the model
model = IGMM_COND(min_components=3, forgetting_factor=0.5)
model.train(X)

# plotting
fig = plt.figure(figsize=(18,15))
ax1 = plt.subplot(111)

# plot the original data points
ax1.plot(X[:,0], X[:,1], 'ok', alpha = 0.2, label="original data")

# sample from the gmm with a fixed x=-0.5, 50 samples
X_new = model.sample_cond_dist([-0.5,np.nan], 50)
ax1.plot(X_new[:,0], X_new[:,1], 'og', alpha = 0.6, label = "new Data with fixed x at -0.5")

# sample from the gmm with a fixed y=3.5, 50 samples
X_new = model.sample_cond_dist([np.nan,3.5], 50)
ax1.plot(X_new[:,0], X_new[:,1], 'or', alpha = 0.6, label = "new Data with fixed y at 3.5")

# plot model elypsis
ax1 = model.plot_gmm_projection(0, 1, axes=ax1)

plt.legend()
plt.show()
