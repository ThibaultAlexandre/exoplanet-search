# exoplanet-search

In this jupyter notebook (main.ipynb), we train a convolutional neural network to detect the presence of an exoplanet in the variation of the flux intensity of a star. The data are derived from observations made by the NASA Kepler space telescope and are accessible at https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data.

Here is the explanation given on Kaggle : "As you can imagine, planets themselves do not emit light, but the stars that they orbit do. If said star is watched over several months or years, there may be a regular 'dimming' of the flux (the light intensity). This is evidence that there may be an orbiting body around the star; such a star could be considered to be a 'candidate' system. Further study of our candidate system, for example by a satellite that captures light at a different wavelength, could solidify the belief that the candidate can in fact be 'confirmed'."

We define a neural network with :

- Two 1d convolution layers (with a maxpooling step and a relu activation function) followed by
- Two fully connected layers (with a sigmoid activation function at the end to do a binary classification)
- We use dropout and L2 regularization techniques to improve generalization.

We process the data with the following steps :

- Standardization : i.e. scale each data to have mean = 0 and variance = 1
- Remove outliers with a certain threshold (like 3)
- Smoothening : Apply a (small) moving average to smooth slightly the data
- Remove high frequency : Apply a Fourier transform, set high frequency (like above 500) to 0, apply inverse Fourier transform

The folder save contains the weights and biases of our trained cnn.
