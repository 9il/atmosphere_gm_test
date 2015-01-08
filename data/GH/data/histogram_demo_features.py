"""
Demo of the histogram (hist) function with a few features.

In addition to the basic histogram, this demo shows a few optional features:

    * Setting the number of data bins
    * The ``normed`` flag, which normalizes bin heights so that the integral of
      the histogram is 1. The resulting histogram is a probability density.
    * Setting the face color of the bars
    * Setting the opacity (alpha value).

"""
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pylab import loadtxt

lll = 'n_ 1 _nu_ 2 _alpha_ 3 _beta_ 1 _mu_ 1 _lambda_ 1'
fun_data = loadtxt(lll + '.csv.txt', unpack=True)
his_data = loadtxt(lll + '.csv.rn', unpack=True)
his_data2 = loadtxt(lll + '.csv', unpack=True)

# print(fun_data)


num_bins = 50
# the histogram of the data
# n2, bins2, patches2 = plt.hist(his_data2, num_bins, normed=1, facecolor='b', alpha=0.5)
n, bins, patches = plt.hist(his_data, num_bins, normed=1, facecolor='g', alpha=0.4)
# add a 'best fit' line
plt.plot(fun_data[0], fun_data[1], 'r')
plt.xlabel('Smarts')
plt.ylabel('Probability')
# plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()


