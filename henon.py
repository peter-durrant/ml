"""Program to generate the Hénon map time series and plot the data"""
import sys
import matplotlib.pyplot as plt
import numpy as np
# get the projection='3d' to work
from mpl_toolkits.mplot3d import Axes3D
del Axes3D

# Hénon map equation:
# x(t) = 1 - ax(t-1)^2 + bx(t-2)

def generate(a, b, numpoints):
    """Generate the Hénon map time series (1D) and return the number of requested points"""
    data = [0]*numpoints

    for i in range(3, numpoints):
        data[i] = 1 - (a * data[i - 1]*data[i - 1]) + (b * data[i - 2])

    return data

def embed(data):
    """Turn a 1D series into a 3D embedding and return the series(t), series(t-1), series(t-2)"""
    data_lag0 = data[:-2]
    data_lag1 = np.roll(data, -1)[:-2].flatten()
    data_lag2 = np.roll(data, -2)[:-2].flatten()
    return data_lag0, data_lag1, data_lag2

def plot(embeddeddata):
    """Plot a 3D embedding"""
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    ax.set_xlabel('t')
    ax.set_ylabel('t-1')
    ax.set_zlabel('t-2')
    ax.set_title('Hénon Map')

    ax.plot3D(embeddeddata[0], embeddeddata[1], embeddeddata[2], ',')

    plt.show()

def main():
    """Generate the time series, embed it into 3D, plot the attractor on a 3D scatter plot"""
    timeseriesdata = generate(1.4, 0.3, 10000)
    embeddeddata = embed(timeseriesdata)
    plot(embeddeddata)

if __name__ == '__main__':
    sys.exit(main())
