# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def Henon(x_t_minus_1, x_t_minus_2, a, b):
    # Hénon map equation:
    # x(t) = 1 - ax(t-1)^2 + bx(t-2)
    x_t = 1 - a * x_t_minus_1 ** 2 + b * x_t_minus_2
    return x_t


def GenerateHenon(a, b, count):
    series = []
    x_t_minus_2 = Henon(0, 0, a, b)
    x_t_minus_1 = Henon(x_t_minus_2, 0, a, b)
    for i in range(count):
        x = Henon(x_t_minus_1, x_t_minus_2, a, b)
        series.append(x)
        x_t_minus_2 = x_t_minus_1
        x_t_minus_1 = x
    return series


def Embed(series, dimension):
    data = []
    for d in range(dimension, 0, -1):
        data.append(series.shift(d - 1))
    data = np.delete(data, list(range(dimension)), axis=1)
    data = pd.DataFrame(data).transpose()
    data = data.rename(columns=CreateEmbedLabels(data, dimension))
    return data


def CreateEmbedLabels(data, dimension):
    labels = {}
    for i in range(0, dimension, 1):
        labels[i] = 'x_'
        if i != dimension - 1:
            labels[i] += f'{{t-{dimension - i - 1}}}'
        else:
            labels[i] += f'{{t}}'
    return labels


def main():
    series = GenerateHenon(1.4, 0.3, 1000)

    embeddedSeries = Embed(pd.Series(series), 3)

    sns.pairplot(embeddedSeries, diag_kind='kde')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        embeddedSeries[embeddedSeries.columns[0]],
        embeddedSeries[embeddedSeries.columns[1]],
        embeddedSeries[embeddedSeries.columns[2]])
    ax.set_xlabel(f'${embeddedSeries.columns[0]}$')
    ax.set_ylabel(f'${embeddedSeries.columns[1]}$')
    ax.set_zlabel(f'${embeddedSeries.columns[2]}$')
    plt.show()


if __name__ == "__main__":
    main()

# %%
