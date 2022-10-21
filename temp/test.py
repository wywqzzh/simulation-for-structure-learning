import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
from numpy import *
from copy import deepcopy


def my_gaussian(x_points, mu, sigma):
    """
    Returns normalized Gaussian estimated at points `x_points`, with parameters: mean `mu` and std `sigma`

    Args:
        x_points (numpy array of floats): points at which the gaussian is evaluated
        mu (scalar): mean of the Gaussian
        sigma (scalar): std of the gaussian

    Returns: 
        (numpy array of floats) : normalized Gaussian evaluated at `x`
    """
    px = np.exp(- 1 / 2 / sigma ** 2 * (mu - x_points) ** 2)

    px = px / px.sum()  # this is the normalization part with a very strong assumption, that
    # x_points cover the big portion of probability mass around the mean.
    # Please think/discuss when this would be a dangerous assumption.

    return px


def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    dim = len(mu)
    t = np.zeros(size)
    for i in range(size):
        temp_x = x[i].reshape(2, -1)
        # norm_const = 1 / np.sqrt(np.power(2 * np.pi, dim) * np.linalg.det(sigma))
        t1 = temp_x - mu
        t2 = np.exp(-1 / 2 / np.matmul(np.matmul(t1.T, sigma), t1))[0]
        t[i] = t2
    t = t / np.sum(t)
    return t


F = True

PC = [0.222, 0.778]


def get_PS1GS1_PX2GS2():
    sigma1 = 3
    sigma2 = 10
    s1 = np.arange(-40, 41, 1)
    s2 = np.arange(-40, 41, 1)
    x1 = np.arange(-50, 51, 1)
    x2 = np.arange(-50, 51, 1)
    PX1GS1 = np.zeros((len(s1), len(x1)))
    PX2GS2 = np.zeros((len(s2), len(x2)))
    for i in range(len(s1)):
        p = my_gaussian(x1, s1[i], sigma1)
        PX1GS1[i, :] = p
        p = my_gaussian(x1, s2[i], sigma2)
        PX2GS2[i, :] = p

    if need_plot == True:
        df = pd.DataFrame(PX1GS1)
        df.columns = np.around(x1, decimals=2)
        df.index = np.around(s1, decimals=2)
        ax = sns.heatmap(df, xticklabels=10, yticklabels=10)

        plt.title("P(x1|s1)")
        plt.xlabel("x1")
        plt.ylabel("s1")
        plt.tight_layout()
        plt.show()

        df = pd.DataFrame(PX2GS2)
        df.columns = np.around(x2, decimals=2)
        df.index = np.around(s2, decimals=2)
        ax = sns.heatmap(df, xticklabels=10, yticklabels=10)

        plt.title("P(x2|s2)")
        plt.xlabel("x2")
        plt.ylabel("s2")
        plt.tight_layout()
        plt.show()

    return PX1GS1, PX2GS2, x1, x2, s1, s2


def get_PS1S2():
    simgas = 10
    C = 2
    s1 = np.arange(-40, 41, 1)
    PS1 = my_gaussian(s1, 0, simgas)
    s2 = np.arange(-40, 41, 1)
    PS2 = my_gaussian(s2, 0, simgas)
    PS1S2 = np.zeros((len(s1), len(s2)))
    for i in range(len(s1)):
        for j in range(len(s2)):
            PS1S2[i][j] = PS1[i] * PS2[j]
    PS1S2_C2 = deepcopy(PS1S2)
    if need_plot == True:
        df = pd.DataFrame(PS1S2)
        df.columns = np.around(s2, decimals=2)
        df.index = np.around(s1, decimals=2)
        ax = sns.heatmap(df, xticklabels=10, yticklabels=10)
        plt.title("P(s1,s2|C=2)")
        plt.xlabel("s2")
        plt.ylabel("s1")
        plt.tight_layout()
        plt.show()
        # C=1

    simgas = 2
    S = np.arange(-40, 41, 1)
    s1 = np.arange(-40, 41, 1)
    s2 = np.arange(-40, 41, 1)
    PS = my_gaussian(S, 0, 15)
    PS1S2_C1 = np.zeros((len(S), len(s1), len(s2)))
    for k, s in enumerate(S):
        PS1 = my_gaussian(s1, s, simgas)
        PS2 = my_gaussian(s2, s, simgas)
        temp = np.zeros((len(s1), len(s2)))
        for i in range(len(s1)):
            for j in range(len(s2)):
                temp[i][j] = PS1[i] * PS2[j] * PS[k]
        PS1S2_C1[k, :, :] = deepcopy(temp)
        # temp_plot(temp, s1, s2)
    PS1S2_C1 = np.sum(PS1S2_C1, axis=0)
    # PS1S2_C1 /= np.sum(PS1S2_C1)
    if need_plot == True:
        df = pd.DataFrame(PS1S2_C1)
        df.columns = np.around(s2, decimals=2)
        df.index = np.around(s1, decimals=2)
        ax = sns.heatmap(df, xticklabels=10, yticklabels=10)
        plt.title("P(s1,s2|C=1)")
        plt.xlabel("s2")
        plt.ylabel("s1")
        plt.tight_layout()
        plt.show()

    return PS1S2_C1, PS1S2_C2


def temp_plot(data, s1, s2):
    df = pd.DataFrame(data)
    df.columns = np.around(s1, decimals=2)
    df.index = np.around(s2, decimals=2)
    ax = sns.heatmap(df, fmt='.2', xticklabels=10, yticklabels=10)
    plt.xlabel("s2")
    plt.ylabel("s1")
    plt.tight_layout()
    plt.show()


def Plot():
    PX1GS1, PX2GS2, x1, x2, s1, s2 = get_PS1GS1_PX2GS2()
    PS1S2_C1, PS1S2_C2 = get_PS1S2()

    x1_ = -5
    x2_ = 15
    index_x1 = np.where(x1 == x1_)[0][0]
    index_x2 = np.where(x2 == x2_)[0][0]

    pX1GS1 = PX1GS1[:, index_x1]
    pX2GS2 = PX2GS2[:, index_x2]

    temp = np.zeros(PS1S2_C1.shape)
    for i in range(len(s1)):
        for j in range(len(s2)):
            temp[i][j] = pX1GS1[i] * pX2GS2[j]

    temp_plot(temp, s1, s2)
    PS1S2GX_C1 = PS1S2_C1 * temp * PC[0]
    # temp_plot(PS1S2GX_C1, s1, s2)
    # PS1S2GX_C1 /= np.sum(PS1S2GX_C1)
    PS1S2GX_C2 = PS1S2_C2 * temp * PC[1]
    # temp_plot(PS1S2GX_C2, s1, s2)
    # PS1S2GX_C2 /= np.sum(PS1S2GX_C2)
    PS1S2GX = PS1S2GX_C1 + PS1S2GX_C2
    # PS1S2GX /= np.sum(PS1S2GX)

    # PS2GX_C1 = np.sum(PS1S2GX_C1, axis=0)
    # PS2GX_C2 = np.sum(PS1S2GX_C2, axis=0)
    # PS2GX = np.sum(PS1S2GX, axis=0)
    
    index_s1=np.where(s1==-3)[0][0]
    PS2GX_C1 = PS1S2GX_C1[index_s1]
    PS2GX_C2 = PS1S2GX_C2[index_s1]
    PS2GX = PS1S2GX[index_s1]
    
    PS2GX_C1 /= np.sum(PS2GX_C1)
    PS2GX_C2 /= np.sum(PS2GX_C2)
    PS2GX /= np.sum(PS2GX)

    ax = plt.gca()
    plt.plot(s1, pX1GS1, color="pink", linestyle="--")
    plt.plot(s1, pX2GS2, color="red", linestyle="--")
    plt.plot(s2, PS2GX_C1, color="#00ffff")
    plt.plot(s2, PS2GX_C2, color="#0066ff")
    plt.plot(s2, PS2GX, color="#000099")
    x_major_locator = MultipleLocator(20)
    ax.xaxis.set_major_locator(x_major_locator)

    # plt.legend(["Likehood of s1,based on x1", "Likehood of s2,based on x2", "Posterior over s2 when C=1",
    #             "Posterior over s2 when C=2", "Posterior over s2 when you dont know C"])
    plt.show()


if __name__ == '__main__':
    global need_plot
    need_plot = True

    Plot()
