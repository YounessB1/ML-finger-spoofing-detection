import numpy as np
import matplotlib.pyplot as plt
from utils import *
def logpdf_GAU_ND (X , mu , C):
    Y = []
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mu=vcol(mu)
    t1 = X.shape[0]* np.log(2 * np.pi)
    t2 = np.linalg.slogdet(C)[1]
    for x in X.T:
        x = vcol(x)
        t3 = np.dot(np.dot((x-mu).T,np.linalg.inv(C)),(x-mu))[0,0]
        Y.append(-0.5*(t1+t2+t3))
    return np.array(Y)

def covariance(D):
    if D.ndim == 1:
        D = D.reshape(1,-1)
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return C

def loglikelihood(D,mu,C):
    return logpdf_GAU_ND (D,mu,C).sum()

def gaussian_fit_plots(D,L):
    DFalse = D[:, L == 0]
    DTrue = D[:, L == 1]
    for i in range(6):
        DxFalse = DFalse[i, :]
        DxTrue = DTrue[i, :]
        plt.figure()
        plt.xlabel("Feature %d" % (i + 1))
        plt.hist(DxFalse, bins=10, density=True, alpha=0.4, label='False', color='blue')
        plt.hist(DxTrue, bins=10, density=True, alpha=0.4, label='True', color='red')
        XPlot = np.linspace(-12, 12, 1000)
        YFalse = np.exp(logpdf_GAU_ND(vrow(XPlot), DxFalse.mean(), covariance(DxFalse)))
        YTrue = np.exp(logpdf_GAU_ND(vrow(XPlot), DxTrue.mean(), covariance(DxTrue)))
        plt.plot(XPlot.ravel(), YFalse, color='blue')
        plt.plot(XPlot.ravel(), YTrue, color='red')
        plt.legend()
        plt.savefig("gaussian_fit_feature_%i.png" % (i + 1))
    plt.show()