import numpy as np
import matplotlib.pyplot as plt

def load(inputPath):
    f=open(inputPath,"r")
    m=[]
    labels=np.array([])
    for line in f:
        parts=line.split(",")
        column = np.array(parts[:6],dtype=float).reshape(6,1)
        m.append(column)
        labels=np.append(labels,int(parts[6].strip()))
    return np.hstack(m),labels


def plot_hist(D,L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    for dIdx in range(6):
        plt.figure()
        plt.xlabel("Feature %d"%(dIdx+1))
        plt.hist(D0[dIdx, :], bins=10, density=True, alpha=0.4, label='False')
        plt.hist(D1[dIdx, :], bins=10, density=True, alpha=0.4, label='True')
        plt.legend()
        #plt.savefig('hist_feature_%d.png'%(dIdx+1))
    plt.show()


def plot_scatter(D,L):
    #plot scatter for last 2 feature analysis
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    x=4

    plt.figure()
    plt.xlabel("Feature %d" %(x+1))
    plt.ylabel("Feature %d" % (x+2))
    plt.scatter(D0[x, :], D0[x+1, :], label='False')
    #plt.savefig('scatter_feature_%d-%d_False.png' % (x + 1, x + 2))
    plt.show()

    plt.figure()
    plt.xlabel("Feature %d" %(x+1))
    plt.ylabel("Feature %d" % (x+2))
    plt.scatter(D1[x, :], D1[x+1, :], label='True',color='orange')
    #plt.savefig('scatter_feature_%d-%d_True.png' % (x + 1, x + 2))
    plt.show()

    plt.figure()
    plt.xlabel("Feature %d" %(x+1))
    plt.ylabel("Feature %d" % (x+2))
    plt.scatter(D0[x,:],D0[x+1,:], label='False')
    plt.scatter(D1[x,:],D1[x+1,:], label='True',color='orange')
    plt.legend()
    #plt.savefig('scatter_feature_%d-%d.png' % (x+1,x+2))
    plt.show()