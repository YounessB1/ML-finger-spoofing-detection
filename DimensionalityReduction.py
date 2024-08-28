from utils import *
import numpy as np
import scipy
import matplotlib.pyplot as plt

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def PCA(D,m):
    mu = vcol(D.mean(1))
    DC = D - mu
    C = np.dot(DC, DC.T) / DC.shape[1]
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    #DP = np.dot(P.T, D)
    return P

def PCA_plots(D,P,L,n):
    DP = np.dot(P.T,D)
    DFalse = DP[:, L == 0]
    DTrue = DP[:, L == 1]
    for i in range(n):
        plt.figure()
        plt.title('PCA projection on %d direction' % (i + 1))
        plt.hist(DFalse[i, :], density=True, alpha=0.4, bins=30, label='False')
        plt.hist(DTrue[i, :], density=True, alpha=0.4, bins=30, label='True')
        plt.legend()
        #plt.savefig('PCA projection on %d direction' % (i + 1))
    plt.show()

def compute_SB(D,L,classes):
    mu = vcol(D.mean(1))
    for i in classes:
        muc = vcol(D[:, L == i].mean(1))
        if (i == classes[0]):
            SB = D[:, L == i].shape[1] * ((muc - mu) @ (muc - mu).T)
        else:
            SB += D[:, L == i].shape[1] * ((muc - mu) @ (muc - mu).T)

    SB = SB / D.shape[1]
    return SB

def compute_SW(D,L,classes):
    mu = vcol(D.mean(1))
    for i in classes:
        muc = vcol(D[:, L == i].mean(1))
        DCL = D[:, L == i] - muc
        if (i == classes[0]):
            SW = (DCL @ DCL.T)
        else:
            SW += (DCL @ DCL.T)
    SW= SW / D.shape[1]
    return SW

def LDA(D,L,classes,m):
    SW=compute_SW(D,L,classes)
    SB = compute_SB(D, L, classes)
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    return W

def LDA_plot(D,U,L):
    DL = np.dot(U[:, 0:1].T, D)
    DFalse = DL[:, L == 0]
    DTrue = DL[:, L == 1]
    plt.figure()
    plt.title('LDA projection')
    plt.hist(DFalse[0, :], density=True, alpha=0.4, bins=30, label='False')
    plt.hist(DTrue[0, :], density=True, alpha=0.4, bins=30, label='True')
    plt.legend()
    #plt.savefig('LDA projection')
    plt.show()

def LDA_classifier(DTR,LTR,DVAL,LVAL):
    U = LDA(DTR,LTR,[0,1],1)
    DTR_lda = np.dot(U[:, 0:1].T, DTR)
    DVAL_lda = np.dot(U[:, 0:1].T, DVAL)

    if DTR_lda[0, LTR == 0].mean() > DTR_lda[0, LTR == 1].mean():
        DTR_lda = -DTR_lda
        DVAL_lda = -DVAL_lda

    threshold = (DTR_lda[0, LTR == 0].mean() + DTR_lda[0, LTR == 1].mean()) / 2.0
    #threshold=-0.085
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 1
    PVAL[DVAL_lda[0] < threshold] = 0
    error = np.sum(PVAL != LVAL)
    error_rate=error / (LVAL.size )* 100
    accuracy= 100-error_rate
    print("threshold : %f,samples: %d, error rate: %f, accuracy : %f" % (threshold,LVAL.size,error_rate,accuracy))