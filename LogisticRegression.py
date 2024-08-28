import matplotlib.pyplot as plt
import numpy
import scipy
from modelEvaluation import *
from utils import *

def  logReg(DTR,LTR,l):
    def logreg_obj(v):
        w,b = v[0:-1], v[-1]
        z = 2*LTR-1
        reg_term = l/2*numpy.linalg.norm(w)**2
        exponent = -z*(numpy.dot(w.T,DTR)+b)
        sum = numpy.logaddexp(0,exponent).sum()
        return reg_term + sum/DTR.shape[1]
    x,f,_ = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, x0=numpy.zeros(DTR.shape[0] + 1), approx_grad=True)
    return x,f


def weightedLogReg(DTR, LTR, l, pT):
    ZTR = LTR * 2.0 - 1.0  # We do it outside the objective function, since we only need to do it once

    wTar = pT / (ZTR > 0).sum()  # Compute the weights for the two classes
    wNon = (1 - pT) / (ZTR < 0).sum()

    def logreg_obj_with_grad(v):
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        # Calcolo della loss function con stabilizzazione numerica
        loss = numpy.logaddexp(0, -ZTR * s)
        loss[ZTR > 0] *= wTar  # Applica i pesi alla loss function
        loss[ZTR < 0] *= wNon

        # Calcolo del gradiente con stabilizzazione numerica
        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G = G * (1.0 - 1.0 / (1.0 + numpy.exp(-numpy.abs(ZTR * s))))  # Stabilizzazione numerica per il gradiente
        G[ZTR > 0] *= wTar  # Applica i pesi al gradiente
        G[ZTR < 0] *= wNon

        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * numpy.linalg.norm(w) ** 2, numpy.hstack([GW, numpy.array(Gb)])

    x,f,_= scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0=numpy.zeros(DTR.shape[0] + 1))
    return x,f
def logreg_sllr(DTR,LTR,l,DVAL):
    x,f=logReg(DTR,LTR,l)
    w, b = x[0:-1], x[-1]
    piemp = numpy.mean(LTR == 1)
    sllr = numpy.dot(w.T, DVAL) + b - numpy.log(piemp / (1 - piemp))
    return sllr

def weightedLogReg_sllr(DTR,LTR,l,DVAL,p):
    x,f=weightedLogReg(DTR,LTR,l,p)
    w, b = x[0:-1], x[-1]
    sllr = numpy.dot(w.T, DVAL) + b - numpy.log(p/(1 - p))
    return sllr

def quadratic_features(X):
    n_samples = X.shape[1]
    quadratic_features = []

    for i in range(n_samples):
        sample = X[:, i]
        quadratic_sample = []
        for j in range(len(sample)):
            for k in range(j, len(sample)):
                quadratic_sample.append(sample[j] * sample[k])
        quadratic_features.append(quadratic_sample)

    quadratic_features = np.array(quadratic_features).T
    return np.vstack((X, quadratic_features))

def qaudraticlogreg_sllr(DTR,LTR,l,DVAL):
    quadratic_DTR=quadratic_features(DTR)
    qaudratic_DVAL=quadratic_features(DVAL)
    x,f=logReg(quadratic_DTR,LTR,l)
    w, b = x[0:-1], x[-1]
    piemp = numpy.mean(LTR == 1)
    sllr = numpy.dot(w.T,qaudratic_DVAL) + b - numpy.log(piemp / (1 - piemp))
    return sllr

def regularization_plot(DTR,LTR,DVAL,LVAL,p,Cfn,Cfp):
    x=numpy.logspace(-4, 2, 13)
    thresh = binaryCostPredictor(p, Cfn, Cfp)
    threshholds=numpy.linspace(-5,5,100)
    dcf=[]
    mindcf=[]
    for l in x:
        sllr = logreg_sllr(DTR, LTR, l, DVAL)
        #sllr=weightedLogReg_sllr(DTR,LTR,l,DVAL,p)
        #sllr=qaudraticlogreg_sllr(DTR,LTR,l,DVAL)
        predictions=np.where(sllr > thresh, 1, 0)
        dcf.append(actualDCF(p,Cfn,Cfp,predictions,LVAL))
        mindcf.append(minDCF(sllr,threshholds,LVAL,p,Cfn,Cfp))

    plt.xscale('log', base = 10)
    plt.plot(x, dcf, label='DCF', color='r')
    plt.plot(x, mindcf, label='minDCF', color='b')
    plt.scatter(x, dcf, color='r')
    plt.scatter(x, mindcf, color='b')
    plt.legend()
    plt.title("Logistic Regression")
    plt.savefig("logreg")
    plt.show()

def regression_minDCF(DTR,LTR,DVAL,LVAL,p,Cfn,Cfp):
    x=numpy.logspace(-4, 2, 13)
    thresh = binaryCostPredictor(p, Cfn, Cfp)
    threshholds=numpy.linspace(-5,5,100)
    logreg_mindcf=[]
    weighted_logreg_mindcf=[]
    quadratic_logreg_mindcf=[]
    for l in x:
        sllr = logreg_sllr(DTR, LTR, l, DVAL)
        weighted_sllr=weightedLogReg_sllr(DTR,LTR,l,DVAL,p)
        quadratic_sllr=qaudraticlogreg_sllr(DTR,LTR,l,DVAL)
        predictions=np.where(sllr > thresh, 1, 0)
        logreg_mindcf.append(minDCF(sllr,sllr,LVAL,p,Cfn,Cfp))
        weighted_logreg_mindcf.append(minDCF(weighted_sllr,weighted_sllr,LVAL,p,Cfn,Cfp))
        quadratic_logreg_mindcf.append(minDCF(quadratic_sllr,quadratic_sllr,LVAL,p,Cfn,Cfp))

    print("min DCF Logistic Regression %f"%min(logreg_mindcf))
    print("min DCF Prior Weighted Logistic Regression %f" % min(weighted_logreg_mindcf))
    print("min DCF Quadratic Logistic Regression %f" % min(quadratic_logreg_mindcf))