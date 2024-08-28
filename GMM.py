import numpy
import scipy
import scipy.special
import matplotlib.pyplot as plt

from utils import *
from GaussianFit import logpdf_GAU_ND
from modelEvaluation import *


def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C


def logpdf_GMM(X, gmm):
    S = []

    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)

    S = numpy.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens


def smooth_covariance_matrix(C, psi):
    U, s, Vh = numpy.linalg.svd(C)
    s[s < psi] = psi
    CUpd = U @ (vcol(s) * U.T)
    return CUpd


# X: Data matrix
# gmm: input gmm
# covType: 'Full' | 'Diagonal' | 'Tied'
# psiEig: factor for eignvalue thresholding
#
# return: updated gmm
def train_GMM_EM_Iteration(X, gmm, covType='Full', psiEig=None):
    assert (covType.lower() in ['full', 'diagonal', 'tied'])
    # E-step
    S = []

    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)

    S = numpy.vstack(S)  # Compute joint densities f(x_i, c), i=1...n, c=1...G
    logdens = scipy.special.logsumexp(S, axis=0)  # Compute marginal for samples f(x_i)

    # Compute posterior for all clusters - log P(C=c|X=x_i) = log f(x_i, c) - log f(x_i)) - i=1...n, c=1...G
    # Each row for gammaAllComponents corresponds to a Gaussian component
    # Each column corresponds to a sample (similar to the matrix of class posterior probabilities in Lab 5, but here the rows are associated to clusters rather than to classes
    gammaAllComponents = numpy.exp(S - logdens)

    # M-step
    gmmUpd = []
    for gIdx in range(len(gmm)):
        # Compute statistics:
        gamma = gammaAllComponents[gIdx]  # Extract the responsibilities for component gIdx
        Z = gamma.sum()
        F = vcol((vrow(gamma) * X).sum(1))  # Exploit broadcasting to compute the sum
        S = (vrow(gamma) * X) @ X.T
        muUpd = F / Z
        CUpd = S / Z - muUpd @ muUpd.T
        wUpd = Z / X.shape[1]
        if covType.lower() == 'diagonal':
            CUpd = CUpd * numpy.eye(X.shape[0])  # An efficient implementation would store and employ only the diagonal terms, but is out of the scope of this script
        gmmUpd.append((wUpd, muUpd, CUpd))

    if covType.lower() == 'tied':
        CTied = 0
        for w, mu, C in gmmUpd:
            CTied += w * C
        gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]

    if psiEig is not None:
        gmmUpd = [(w, mu, smooth_covariance_matrix(C, psiEig)) for w, mu, C in gmmUpd]

    return gmmUpd


# Train a GMM until the average dela log-likelihood becomes <= epsLLAverage
def train_GMM_EM(X, gmm, covType='Full', psiEig=None, epsLLAverage=1e-6, verbose=True):
    llOld = logpdf_GMM(X, gmm).mean()
    llDelta = None
    if verbose:
        print('GMM - it %3d - average ll %.8e' % (0, llOld))
    it = 1
    while (llDelta is None or llDelta > epsLLAverage):
        gmmUpd = train_GMM_EM_Iteration(X, gmm, covType=covType, psiEig=psiEig)
        llUpd = logpdf_GMM(X, gmmUpd).mean()
        llDelta = llUpd - llOld
        if verbose:
            print('GMM - it %3d - average ll %.8e' % (it, llUpd))
        gmm = gmmUpd
        llOld = llUpd
        it = it + 1

    if verbose:
        print('GMM - it %3d - average ll %.8e (eps = %e)' % (it, llUpd, epsLLAverage))
    return gmm


def split_GMM_LBG(gmm, alpha=0.1, verbose=True):
    gmmOut = []
    if verbose:
        print('LBG - going from %d to %d components' % (len(gmm), len(gmm) * 2))
    for (w, mu, C) in gmm:
        U, s, Vh = numpy.linalg.svd(C)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut


# Train a full model using LBG + EM, starting from a single Gaussian model, until we have numComponents components. lbgAlpha is the value 'alpha' used for LBG, the otehr parameters are the same as in the EM functions above
def train_GMM_LBG_EM(X, numComponents, covType='Full', psiEig=None, epsLLAverage=1e-6, lbgAlpha=0.1, verbose=True):
    mu, C = compute_mu_C(X)

    if covType.lower() == 'diagonal':
        C = C * numpy.eye(X.shape[0])  # We need an initial diagonal GMM to train a diagonal GMM

    if psiEig is not None:
        gmm = [(1.0, mu, smooth_covariance_matrix(C,psiEig))]  # 1-component model - if we impose the eignevalus constraint, we must do it for the initial 1-component GMM as well
    else:
        gmm = [(1.0, mu, C)]  # 1-component model

    while len(gmm) < numComponents:
        # Split the components
        if verbose:
            print('Average ll before LBG: %.8e' % logpdf_GMM(X, gmm).mean())
        gmm = split_GMM_LBG(gmm, lbgAlpha, verbose=verbose)
        if verbose:
            print('Average ll after LBG: %.8e' % logpdf_GMM(X,gmm).mean())  # NOTE: just after LBG the ll CAN be lower than before the LBG - LBG does not optimize the ll, it just increases the number of components
        # Run the EM for the new GMM
        gmm = train_GMM_EM(X, gmm, covType=covType, psiEig=psiEig, verbose=verbose, epsLLAverage=epsLLAverage)
    return gmm

def GMM_cls_iteration(DTR,LTR,DVAL,numC,covType):
    gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], numC, covType = covType, verbose=False, psiEig = 0.01)
    gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], numC, covType = covType, verbose=False, psiEig = 0.01)

    sllr = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
    return sllr


def GMM_cls(DTR,LTR,DVAL,LVAL,covType,p,Cfn,Cfp):
    x = numpy.array([2,4,8,16,32])
    thresh = binaryCostPredictor(p, Cfn, Cfp)
    dcf=[]
    mindcf=[]
    for numC in x:
        sllr= GMM_cls_iteration(DTR,LTR,DVAL,numC,covType)
        predictions=np.where(sllr > thresh, 1, 0)
        dcf.append(actualDCF(p,Cfn,Cfp,predictions,LVAL))
        mindcf.append(minDCF(sllr,sllr,LVAL,p,Cfn,Cfp))
        print(numC)

    plt.plot(x, dcf, label='DCF', color='r')
    plt.plot(x, mindcf, label='minDCF', color='b')
    plt.scatter(x, dcf, color='r')
    plt.scatter(x, mindcf, color='b')
    plt.legend()
    plt.title("GMM diagonal")
    plt.savefig("GMM_diagonal")
    plt.show()

