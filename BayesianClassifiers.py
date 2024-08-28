import numpy as np
import matplotlib.pyplot as plt
from GaussianFit import logpdf_GAU_ND,covariance
from utils import *
from DimensionalityReduction import compute_SW

def compute_MVG_llr(DTR,LTR,DTE):
    DTRTrue = DTR[:, LTR == 1]
    DTRFalse = DTR[:, LTR == 0]
    muTrue = DTRTrue.mean(axis=1)
    muFalse = DTRFalse.mean(axis=1)
    CTrue = covariance(DTRTrue)
    CFalse = covariance(DTRFalse)
    llr = logpdf_GAU_ND(DTE, muTrue, CTrue) - logpdf_GAU_ND(DTE, muFalse, CFalse)
    return llr

def compute_naive_llr(DTR,LTR,DTE):
    DTRTrue = DTR[:, LTR == 1]
    DTRFalse = DTR[:, LTR == 0]
    muTrue = DTRTrue.mean(axis=1)
    muFalse = DTRFalse.mean(axis=1)
    CTrue = covariance(DTRTrue)*np.eye(DTR.shape[0])
    CFalse = covariance(DTRFalse)*np.eye(DTR.shape[0])
    llr = logpdf_GAU_ND(DTE, muTrue, CTrue) - logpdf_GAU_ND(DTE, muFalse, CFalse)
    return llr

def compute_tied_llr(DTR,LTR,DTE):
    DTRTrue = DTR[:, LTR == 1]
    DTRFalse = DTR[:, LTR == 0]
    muTrue = DTRTrue.mean(axis=1)
    muFalse = DTRFalse.mean(axis=1)
    C= compute_SW(DTR, LTR, [0, 1])
    llr = logpdf_GAU_ND(DTE, muTrue, C) - logpdf_GAU_ND(DTE, muFalse, C)
    return llr
def error_rate_accuracy(llr,LVAL):
    l = np.where(llr >= 0, 1, 0)
    error= np.sum(l!= LVAL)
    error_rate=error / (LVAL.size )* 100
    accuracy= 100-error_rate
    return error_rate, accuracy

def MVG_classfier(DTR,LTR,DVAL,LVAL):
    llr=compute_MVG_llr(DTR,LTR,DVAL)
    print("MVG : error rate: %f, accuracy : %f" % error_rate_accuracy(llr,LVAL))

def tied_classfier(DTR,LTR,DVAL,LVAL):
    llr=compute_tied_llr(DTR,LTR,DVAL)
    print("tied : error rate: %f, accuracy : %f" % error_rate_accuracy(llr,LVAL))

def naive_classfier(DTR,LTR,DVAL,LVAL):
    llr=compute_naive_llr(DTR,LTR,DVAL)
    print("naive : error rate: %f, accuracy : %f" % error_rate_accuracy(llr,LVAL))

def correlationMatrixes(D):
    C=covariance(D)
    Corr = C / (vcol(C.diagonal() ** 0.5) * vrow(C.diagonal() ** 0.5))
    return Corr
