import numpy as np
import matplotlib.pyplot as plt

def confusionMatrix(predictions,labels,n_classes):
    confusion_matrix = [[0] * n_classes for _ in range(n_classes)]
    labels=labels.astype(int)
    for pred, actual in zip(predictions,labels):
        confusion_matrix[pred][actual] += 1

    return confusion_matrix

def effective_prior(p,Cfn,Cfp):
    ep = (p*Cfn)/((p*Cfn)+((1-p)*Cfp))
    return ep

def binaryCostPredictor(p,Cfn,Cfp):
    thresh=-np.log((p*Cfn)/((1-p)*Cfp))
    return thresh

def DCF(p,Cfn,Cfp,predictions,labels):
    cm=confusionMatrix(predictions,labels,2)
    FNR=cm[0][1]/(cm[0][1]+cm[1][1])
    FPR = cm[1][0] / (cm[1][0] + cm[0][0])
    dcf=p*Cfn*FNR + (1-p)*Cfp*FPR
    return dcf

def actualDCF(p,Cfn,Cfp,predictions,labels):
    dcf=DCF(p,Cfn,Cfp,predictions,labels)
    dcf_dummy=min((p*Cfn),((1-p)*Cfp))
    return dcf/dcf_dummy

def minDCF(llr,thresholds,labels,p,Cfn,Cfp):
    DCFarray = []
    for thresh in thresholds:
        predictions = np.where(llr > thresh, 1, 0)
        DCFarray.append(actualDCF(p, Cfn, Cfp, predictions, labels))
    return min(DCFarray)

def miscalibartion_score(actual_dcf,dcf_min):
    return (abs(actual_dcf - dcf_min)/dcf_min)*100

def modelEvaluation(llr,thresholds,labels,p,Cfn,Cfp ):
    thresh=binaryCostPredictor(p,Cfn,Cfp)
    predictions = np.where(llr > thresh, 1, 0)
    actual_dcf= actualDCF(p, Cfn, Cfp, predictions,labels)
    min_dcf=minDCF(llr,thresholds,labels,p,Cfn,Cfp)
    miscalibration_score= miscalibartion_score(actual_dcf,min_dcf)
    print("%f %f %f "%(actual_dcf,min_dcf,miscalibration_score))

def bayesErrorPlot(x,y,n,llr,thresholds,labels,Cfn,Cfp,title):
    effPriorLogOdds = np.linspace(x, y, n)
    pi=1/(1 + np.exp(-effPriorLogOdds))
    dcf=[]
    mindcf=[]
    for i in pi:
        thresh= binaryCostPredictor(i,Cfn,Cfp)
        predictions = np.where(llr > thresh, 1, 0)
        dcf.append(actualDCF(i,Cfn,Cfp,predictions,labels))
        mindcf.append(minDCF(llr,thresholds,labels,i,Cfn,Cfp))

    plt.plot(effPriorLogOdds, dcf, label='DCF', color ='r')
    plt.plot(effPriorLogOdds, mindcf, label='minDCF', color ='b')
    plt.ylim([0, 1.1])
    plt.xlim([x,y])
    plt.legend(loc='lower left')
    plt.title(title+" Bayes error plot")
    plt.savefig(title+" Bayes error plot")
    plt.show()
