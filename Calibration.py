
from LogisticRegression import *
from modelEvaluation import *
from utils import *

def extract_train_val_folds_from_ary(X, idx,KFOLD):
    return numpy.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]

def Kfold_calibration(scores,labels,p,Cfn,Cfp):
    KFOLD = 5
    priors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    thresh = binaryCostPredictor(p,Cfn,Cfp)
    predictions= np.where(scores > thresh,1,0)
    dcf= actualDCF(p,Cfn,Cfp,predictions,labels)
    min_dcf= minDCF(scores,scores,labels,p,Cfn,Cfp)
    print(dcf,min_dcf)
    print()

    for prior in priors :
        calibrated_scores = []
        labels_calibration= []
        for foldIdx in range(KFOLD):
            SCAL, SVAL = extract_train_val_folds_from_ary(scores, foldIdx ,KFOLD)
            LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx, KFOLD)

            calibrated_SVAL = weightedLogReg_sllr(vrow(SCAL),LCAL,0,vrow(SVAL),prior)

            calibrated_scores.append(calibrated_SVAL)
            labels_calibration.append(LVAL)

        calibrated_scores= numpy.hstack(calibrated_scores)
        labels_calibrated= numpy.hstack(labels_calibration)

        calibrated_predictions = np.where(calibrated_scores > thresh,1,0)

        dcf=actualDCF(p,Cfn,Cfp,calibrated_predictions,labels_calibrated)
        min_dcf=minDCF(calibrated_scores,calibrated_scores,labels_calibrated,p,Cfn,Cfp)
        print(prior,f"{dcf:.4f}")



def fusion(scores1,scores2,scores3,labels,p,Cfn,Cfp):
    KFOLD = 5
    priors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    thresh = binaryCostPredictor(p,Cfn,Cfp)

    for prior in priors :
        calibrated_scores = []
        labels_calibration= []
        for foldIdx in range(KFOLD):
            SCAL1, SVAL1 = extract_train_val_folds_from_ary(scores1, foldIdx ,KFOLD)
            SCAL2, SVAL2 = extract_train_val_folds_from_ary(scores2, foldIdx, KFOLD)
            SCAL3, SVAL3 = extract_train_val_folds_from_ary(scores3, foldIdx, KFOLD)
            LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx, KFOLD)

            SCAL = numpy.vstack([SCAL1, SCAL2, SCAL3])
            SVAL = numpy.vstack([SVAL1, SVAL2, SVAL3])
            calibrated_SVAL = weightedLogReg_sllr(SCAL,LCAL,0,SVAL,prior)

            calibrated_scores.append(calibrated_SVAL)
            labels_calibration.append(LVAL)

        calibrated_scores= numpy.hstack(calibrated_scores)
        labels_calibrated= numpy.hstack(labels_calibration)

        calibrated_predictions = np.where(calibrated_scores > thresh,1,0)

        dcf=actualDCF(p,Cfn,Cfp,calibrated_predictions,labels_calibrated)
        min_dcf=minDCF(calibrated_scores,calibrated_scores,labels_calibrated,p,Cfn,Cfp)
        print(prior,f"{dcf:.4f}")


def get_calibrated_scores(scores,prior,labels):
    calibrated_scores = weightedLogReg_sllr(vrow(scores), labels, 0, vrow(scores), prior)
    return calibrated_scores

def get_fusion_scores(scores,prior,labels):
    calibrated_scores = weightedLogReg_sllr(scores, labels, 0, scores, prior)
    return calibrated_scores

def calibrated_models_bayes_error_plot(logreg,SVM,GMM,labels,p,Cfn,Cfp):

    thresh = binaryCostPredictor(p, Cfn, Cfp)
    predictions = np.where(logreg > thresh, 1, 0)
    print(actualDCF(p,Cfn,Cfp,predictions,labels),minDCF(logreg,logreg,labels,p,Cfn,Cfp))
    predictions = np.where(SVM > thresh, 1, 0)
    print(actualDCF(p, Cfn, Cfp,predictions, labels), minDCF(SVM, SVM, labels, p, Cfn, Cfp))
    predictions = np.where(GMM > thresh, 1, 0)
    print(actualDCF(p, Cfn, Cfp, predictions,labels), minDCF(GMM, GMM, labels, p, Cfn, Cfp))

    x,y,n= -4,4,31
    effPriorLogOdds = np.linspace(x, y, n)
    pi=1/(1 + np.exp(-effPriorLogOdds))
    logreg_dcf=[]
    logreg_mindcf=[]
    SVM_dcf=[]
    SVM_mindcf=[]
    GMM_dcf=[]
    GMM_mindcf=[]
    for i in pi:
        thresh= binaryCostPredictor(i,Cfn,Cfp)
        logreg_p = np.where(logreg > thresh, 1, 0)
        logreg_dcf.append(actualDCF(i,Cfn,Cfp,logreg_p,labels))
        logreg_mindcf.append(minDCF(logreg,logreg,labels,i,Cfn,Cfp))

        SVM_p = np.where(SVM > thresh, 1, 0)
        SVM_dcf.append(actualDCF(i,Cfn,Cfp,SVM_p,labels))
        SVM_mindcf.append(minDCF(SVM,SVM,labels,i,Cfn,Cfp))

        GMM_p = np.where(GMM > thresh, 1, 0)
        GMM_dcf.append(actualDCF(i,Cfn,Cfp,GMM_p,labels))
        GMM_mindcf.append(minDCF(GMM,GMM,labels,i,Cfn,Cfp))

    plt.plot(effPriorLogOdds, logreg_dcf, label='quadratic logreg DCF', color ='r')
    plt.plot(effPriorLogOdds, logreg_mindcf,linestyle='--', label='quadratic logreg minDCF', color ='r')
    plt.plot(effPriorLogOdds, SVM_dcf, label='SVM RBF DCF', color ='b')
    plt.plot(effPriorLogOdds, SVM_mindcf,linestyle='--', label='SVM RBF minDCF', color ='b')
    plt.plot(effPriorLogOdds, GMM_dcf, label='diagonal GMM DCF', color ='g')
    plt.plot(effPriorLogOdds, GMM_mindcf,linestyle='--', label='diagonal GMM minDCF', color ='g')

    plt.ylim([0, 1.1])
    plt.xlim([x,y])
    plt.legend(loc='upper left')
    plt.title("calibrated models Bayes error plot")
    plt.savefig("calibrated models Bayes error plot")
    plt.show()

