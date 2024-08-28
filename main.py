import numpy as np

from feutureAnalysis import *
from DimensionalityReduction import *
from GaussianFit import *
from BayesianClassifiers import *
from modelEvaluation import *
from LogisticRegression import *
from SVM import *
from GMM import *
from Calibration import *

if __name__ == '__main__':
    D,L=load("trainData.txt")
    DFalse = D[:, L == 0]
    DTrue = D[:, L == 1]

    # Feature analysis
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=6)
    D0_mean = DFalse.mean(axis=1)
    D1_mean = DTrue.mean(axis=1)
    D0_var = DFalse.var(axis=1)
    D1_var = DTrue.var(axis=1)
    #print(D0_mean,D1_mean)
    #print(D0_var,D1_var)
    #plot_hist(D,L)
    #plot_scatter(D,L)

    # Dimensionality Reduction
    # for now we applying PCA and LDA to hole dataset
    # P=PCA(D,6)
    # U = LDA(D, L, [0, 1], 1)
    # PCA_plots(D,P,L,6)
    # LDA_plot(D,U,L)

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    # Now applying LDA as classifier
    P=PCA(DTR,6)
    # U = LDA(D,L, [0, 1], 1)
    # LDA_classifier(DTR,LTR,DVAL,LVAL)

    # PCA+LDA classifier
    DTR_PCA=np.dot(P.T,DTR)
    DVAL_PCA=np.dot(P.T,DVAL)
    # for i in range(6):
    #     LDA_classifier(DTR_PCA[0:i+1,:],LTR,DVAL_PCA[0:i+1,:],LVAL)

    # Gaussian Fit
    #gaussian_fit_plots(D,L)

    #Bayesian classfiers
    # print("Bayesian classfiiers")
    # MVG_classfier(DTR,LTR,DVAL,LVAL)
    # tied_classfier(DTR,LTR,DVAL,LVAL)
    # naive_classfier(DTR,LTR,DVAL,LVAL)

    # Correlation matrixes
    CorrTrue=correlationMatrixes(DTrue)
    CorrFalse=correlationMatrixes(DFalse)

    # bayesian classifiers discarding last 2 features
    # print("Bayesian classfiiers discarding last 2 featueres")
    # MVG_classfier(DTR[0:4,:],LTR,DVAL[0:4,:],LVAL)
    # tied_classfier(DTR[0:4,:],LTR,DVAL[0:4,:],LVAL)
    # naive_classfier(DTR[0:4,:],LTR,DVAL[0:4,:],LVAL)

    # bayesian classifiers 1-2 feature
    # print("Bayesian classfiiers just 1-2 feature")
    # MVG_classfier(DTR[0:2,:],LTR,DVAL[0:2,:],LVAL)
    # tied_classfier(DTR[0:2,:],LTR,DVAL[0:2,:],LVAL)
    #
    # bayesian classifiers 3-4 feature
    # print("Bayesian classfiiers just 3-4 feature")
    # MVG_classfier(DTR[2:4,:],LTR,DVAL[2:4,:],LVAL)
    # tied_classfier(DTR[2:4,:],LTR,DVAL[2:4,:],LVAL)

    # PCA + Bayesian classifiers
    # for i in range(6):
    #      print("PCA %d"%(i+1))
    #      MVG_classfier(DTR_PCA[0:i+1,:],LTR,DVAL_PCA[0:i+1,:],LVAL)
    #      tied_classfier(DTR_PCA[0:i + 1, :], LTR, DVAL_PCA[0:i + 1, :], LVAL)
    #      naive_classfier(DTR_PCA[0:i + 1, :], LTR, DVAL_PCA[0:i + 1, :], LVAL)

    # Model Evaluation
    # effective prior computing
    # p = [0.5, 0.9, 0.1, 0.5, 0.5]
    # Cfn = [1, 1, 1, 1, 9]
    # Cfp = [1, 1, 1, 9, 1]
    # for i in range(5):
    #     print("[%f  %d   %d] = %f"%(p[i],Cfn[i],Cfp[i],effective_prior(p[i],Cfn[i],Cfp[i])))

    # bayesian model evaluation
    # p = [0.5, 0.9, 0.1]
    # Cfn = [1, 1, 1]
    # Cfp = [1, 1, 1]
    # for i in range(3):
    #     print(p[i],Cfn[i],Cfp[i])
    #     print("MVG")
    #     llr=compute_MVG_llr(DTR,LTR,DVAL)
    #     modelEvaluation(llr,llr,LVAL,p[i],Cfn[i],Cfp[i])
    #     print("tied")
    #     llr=compute_tied_llr(DTR,LTR,DVAL)
    #     modelEvaluation(llr,llr,LVAL,p[i],Cfn[i],Cfp[i])
    #     print("naive")
    #     llr=compute_naive_llr(DTR,LTR,DVAL)
    #     modelEvaluation(llr,llr,LVAL,p[i],Cfn[i],Cfp[i])

    # bayes + PCA model evaliuation
    # for i in range(3):
    #     print(p[i], Cfn[i], Cfp[i])
    #     for j in range(6):
    #         print("PCA dimensions %d" % (j+1))
    #         print("MVG")
    #         llr=compute_MVG_llr(DTR_PCA[0:j+1,:],LTR,DVAL_PCA[0:j+1,:])
    #         modelEvaluation(llr,llr,LVAL,p[i],Cfn[i],Cfp[i])
    #         print("tied")
    #         llr=compute_tied_llr(DTR_PCA[0:j+1,:],LTR,DVAL_PCA[0:j+1,:])
    #         modelEvaluation(llr,llr,LVAL,p[i],Cfn[i],Cfp[i])
    #         print("naive")
    #         llr=compute_naive_llr(DTR_PCA[0:j+1,:],LTR,DVAL_PCA[0:j+1,:])
    #         modelEvaluation(llr,llr,LVAL,p[i],Cfn[i],Cfp[i])

    #Bayes error plots
    # llr = compute_MVG_llr(DTR_PCA, LTR, DVAL_PCA)
    # bayesErrorPlot(-4,4,20,llr,llr,LVAL,Cfn[2],Cfp[2],"MVG")
    # llr = compute_tied_llr(DTR_PCA, LTR, DVAL_PCA)
    # bayesErrorPlot(-4,4,20,llr,llr,LVAL,Cfn[2],Cfp[2],"Tied")
    # llr = compute_naive_llr(DTR_PCA, LTR, DVAL_PCA)
    # bayesErrorPlot(-4,4,20,llr,llr,LVAL,Cfn[2],Cfp[2],"Naive")

    p, Cfn, Cfp= 0.1, 1, 1
    thresh = binaryCostPredictor(p, Cfn, Cfp)

    # Loigistic regression
    # regularization_plot(DTR,LTR,DVAL,LVAL,p,Cfn,Cfp)
    #regularization_plot(DTR[:,::50], LTR[::50], DVAL, LVAL, p, Cfn, Cfp)
    #regularization_plot(DTR-vcol(DTR.mean(1)), LTR, DVAL-vcol(DTR.mean(1)), LVAL, p, Cfn, Cfp)

    #regression_minDCF(DTR, LTR, DVAL, LVAL, p, Cfn, Cfp)

    #SVM
    #SVM_plot(DTR,LTR,DVAL,LVAL, p, Cfn, Cfp)
    #RBF_grid_search(DTR, LTR, DVAL, LVAL, p, Cfn, Cfp)
    #feature_5_6_combined_space(D, L)


    # GMM
    # GMM_cls(DTR,LTR,DVAL,LVAL,'diagonal',p,Cfn,Cfp)

    # selected models
    #qudratic logistic regression
    # sllr = qaudraticlogreg_sllr(DTR, LTR, 0.01, DVAL)
    # np.save("quadratic_logreg_sllr", sllr)
    # bayesErrorPlot(-4,4,31,sllr,sllr,LVAL,Cfn,Cfp,"quadratic logistic regression ")
    #
    # #SVM_RBF
    # scores= RBF_SVM(DTR,LTR,DVAL,100,1.0,0.1)
    # np.save("SVM_RBF_scores", scores)
    # bayesErrorPlot(-4, 4, 31, scores,scores, LVAL, Cfn, Cfp, "SVM_RBF")
    #
    # #GMM
    # sllr = GMM_cls_iteration(DTR,LTR,DVAL,8,"diagonal")
    # np.save("GMM_diagoanl_sllr", sllr)
    # bayesErrorPlot(-4, 4, 31, sllr, sllr, LVAL, Cfn, Cfp, "GMM diagonal")

    quadratic_logreg_sllr = np.load("quadratic_logreg_sllr.npy")
    SVM_RBF_scores = np.load("SVM_RBF_scores.npy")
    GMM_diagonal_sllr = np.load("GMM_diagoanl_sllr.npy")

    # Kfold_calibration(quadratic_logreg_sllr,LVAL,p,Cfn,Cfp)
    # Kfold_calibration(SVM_RBF_scores,LVAL,p,Cfn,Cfp)
    # Kfold_calibration(GMM_diagonal_sllr,LVAL,p,Cfn,Cfp)
    #
    # prior_cal_logreg =  0.4
    # prior_cal_SVM_RBF = 0.9
    # prior_cal_diagGMM = 0.8
    #
    # quad_logreg_cal_sllr = get_calibrated_scores(quadratic_logreg_sllr,prior_cal_logreg,LVAL)
    # SVM_RBF_cal_scores= get_calibrated_scores(SVM_RBF_scores,prior_cal_SVM_RBF, LVAL)
    # GMM_diagonal_cal_sllr= get_calibrated_scores(GMM_diagonal_sllr,prior_cal_diagGMM, LVAL)
    #
    # calibrated_models_bayes_error_plot(quad_logreg_cal_sllr,SVM_RBF_cal_scores,GMM_diagonal_cal_sllr, LVAL, p, Cfn, Cfp)

    # fusion(quadratic_logreg_sllr,SVM_RBF_scores, GMM_diagonal_sllr,LVAL,p,Cfn,Cfp)
    # prior_fusion=0.8
    fusion_scores= numpy.vstack([quadratic_logreg_sllr,SVM_RBF_scores,GMM_diagonal_sllr])
    # fusion_cal_sllr = get_fusion_scores(fusion_scores,prior_fusion,LVAL)
    # predictions = np.where(fusion_cal_sllr > thresh, 1, 0)
    # print(actualDCF(p, Cfn, Cfp, predictions,LVAL), minDCF(fusion_cal_sllr,fusion_cal_sllr,LVAL, p, Cfn, Cfp))


    # Evalutaion

    DEVAL, LEVAL= load("evalData.txt")

    # sllr_logreg = qaudraticlogreg_sllr(DTR, LTR, 0.01, DEVAL)
    # np.save("logreg_eval", sllr_logreg)
    # sllr_SVM= RBF_SVM(DTR,LTR,DEVAL,100,1.0,0.1)
    # np.save("SVM_eval", sllr_SVM)
    # sllr_GMM = GMM_cls_iteration(DTR,LTR,DEVAL,8,"diagonal")
    # np.save("GMM_eval", sllr_GMM)
    #
    # sllr_logreg= np.load("logreg_eval.npy")
    # sllr_SVM = np.load("SVM_eval.npy")
    # sllr_GMM = np.load("GMM_eval.npy")
    #
    # x,_=weightedLogReg(fusion_scores,LVAL,0,0.8)
    # w, b = x[0:-1], x[-1]
    # sllr_fusion = numpy.dot(w.T, np.vstack([sllr_logreg,sllr_SVM,sllr_GMM])) + b - numpy.log(0.8/(1 - 0.8))
    # predictions_fusion = np.where(sllr_fusion > thresh, 1, 0)

    # error = np.sum(predictions_fusion != LEVAL)
    # actual_dcf= actualDCF(p,Cfn,Cfp,predictions_fusion,LEVAL)
    # min_dcf = minDCF(sllr_fusion,sllr_fusion,LEVAL,p,Cfn,Cfp)
    # print(error,actual_dcf,min_dcf)
    # bayesErrorPlot(-4, 4, 31,  sllr_fusion,  sllr_fusion, LEVAL, Cfn, Cfp, "Evaluation set fusion")

    # x,_=weightedLogReg(vrow(quadratic_logreg_sllr),LVAL,0,0.4)
    # w, b = x[0:-1], x[-1]
    # sllr_logreg = numpy.dot(w.T, vrow(sllr_logreg)) + b - numpy.log(0.4/(1 - 0.4))
    # predictions_logreg = np.where(sllr_logreg > thresh, 1, 0)
    #
    #
    # x,_=weightedLogReg(vrow(SVM_RBF_scores),LVAL,0,0.9)
    # w, b = x[0:-1], x[-1]
    # sllr_SVM = numpy.dot(w.T, vrow(sllr_SVM)) + b - numpy.log(0.9/(1 - 0.9))
    # predictions_SVM = np.where(sllr_SVM > thresh, 1, 0)
    #
    # x,_=weightedLogReg(vrow(GMM_diagonal_sllr),LVAL,0,0.8)
    # w, b = x[0:-1], x[-1]
    # sllr_GMM = numpy.dot(w.T, vrow(sllr_GMM)) + b - numpy.log(0.8/(1 - 0.8))
    # predictions_GMM = np.where(sllr_GMM > thresh, 1, 0)
    #
    # print(actualDCF(p,Cfn,Cfp,predictions_logreg,LEVAL),minDCF(sllr_logreg,sllr_logreg,LEVAL,p,Cfn,Cfp))
    # print(actualDCF(p, Cfn, Cfp,predictions_SVM, LEVAL), minDCF(sllr_SVM, sllr_SVM, LEVAL, p, Cfn, Cfp))
    # print(actualDCF(p, Cfn, Cfp, predictions_GMM,LEVAL), minDCF(sllr_GMM,sllr_GMM, LEVAL, p, Cfn, Cfp))
    #
    # x,y,n= -4,4,31
    # effPriorLogOdds = np.linspace(x, y, n)
    # pi=1/(1 + np.exp(-effPriorLogOdds))
    # logreg_dcf=[]
    # logreg_mindcf=[]
    # SVM_dcf=[]
    # SVM_mindcf=[]
    # GMM_dcf=[]
    # GMM_mindcf=[]
    # fusion_dcf=[]
    # fusion_mindcf=[]
    # for i in pi:
    #     thresh= binaryCostPredictor(i,Cfn,Cfp)
    #
    #     logreg_p = np.where(sllr_logreg > thresh, 1, 0)
    #     logreg_dcf.append(actualDCF(i,Cfn,Cfp,logreg_p,LEVAL))
    #     logreg_mindcf.append(minDCF(sllr_logreg,sllr_logreg,LEVAL,i,Cfn,Cfp))
    #
    #     SVM_p = np.where(sllr_SVM > thresh, 1, 0)
    #     SVM_dcf.append(actualDCF(i,Cfn,Cfp,SVM_p,LEVAL))
    #     SVM_mindcf.append(minDCF(sllr_SVM,sllr_SVM,LEVAL,i,Cfn,Cfp))
    #
    #     GMM_p = np.where(sllr_GMM > thresh, 1, 0)
    #     GMM_dcf.append(actualDCF(i,Cfn,Cfp,GMM_p,LEVAL))
    #     GMM_mindcf.append(minDCF(sllr_GMM,sllr_GMM,LEVAL,i,Cfn,Cfp))
    #
    #     fusion_p = np.where(sllr_fusion > thresh, 1, 0)
    #     fusion_dcf.append(actualDCF(i,Cfn,Cfp,fusion_p,LEVAL))
    #     fusion_mindcf.append(minDCF(sllr_fusion,sllr_fusion,LEVAL,i,Cfn,Cfp))
    #
    # plt.plot(effPriorLogOdds, logreg_dcf, label='quadratic logreg DCF', color ='r')
    # plt.plot(effPriorLogOdds, logreg_mindcf,linestyle='--', label='quadratic logreg minDCF', color ='r')
    # plt.plot(effPriorLogOdds, SVM_dcf, label='SVM RBF DCF', color ='b')
    # plt.plot(effPriorLogOdds, SVM_mindcf,linestyle='--', label='SVM RBF minDCF', color ='b')
    # plt.plot(effPriorLogOdds, GMM_dcf, label='diagonal GMM DCF', color ='g')
    # plt.plot(effPriorLogOdds, GMM_mindcf,linestyle='--', label='diagonal GMM minDCF', color ='g')
    # plt.plot(effPriorLogOdds, fusion_dcf, label='fusion DCF', color ='#FFA500')
    # plt.plot(effPriorLogOdds, fusion_mindcf,linestyle='--', label='fusion minDCF', color ='#FFA500')
    #
    # plt.ylim([0, 1.1])
    # plt.xlim([x,y])
    # plt.legend(loc='upper left')
    # plt.title("final models Bayes error plot")
    # plt.savefig("final Bayes error plot")
    # plt.show()


    # Logistic Regression Variants
    # x=numpy.logspace(-4, 2, 13)
    # mindcf_logreg=[]
    # mindcf_weightedlogreg=[]
    # mindcf_quadraticlogreg=[]
    # for l in x:
    #     sllr1 = logreg_sllr(DTR, LTR, l, DEVAL)
    #     mindcf1=minDCF(sllr1,sllr1,LEVAL,p,Cfn,Cfp)
    #     mindcf_logreg.append(mindcf1)
    #
    #     sllr2=weightedLogReg_sllr(DTR,LTR,l,DEVAL,p)
    #     mindcf2 = minDCF(sllr2, sllr2, LEVAL, p, Cfn, Cfp)
    #     mindcf_weightedlogreg.append(mindcf2)
    #
    #     sllr3=qaudraticlogreg_sllr(DTR,LTR,l,DEVAL)
    #     mindcf3 = minDCF(sllr3, sllr3, LEVAL, p, Cfn, Cfp)
    #     mindcf_quadraticlogreg.append(mindcf3)
    #
    #     print(l,mindcf1,mindcf2,mindcf3)
    #
    #
    # plt.xscale('log', base = 10)
    # plt.plot(x, mindcf_logreg, label='minDCF logreg', color='b')
    # plt.plot(x, mindcf_weightedlogreg, label='minDCF weighted logreg', color='r')
    # plt.plot(x, mindcf_quadraticlogreg, label='minDCF quadratic logreg', color='g')
    # plt.legend()
    # plt.title("Logistic Regression variants evaluation set")
    # plt.savefig("logreg variants evaluation set")
    # plt.show()