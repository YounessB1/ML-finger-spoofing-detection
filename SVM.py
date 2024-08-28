import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
from modelEvaluation import *
def compute_H_matrix(train_data, train_labels, K):
    # Extend the training data matrix
    extended_train_data = np.vstack((train_data, K * np.ones(train_data.shape[1])))
    Z = np.where(train_labels == 1, 1, -1)

    # Compute the H matrix
    H = np.dot(extended_train_data.T, extended_train_data)
    H = np.outer(Z, Z) * H

    return H, Z, extended_train_data


def linear_SVM(train_data, train_labels, test_data, C, K):
    H, Z, extended_train_data = compute_H_matrix(train_data, train_labels, K)

    # Define the dual objective function
    def dual_objective(alpha_vector):
        Ha_vector = np.dot(H, alpha_vector.reshape(-1, 1))
        aHa_scalar = np.dot(alpha_vector, Ha_vector)
        a1_scalar = alpha_vector.sum()
        result = -0.5 * aHa_scalar + a1_scalar, (-Ha_vector + 1).flatten()
        l = - result[0]
        g = - result[1]
        return l, g


    # Optimize the dual objective function
    optimal_alpha, _, _ = fmin_l_bfgs_b(dual_objective,np.zeros(train_data.shape[1]),bounds=([(0, C)] * train_data.shape[1]),factr=1.0,maxiter=100000,maxfun=100000,)

    # Compute wStar
    optimal_weight = np.dot(extended_train_data, (optimal_alpha * Z))

    # Extend the test data matrix
    extended_test_data  = np.vstack((test_data, np.array([K for _ in range(test_data.shape[1])])))

    computed_scores  = np.dot(optimal_weight .T, extended_test_data )
    computed_scores=computed_scores.ravel()

    return computed_scores


def compute_H_poly(train_data, train_labels, K, d, c):

    # Extend the training data matrix
    extended_train_data = np.vstack((train_data, K * np.ones(train_data.shape[1])))
    Z = np.where(train_labels == 1, 1, -1)

    # Compute the H matrix using kernel function
    H = ((np.dot(extended_train_data.T, extended_train_data) + c) ** d) + K * K
    H = np.outer(Z, Z) * H

    return H, Z, extended_train_data

def poly_SVM(train_data, train_labels, test_data, C, K, d, c):
    H, Z, extended_train_data = compute_H_poly(train_data, train_labels, K, d, c)

    # Define the dual objective function
    def dual_func(alpha_values):
        Ha_values = np.dot(H, alpha_values.reshape(-1, 1))
        aHa_value = np.dot(alpha_values, Ha_values)
        a1_value = alpha_values.sum()
        result = -0.5 * aHa_value + a1_value, (-Ha_values + 1).flatten()
        l = - result[0]
        g = - result[1]
        return l, g

    # Optimize the dual objective function
    optimal_alpha, _, _ = fmin_l_bfgs_b(dual_func,np.zeros(train_data.shape[1]),bounds=([(0, C)] * train_data.shape[1]) ,factr=1.0,maxiter=100000,maxfun=100000,)


    KERN = ((np.dot(train_data.T, test_data) + c) ** d) + K * K
    Z_row = Z.reshape((1, Z.size))
    computed_scores = np.sum(np.dot((optimal_alpha * Z_row), KERN), axis=0)
    computed_scores = computed_scores.ravel()

    return computed_scores


def compute_H_rbf(train_data, train_labels, K, gamma):
    # Compute the label vector
    Z = np.where(train_labels == 1, 1, -1)

    # Compute the H matrix using RBF kernel function
    H = np.zeros((train_data.shape[1], train_data.shape[1]))
    for i in range(train_data.shape[1]):
        for j in range(train_data.shape[1]):
            H[i, j] = np.exp(-gamma * (np.linalg.norm(train_data[:, i] - train_data[:, j]) ** 2)) + K * K
    H = np.outer(Z, Z) * H

    return H, Z

def compute_kernel(train_data, test_data, gamma):

    kernel_matrix = np.zeros((train_data.shape[1], test_data.shape[1]))
    for i in range(train_data.shape[1]):
        for j in range(test_data.shape[1]):
            kernel_matrix[i, j] = np.exp(-gamma * (np.linalg.norm(train_data[:, i] - test_data[:, j]) ** 2))
    return kernel_matrix


def RBF_SVM(train_data, train_labels, test_data, C, K, gamma):
    H, Z = compute_H_rbf(train_data, train_labels, K, gamma)

    # Define the dual objective function
    def dual_func(alpha_values):
        Ha_values = np.dot(H, alpha_values.reshape(-1, 1))
        aHa_value = np.dot(alpha_values, Ha_values)
        a1_value = alpha_values.sum()
        result = -0.5 * aHa_value + a1_value, (-Ha_values + 1).flatten()
        l = - result[0]
        g = - result[1]
        return l, g

    # Optimize the dual objective function
    optimal_alpha, _, _ = fmin_l_bfgs_b(dual_func,np.zeros(train_data.shape[1]),bounds=([(0, C)] * train_data.shape[1]),factr=1.0,maxiter=100000,maxfun=100000,)

    # Compute the kernel matrix outside the function
    kernel_matrix = compute_kernel(train_data, test_data, gamma) + K * K

    Z_row = Z.reshape((1, Z.size))
    # Compute the scores
    computed_scores = np.sum(np.dot(optimal_alpha * Z_row, kernel_matrix), axis=0)
    return computed_scores.ravel()

def SVM_plot(DTR,LTR,DVAL,LVAL,p,Cfn,Cfp):
    C=np.logspace(-5, 0, 11)
    thresh = binaryCostPredictor(p, Cfn, Cfp)
    K=0
    c=1
    d=4
    dcf=[]
    mindcf=[]
    for i in C:
        #scores = linear_SVM(DTR,LTR,DVAL,i,K)
        scores = poly_SVM(DTR,LTR,DVAL,i,K,d,c)
        predictions=np.where(scores > thresh, 1, 0)
        dcf.append(actualDCF(p,Cfn,Cfp,predictions,LVAL))
        mindcf.append(minDCF(scores,scores,LVAL,p,Cfn,Cfp))

    print(min(mindcf))
    plt.xscale('log', base = 10)
    plt.plot(C, dcf, label='DCF', color='r')
    plt.plot(C, mindcf, label='minDCF', color='b')
    plt.scatter(C, dcf, color='r')
    plt.scatter(C, mindcf, color='b')
    plt.legend()
    plt.title("poly 4d SVM")
    plt.savefig("poly_d4_SVM")
    plt.show()


def RBF_grid_search(DTR,LTR,DVAL,LVAL,p,Cfn,Cfp):
    C=np.logspace(-3, 2, 11)
    thresh = binaryCostPredictor(p, Cfn, Cfp)
    K=1.0
    gamma=[1e-4,1e-3,1e-2,1e-1]
    best_minDCF=[]
    plt.xscale('log', base=10)
    for g in gamma:
        dcf = []
        mindcf = []
        for i in C:
            scores = RBF_SVM(DTR,LTR,DVAL,i,K,g)
            predictions=np.where(scores > thresh, 1, 0)
            dcf.append(actualDCF(p,Cfn,Cfp,predictions,LVAL))
            mindcf.append(minDCF(scores,scores,LVAL,p,Cfn,Cfp))

        best_minDCF.append(min(mindcf))
        plt.plot(C, dcf,label=f'DCF (gamma={g})')
        plt.plot(C, mindcf, label=f'minDCF (gamma={g})')
        plt.scatter(C, dcf)
        plt.scatter(C, mindcf)

    print(best_minDCF)
    plt.legend()
    plt.title("RBF SVM grid search")
    plt.savefig("RBF_SVM_grid_search")
    plt.show()

def feature_5_6_combined_space(D,L):
    z=D[4,:]*D[5,:]
    zFalse = z[L == 0]
    zTrue = z[L == 1]
    plt.scatter(np.where(L == 0)[0], zFalse, color='red',alpha=0.4, label='Class 0')
    plt.scatter(np.where(L == 1)[0], zTrue, color='blue',alpha=0.4, label='Class 1')
    plt.title('Feature 5 * Feature 6')
    plt.savefig('Feature5_Feature6')
    plt.legend()
    plt.show()