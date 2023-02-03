from tabnanny import verbose
import cvxpy as cp 
import numpy as np
import numpy as np
import time
import numpy.linalg as la
import matplotlib.pyplot as plt
from torch._C import ThroughputBenchmark
from sklearn.svm import SVC
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import math
import random

import warnings
warnings.filterwarnings('ignore')

seed=42
random.seed(a=seed)
np.random.seed(seed=seed)

def relu(x):
    return np.maximum(0,x)
def drelu(x):
    return x>=0


def convex_lasso_threshold(X, Y, X_test, Y_test, beta, n_train, n_test,big_num):
    # beta=1e-5
    P=2000
    P = big_num
    n = n_train
    d = X.shape[1]

    dmat=np.empty((n,0))

    ## Finite approximation of all possible sign patterns
    Up=np.zeros((d,P))
    for i in range(P):
        u=np.random.randn(d,1)
        Up[:,i]=u.reshape(-1,)
        dmat=np.append(dmat,drelu(X@u),axis=1)

    dmat, indices=np.unique(dmat,axis=1, return_index=True)
    U=Up[:,indices]

    m=dmat.shape[1]
    uopt=cp.Variable((m,))
    print(f"number of neuron for lasso: {m}")
    yopt=cp.Parameter((n,1))
    yopt=dmat*uopt
    # cost=cp.sum_squares(Y-yopt)/(2*n)+beta*cp.norm(uopt,1)
    cost=cp.sum_squares(Y-yopt)/2+beta*cp.norm(uopt,1)

    constraints=[]
    prob=cp.Problem(cp.Minimize(cost),constraints)
    start_time = time.time()
    prob.solve(solver=cp.MOSEK,warm_start=True,verbose=False)
    cvx_opt_thr=prob.value
    end_time = time.time()
    if prob.status != "optimal":
        print("Threshold convex: Status convex: ",prob.status)
        raise "Convex problem status error"
    print("2-layer threshold convex program objective value: ",cvx_opt_thr)

    uoptv=uopt.value

    yest_cvx_thr=drelu(X@U)@uoptv
    yest_cvx_thr[yest_cvx_thr<0] = -1
    yest_cvx_thr[yest_cvx_thr>=0] = 1
    train_acc = np.sum(yest_cvx_thr == Y)/n

    yest_cvx_thr=drelu(X_test@U)@uoptv
    yest_cvx_thr[yest_cvx_thr<0] = -1
    yest_cvx_thr[yest_cvx_thr>=0] = 1
    test_acc = np.sum(yest_cvx_thr == Y_test)/n_test

    lasso_time = end_time - start_time

    return test_acc,train_acc, cvx_opt_thr/n ,lasso_time



