import cvxpy as cp 
import numpy as np
import numpy as np
import time
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def cripped_relu(x):
    x = np.maximum(x,0)
    x = np.minimum(x,1)
    return x 

def SVM(Xmat,dvct):
    dvec = dvct.copy()
    dvec[dvec==1] = 1
    dvec[dvec==0] = -1
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(Xmat, dvec)
    vct = np.reshape(svclassifier.coef_.T,newshape=(svclassifier.coef_.T.shape[0],))
    est = Xmat.dot(vct)
    est = est + svclassifier.intercept_
    est[est>=0] = 1
    est[est<0] = 0
    # assert (est==dvct).all(), 'SVM does not working'
    dest = svclassifier.predict(Xmat)
    return vct,svclassifier.intercept_

def relu(x):
    return np.maximum(0,x)

def drelu(x):
    return x>=0



class convex_nn_3layer:
    def __init__(self, X,Y,Xtest,Ytest,beta,nb_neuron,nb_epoch,n_train,n_test,big_number,rec_style,rep_matrix):
        self.n_test = n_test
        self.n_train = n_train
        self.X = X
        self.Y = Y
        self.Xtest = Xtest
        self.Ytest = Ytest 
        self.beta = beta
        self.nb_neuron = nb_neuron
        self.nb_epoch = nb_epoch
        self.alpha_matrix = None
        self.alpha_matrix2 = None
        self.delta_ans = None
        self.delta_np = None
        self.vmatrix = None
        self.vmatrix2 = None
        self.convex_classification_acc = None
        self.convex_test_l2_error = None
        self.convex_train_l2_error = None
        self.nn_l2_loss = None
        self.convex_classification_acc_relu, self.convex_classification_acc_relu_training = None, None
        self.convex_classification_acc_unitstep, self.convex_classification_acc_unitstep_tranining = None, None
        self.convex_classification_acc_unitstep_05, self.convex_classification_acc_unitstep_05_tranining = None, None
        self.delta = None
        convex_classification_acc_unitstep_05 = None
        convex_classification_acc_unitstep_training_05 = None
        self.optimal_value = None
        self.loss_value_after_reconstruction = None
        self.loss_value_after_reconstruction2 = None
        self.U = None
        self.big_number = big_number
        self.weirdo = None
        self.rec_style = rec_style
        self.intercept = None
        self.rep_matrix = rep_matrix
        self.time = None
        
    ##CONVEX PROBLEM SOLVER
    def convex_nn_unitstep_3layer(self):
        # CVXPY variables
        delta = cp.Variable((self.n_train,))
        # regularizations
        regw = cp.norm_inf(cp.pos(delta)) + cp.norm_inf(cp.neg(delta))
        # constructs the optimization problem
        betaval = cp.Parameter(nonneg=True)
        betaval.value = self.beta
        cost = cp.sum_squares(delta - self.Y) / 2 \
               + betaval * regw
        constraints = []
        params = {"MSK_IPAR_NUM_THREADS": 8}
        # solve the problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False, mosek_params=params)
        cvx_opt = prob.value
        # print("Opt_value = {}".format(cvx_opt))
        self.optimal_value = cvx_opt
        # print("Status: ", prob.status)
        assert prob.status=="optimal", 'Convex Program is not optimal'
#         print("delta value {}".format(delta.value))
        self.delta = delta
        return

    def convex_nn_construction_3layer(self):
        ## neural network construction From delta
        nb_pos = np.sum(np.array(self.Y) > 0, axis=0)
        nb_neg = np.sum(np.array(self.Y) < 0, axis=0)
        delta_np = np.array(self.delta.value)
        delta_ans = delta_np.copy()
        delta_ans[delta_ans>0] = 1
        delta_ans[delta_ans<=0] = -1
        self.delta_ans = delta_ans
        d = np.maximum(delta_np,0)
        d_norm = np.max(d)
        d_p = np.maximum(-delta_np,0)
        d_p_norm = np.max(d_p)

        d /= d_norm
        d_p /= d_p_norm

        inds_d = np.argsort(d)
        inds_d_p = np.argsort(d_p)

        d_positives = np.empty((self.n_train, 1))
        d_negatives = np.empty((self.n_train, 1))
        gamma_positives = []
        gamma_negatives = []
        pos_list = []
        neg_list = []
        for i in range(nb_neg):
            pos_list.append(inds_d[i])
        for i in range(nb_neg, self.n_train):
            gamma = d[inds_d[i]] - d[inds_d[i - 1]]
            gamma_positives.append(gamma)
            d_vect = np.ones((self.n_train, 1))
            for k in pos_list:
                d_vect[k] = 0
            d_positives = np.concatenate((d_positives, d_vect), axis=1)
            pos_list.append(inds_d[i])

        for i in range(nb_pos):
            neg_list.append(inds_d_p[i])
        for i in range(nb_pos, self.n_train):
            gamma = d_p[inds_d_p[i]] - d_p[inds_d_p[i - 1]]
            gamma_negatives.append(gamma)
            d_vect = np.ones((self.n_train, 1))
            for k in neg_list:
                d_vect[k] = 0
            d_negatives = np.concatenate((d_negatives, d_vect), axis=1)
            neg_list.append(inds_d_p[i])

        d_positives = np.delete(d_positives, 0, 1)
        d_negatives = np.delete(d_negatives, 0, 1)

        v_list = []
        alpha_list = []
        s_list = []

        if self.rep_matrix == "big_matrix":
            U = np.random.normal(loc=0, scale=1, size=(self.X.shape[1],self.big_number))
            print("fsg")
            print(U)
            self.U = U.copy()
            X_hat = self.X@U
            print(X_hat)
            X_hat[X_hat>0] = 1
            X_hat[X_hat<=0] = 0

        if self.rep_matrix == "unique":
            P=int(self.big_number)
            dim = self.X.shape[1]
            dmat=np.empty((self.n_train,0))
            U = np.empty((dim,0))
            ## Finite approximation of all possible sign patterns
            for i in range(P):
                u=np.random.randn(dim,1)
                U = np.append(U,u,axis=1)
                dmat=np.append(dmat,drelu(np.dot(self.X,u)),axis=1)
            for i in range(P):
                u=np.random.uniform(low=-1,high=1,size=(dim,1))
                U = np.append(U,u,axis=1)
                dmat=np.append(dmat,drelu(np.dot(self.X,u)),axis=1)
            dmat,index=(np.unique(dmat,axis=1,return_index=True))
            U = U[:,index]
            X_hat = self.X.dot(U)
            X_hat[X_hat<0] = 0
            X_hat[X_hat>0] = 1
            # print(f"X_hat rank : {la.matrix_rank(X_hat)}")
            self.U = U


        if self.rep_matrix == "smart_choice":
            a = np.random.uniform(low=-3, high=3, size=(self.X.shape[1]-1,))
            z = self.X[:,:-1].dot(a)
            indx = np.argsort(z)
            b = np.zeros((self.n_train,))
            b[0] = z[indx[0]] - (z[indx[1]]- z[indx[0]])/2
            for i in range(1,self.n_train):
                b[i] = (z[indx[i-1]] + z[indx[i]]) / 2
            z_sorted = np.sort(z).copy()

            U = np.zeros((self.X.shape[1],self.n_train))
            for i in range(self.n_train):
                U[:self.X.shape[1]-1,i] = a
                U[self.X.shape[1]-1,i] = -b[i]

            X_hat = self.X.dot(U)
            X_hat[X_hat<0] = 0
            X_hat[X_hat>0] = 1
            self.U = U.copy()



        X_hat_inv = la.pinv(X_hat)
        intercept_list = []
        
        for i in range(nb_pos):
            if self.rec_style=="standard":
                v_list.append(X_hat_inv.dot(d_positives[:, i])) # standard reconstruction
            if self.rec_style=="SVM":
                vct , intercept = SVM(X_hat,d_positives[:, i])
                v_list.append(vct)                             # SVM reconstruction
                intercept_list.append(intercept)
            alpha_list.append(d_norm * gamma_positives[i])  
            s_list.append(+1)
        for i in range(nb_neg):
            if self.rec_style=="standard":
                v_list.append(X_hat_inv.dot(d_negatives[:, i])) # standard reconstruction
            if self.rec_style=="SVM":
                vct, intercept = SVM(X_hat,d_negatives[:, i])
                v_list.append(vct)                              # SVM reconstruction
                intercept_list.append(intercept)
            alpha_list.append(d_p_norm * gamma_negatives[i])
            s_list.append(-1)
        convex_output = np.zeros((self.n_train, ))
        for i in range(len(v_list)):
            convex_output += s_list[i] * alpha_list[i] * np.maximum(X_hat.dot(v_list[i]), 0)

        v_matrix = np.zeros((X_hat.shape[1],len(v_list)))
        for i in range(len(v_list)):
            v_matrix[:,i] = v_list[i]
        alpha_matrix = np.array(alpha_list)
        alpha_matrix = np.reshape(alpha_matrix,newshape=(len(alpha_list),))
        s_matrix = np.reshape(np.array(s_list),newshape=(len(alpha_list),))
        alpha_matrix_ = np.multiply(alpha_matrix,s_matrix)
        self.vmatrix = v_matrix
        self.alpha_matrix = alpha_matrix_
        self.intercept = np.array(intercept_list)
        return



    def convex_error_finder_3layer(self):
        #####TEST CASE ######  we find test performance of the model
        if self.rec_style == "SVM":
            self.vmatrix = np.concatenate((self.vmatrix,self.intercept.T),axis=0)
        th = 1e-8

        if self.rec_style == "standard":
            ans1 = np.where(np.where(self.Xtest.dot(self.U)>th,1,0).dot(self.vmatrix)>th,1,0).dot(self.alpha_matrix)
        if self.rec_style == "SVM":
            ans1 = np.where(self.Xtest.dot(self.U)>th,1,0)
            ans1 = np.concatenate((ans1,np.ones((ans1.shape[0],1))),axis=1)
            ans1 = np.where(ans1.dot(self.vmatrix)>th,1,0).dot(self.alpha_matrix)
        ans1[ans1<0] = -1
        ans1[ans1>=0] = 1
        self.convex_classification_acc_unitstep = int(np.sum(ans1 == self.Ytest)) / self.Ytest.shape[0]

        if self.rec_style == "standard":
            ans1 = np.where(np.where(self.Xtest.dot(self.U)>0,1,0).dot(self.vmatrix)>0.5,1,0).dot(self.alpha_matrix)
        if self.rec_style == "SVM":
            ans1 = np.where(self.Xtest.dot(self.U)>0,1,0)
            ans1 = np.concatenate((ans1,np.ones((ans1.shape[0],1))),axis=1)
            ans1 = np.where(ans1.dot(self.vmatrix)>0.5,1,0).dot(self.alpha_matrix)
        ans1[ans1<0] = -1
        ans1[ans1>=0] = 1
        self.convex_classification_acc_unitstep_05 = int(np.sum(ans1 == self.Ytest)) / self.Ytest.shape[0]

        if self.rec_style == "standard":
            ans1 = cripped_relu(np.where(self.Xtest.dot(self.U)>th,1,0).dot(self.vmatrix)).dot(self.alpha_matrix)
        if self.rec_style == "SVM":
            ans1 = np.where(self.Xtest.dot(self.U)>0,1,0)
            ans1 = np.concatenate((ans1,np.ones((ans1.shape[0],1))),axis=1)
            ans1 = cripped_relu(ans1.dot(self.vmatrix)).dot(self.alpha_matrix)
        ans1[ans1<0] = -1
        ans1[ans1>=0] = 1
        self.convex_classification_acc_relu = int(np.sum(ans1 == self.Ytest)) / self.Ytest.shape[0]


        ######TRAIN CASE#########
        th = 1e-8

        if self.rec_style == "standard":
            ans1 = np.where(np.where(self.X.dot(self.U)>th,1,0).dot(self.vmatrix)>th,1,0).dot(self.alpha_matrix)
        if self.rec_style == "SVM":
            ans1 = np.where(self.X.dot(self.U)>th,1,0)
            ans1 = np.concatenate((ans1,np.ones((ans1.shape[0],1))),axis=1)
            ans1 = np.where(ans1.dot(self.vmatrix)>th,1,0).dot(self.alpha_matrix)
        ans1[ans1<0] = -1
        ans1[ans1>=0] = 1
        self.convex_classification_acc_unitstep_training = int(np.sum(ans1 == self.Y)) / self.Y.shape[0]

        if self.rec_style == "standard":
            ans1 = np.where(np.where(self.X.dot(self.U)>0,1,0).dot(self.vmatrix)>0.5,1,0).dot(self.alpha_matrix)
        if self.rec_style == "SVM":
            ans1 = np.where(self.X.dot(self.U)>0,1,0)
            ans1 = np.concatenate((ans1,np.ones((ans1.shape[0],1))),axis=1)
            ans1 = np.where(ans1.dot(self.vmatrix)>0.5,1,0).dot(self.alpha_matrix)
        ans1[ans1<0] = -1
        ans1[ans1>=0] = 1
        self.convex_classification_acc_unitstep_05_training = int(np.sum(ans1 == self.Y)) / self.Y.shape[0]

        if self.rec_style == "standard":
            ans1 = cripped_relu(np.where(self.X.dot(self.U)>th,1,0).dot(self.vmatrix)).dot(self.alpha_matrix)
        if self.rec_style == "SVM":
            ans1 = np.where(self.X.dot(self.U)>0,1,0)
            ans1 = np.concatenate((ans1,np.ones((ans1.shape[0],1))),axis=1)
            ans1 = cripped_relu(ans1.dot(self.vmatrix)).dot(self.alpha_matrix)
        ans1[ans1<0] = -1
        ans1[ans1>=0] = 1
        self.convex_classification_acc_relu_training = int(np.sum(ans1 == self.Y)) / self.Y.shape[0]


    def get_acc_loss_3layer(self):
        start_time = time.time()
        self.convex_nn_unitstep_3layer()
        self.convex_nn_construction_3layer()
        end_time = time.time()
        self.time = end_time - start_time
        self.convex_error_finder_3layer()
        return self.convex_classification_acc_relu , self.convex_classification_acc_unitstep \
                ,self.convex_classification_acc_unitstep_05\
                ,self.convex_classification_acc_relu_training , self.convex_classification_acc_unitstep_training \
                ,self.convex_classification_acc_unitstep_05_training\
                ,self.optimal_value/self.n_train \
                ,self.time











