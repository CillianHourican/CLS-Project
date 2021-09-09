# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 18:27:24 2021

@author: Cillian
"""
import pickle 
import numpy as np
import GPy


Inputs = pickle.load(open("ISR3D_data\InputList.pkl", "rb")).to_numpy()
data = np.load('ISR3D_data\LumenData.npy')
# Convert to Areas
Outputs = data[:,-1,:].mean(axis = 1)

#--------------------  DATA PREPARATION ---------------#
Ntr = 40 # Number of training points to use 
Nts = 50 # Number of test points to use

# All data represented in data['Y'], which is the angles of the movement of the subject
perm = np.random.permutation(Ntr+Nts) # Random selection of data to form train/test set
index_training = np.sort(perm[0:Ntr])
index_test     = np.sort(perm[Ntr:Ntr+Nts])
X_tr = Inputs[index_training,:]
X_ts = Inputs[index_test,    :]

Y_tr = Outputs[index_training, None]
Y_ts = Outputs[index_test,None]

# It can help to normalize the input and/or output data.
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
scalerY = StandardScaler()
scalerX.fit(X_tr)
scalerY.fit(Y_tr)
X_tr_scaled = scalerX.transform(X_tr)
X_ts_scaled = scalerX.transform(X_ts)
Y_tr_scaled = scalerY.transform(Y_tr)

def rmse(predictions, targets):
    return np.sqrt(((predictions.flatten() - targets.flatten()) ** 2).mean())

####################################################
# Pass the predictions through inverse scaling transform to compare them in the original data space
#Y_pred    = scalerY.inverse_transform(m.predict(X_ts_scaled)[0])
#Y_pred_s  = scalerY.inverse_transform(m.predict_withSamples(X_ts_scaled, nSamples=500)[0])
#Y_pred_GP = scalerY.inverse_transform(m_GP.predict(X_ts_scaled)[0])

# Y_pred    = (m.predict(X_ts)[0])
# Y_pred_s  = (m.predict_withSamples(X_ts, nSamples=500)[0])
# Y_pred_GP = (m_GP.predict(X_ts_scaled)[0])

#print('# RMSE DGP               : ' + str(rmse(Y_pred, Y_ts)))
#print('# RMSE DGP (with samples): ' + str(rmse(Y_pred_s, Y_ts)))
#print('# RMSE GP                : ' + str(rmse(Y_pred_GP, Y_ts)))


# #==============================================================
# # # Non-stationary kernel
from scipy.optimize import differential_evolution

from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels \
#     import ConstantKernel as C, Matern
# # from sklearn.metrics import mean_squared_error
# # from sklearn.model_selection import learning_curve

from kernels_non_stationary import LocalLengthScalesKernel, ManifoldKernel

from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, ConstantKernel as C

# # Define custom optimizer for hyperparameter-tuning of non-stationary kernel
# def de_optimizer(obj_func, initial_theta, bounds):
#     res = differential_evolution(lambda x: obj_func(x, eval_gradient=False),
#                                   bounds, maxiter=50, disp=False, polish=False)
#     return res.x, obj_func(res.x, eval_gradient=False)

# Define custom optimizer for hyperparameter-tuning of non-stationary kernel
def de_optimizer(obj_func, initial_theta, bounds):
    res = differential_evolution(lambda x: obj_func(x, eval_gradient=True),
                                  bounds, maxiter=300, disp=False, polish=False)
    return res.x, obj_func(res.x, eval_gradient=False)

# # # Specify stationary and non-stationary kernel
# # kernel_matern = C(1.0, (1e-10, 1000)) \
# #     * Matern(length_scale_bounds=(1e-1, 1e3), nu=1.5)
# # gp_matern = GaussianProcessRegressor(kernel=kernel_matern)

# kernel_lls = C(1.0, (1e-10, 1000)) \
#   * LocalLengthScalesKernel.construct(X_tr_scaled, l_L=0.1, l_U=2.0, l_samples=5)
# gp_lls = GaussianProcessRegressor(kernel=kernel_lls, optimizer=de_optimizer)

# # kernel_m = C(1.0, (1e-10, 1000)) \
# #   * ManifoldKernel.construct(X_tr_scaled, l_L=0.1, l_U=2.0, l_samples=5)
# # gp_m = GaussianProcessRegressor(kernel=kernel_m, optimizer=de_optimizer)

# # #n_samples = 100
# # n_features = 4
# # n_dim_manifold = 2
# # n_hidden = 3
# # architecture=((n_features, n_hidden, n_dim_manifold),)
# # kernel_nn = C(1.0, (1e-10, 100)) \
# #     * ManifoldKernel.construct(base_kernel=RBF(0.1, (1.0, 1000.0)),
# #                                architecture=architecture,
# #                                transfer_fct="tanh", max_nn_weight=10.0) \
# #     + WhiteKernel(1e-3, (1e-10, 1e-1))
# # gp_nn = GaussianProcessRegressor(kernel=kernel_nn, alpha=0,
# #                                  n_restarts_optimizer=3)

# # # Fit GPs
# gp_lls.fit(X_tr_scaled, Y_tr_scaled)
# # gp_nn.fit(X_tr_scaled, Y_tr_scaled)


# y_mean_lls, y_std_lls = gp_lls.predict(X_ts_scaled, return_std=True)
# # print('# RMSE lls : ' + str(rmse(y_mean_lls, Y_ts)))

# # y_mean_nn, y_std_nn = gp_nn.predict(X_ts_scaled, return_std=True)
# # print('# RMSE lls : ' + str(rmse(y_mean_nn, Y_ts)))

# # # Pass the predictions through inverse scaling transform to compare them in the original data space
# Y_pred    = scalerY.inverse_transform(y_mean_lls)
# # Y_pred_s  = scalerY.inverse_transform(y_mean_nn)
# print('# RMSE lls : ' + str(rmse(Y_pred, Y_ts)))
# # print('# RMSE lls : ' + str(rmse(Y_pred_s, Y_ts)))

#Matern52
#RBF
#Matern32

from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy.optimize

class MyGPR(GaussianProcessRegressor):
    def __init__(self, *args, max_iter=15000, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_iter = max_iter

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        def new_optimizer(obj_func, initial_theta, bounds):
            return scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                max_iter=self._max_iter,
            )
        self.optimizer = new_optimizer
        return super()._constrained_optimization(obj_func, initial_theta, bounds)


def GP_model(train_inputs,train_outputs,test_inputs, test_outputs, k_type, k_name, run_id, kk, sema):
    """Train Gaussian Process surrigate for a given function

    Parameters UPDATE and add SHAPES for inputs/outputs
    ----------
    retrieved_Inputs : (Mxn) numpy array
        Collection of possible input values to the black-box
        
    Returns
    -------
    GPy regression model
        a trained gaussian process model
    """ 
    trained_flag = 0
    attempt = 0
    while(trained_flag==0):
        try:
            
            # Train regression function
            model = GPy.models.GPRegression(train_inputs,train_outputs,kernel = k_type)
            model['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)
        
            # optimise parameters
            model.optimize()
            model.optimize_restarts(num_restarts = 10, verbose=False)
                
            GP_means, GP_vars = model.predict(test_inputs)
                      
            # Get RMSE of Validation data
            RMSE = mean_squared_error(GP_means, test_outputs, squared=False) 
              
            # Get Max Error
            MAX = max_error(GP_means, test_outputs) 
              
            # R2 Score
            r2 = r2_score(test_outputs,GP_means) 
            
            with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_GP_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
              pickle.dump(model, f)        
            with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_RMSE_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
                pickle.dump(RMSE, f) 
            with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_MAX_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
                pickle.dump(MAX, f) 
            with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_r2_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
                pickle.dump(r2, f)  
            with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_test_inputs_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
                pickle.dump(test_inputs, f)  
            with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_test_outputs_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
                pickle.dump(test_outputs, f)  
            # with open("ComparingKernels\k_"+str(k_name)+ "_k_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
            #     pickle.dump(k_type, f)  
            
            if sema:
                sema.release()
            trained_flag = 1
    
        except:
            trained_flag = 0
            attempt += 1
            print("........FAILED........Attempt",attempt )
            
            if attempt == 10:
                print("FAILED:",k_name+"_GP_rep_"+str(run_id)+"_k_" +str(kk) )
                trained_flag  = 1
                if sema:
                    sema.release()
                
        # while(trained_flag == 0):
        #     try:
                
        #         # Train regression function
        #         model = GPy.models.GPRegression(train_inputs,train_outputs,kernel = k_type)
        #         model['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)
            
        #         # optimise parameters
        #         model.optimize()
        #         model.optimize_restarts(num_restarts = 10, verbose=False)
                    
        #         GP_means, GP_vars = model.predict(test_inputs)
                          
        #         # Get RMSE of Validation data
        #         RMSE = mean_squared_error(GP_means, test_outputs, squared=False) 
                  
        #         # Get Max Error
        #         MAX = max_error(GP_means, test_outputs) 
                  
        #         # R2 Score
        #         r2 = r2_score(test_outputs,GP_means) 
                
        #         with open("ComparingKernels\k_"+str(k_name)+ "_GP_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
        #           pickle.dump(model, f)        
        #         with open("ComparingKernels\k_"+str(k_name)+ "_RMSE_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
        #             pickle.dump(RMSE, f) 
        #         with open("ComparingKernels\k_"+str(k_name)+ "_MAX_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
        #             pickle.dump(MAX, f) 
        #         with open("ComparingKernels\k_"+str(k_name)+ "_r2_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
        #             pickle.dump(r2, f)  
        #         with open("ComparingKernels\k_"+str(k_name)+ "_test_inputs_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
        #             pickle.dump(test_inputs, f)  
        #         with open("ComparingKernels\k_"+str(k_name)+ "_test_outputs_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
        #             pickle.dump(test_outputs, f)  
        #         # with open("ComparingKernels\k_"+str(k_name)+ "_k_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
        #         #     pickle.dump(k_type, f)  
                
        #         if sema:
        #             sema.release()
                    
        #         trained_flag += 1
        
        #     except:
        #         trained_flag = 0
        #         attempt += 1
        #         print("........FAILED........Attempt",attempt )
        # #     if sema:
        # #         sema.release()
                        
    #return()
            
#---------------  Cross Validation of Performances  ---------------#
from sklearn.metrics import mean_squared_error,max_error,r2_score
from sklearn.model_selection import KFold
import multiprocess
from numpy import genfromtxt
def run_stationary_kernels_comparison():
    #Inputs = pickle.load(open("ISR3D_data\InputList.pkl", "rb")).to_numpy()
    #data = np.load('ISR3D_data\LumenData.npy')
    Inputs = np.load('ISR3D_data\input_list_numberseq.npy') #pickle.load(open("ISR3D_data\InputList.pkl", "rb")).to_numpy()
    #data = np.load('ISR3D_data\LumenData.npy')
    MinLumenArea = np.zeros((128))
    for i in range(128):
        MinLumenArea[i] = genfromtxt('ISR3D_data\MinAreaData\d'+str(i)+'.csv', delimiter=',')[1:,0].min()*(0.03125**2) 
    retrieved_Outputs = MinLumenArea
    n_sims = 30
    
    #sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    sema = multiprocess.Semaphore(int(8))
    
    # Convert to Areas
    #retrieved_Outputs = data[:,-1,:].mean(axis = 1)
    
    rn = range(1,128)
    jobs = []
    
    n = 4
    #Kernel_names = ["RBF","Bias","Cosine","ExpQuad","Exponential","GridRBF","Integral","Integral_Limits","Linear","MLP","Matern32","Matern52","OU","Poly","RBF","RatQuad","Spline","StdPeriodic","sde_Bias","sde_Exponential","sde_Matern32","sde_Matern52" ,"sde_RBF","sde_RatQuad"]
    #Kernel_list = [GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.Bias(input_dim=n),GPy.kern.Cosine(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.ExpQuad(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.Exponential(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.GridRBF(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.Integral(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.Integral_Limits(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.Linear(n, np.ones(n) * 2., ARD=True),GPy.kern.MLP(n, np.ones(n) * 2, np.ones(n) * 1.5,np.ones(n) * 1.5, ARD=True),GPy.kern.Matern32(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.Matern52(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.OU(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.Poly(n, np.ones(n) * 2, np.ones(n) * 2,np.ones(n) * 2),GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.RatQuad(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.Spline(n, np.ones(n) * .5, np.ones(n) * 2),GPy.kern.StdPeriodic(n, .5, np.ones(1) * 2),GPy.kern.sde_Bias(input_dim=n),GPy.kern.sde_Exponential(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.sde_Matern32(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.sde_Matern52(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.sde_RBF(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.sde_RatQuad(n, .5, np.ones(n) * 2., ARD=True)]
    
    #Kernel_names = ["RBF","Bias","ExpQuad","Exponential","GridRBF","Linear","Matern32","Matern52","OU","RatQuad","StdPeriodic"]
    #Kernel_list = [GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.Bias(input_dim=n),GPy.kern.ExpQuad(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.Exponential(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.GridRBF(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.Linear(n, np.ones(n) * 2., ARD=True),GPy.kern.Matern32(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.Matern52(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.OU(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.RatQuad(n, .5, np.ones(n) * 2., ARD=True),GPy.kern.StdPeriodic(n, .5, np.ones(1) * 2)]


    #Kernel_list = [GPy.kern.Cosine(n, .5, np.ones(n) * 2., ARD=True)]
    #Kernel_names = ["Cosine"]
    for i in range(13,14):
        print("Starting iteration " +str(i))
        
        kf10 = KFold(n_splits=5, shuffle=True)
        k = 0
        for train_index, test_index in kf10.split(rn):
            
            train_Inputs = Inputs[train_index]
            train_Outputs = retrieved_Outputs[train_index,None]
            
            test_Inputs = Inputs[test_index]
            test_Outputs = retrieved_Outputs[test_index,None]
            
            
            Kernel_list = [GPy.kern.RatQuad(n, .5, np.ones(n) * 2., ARD=True)]#GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True)
            Kernel_names = ["RatQuad"]
            #GP_model(train_Inputs,train_Outputs,test_Inputs,test_Outputs,kernel_type,kernel_name,i,k,sema = None)
    
            for indx,kernel_name in enumerate(Kernel_names):
                kernel_type = Kernel_list[indx]
                #GP_model(train_Inputs,train_Outputs,test_Inputs,test_Outputs,kernel_type,kernel_name,i,k,sema = None)
    
                
                sema.acquire()       
                p = multiprocess.Process(target=GP_model, args=(train_Inputs,train_Outputs,test_Inputs,test_Outputs,kernel_type,kernel_name,i,k,sema))
                jobs.append(p)
                p.start()
                

                    #GP_model(train_Inputs,train_Outputs,test_Inputs,test_Outputs,kernel_type,kernel_name,i,k,sema = None)
          
                sema.acquire()       
                p = multiprocess.Process(target=GP_model, args=(train_Inputs,train_Outputs,test_Inputs,test_Outputs,kernel_type,kernel_name,i,k,sema))
                jobs.append(p)
                p.start()
            
            
            
            k += 1
            
    # for i in range(n_sims):
    #     print("Starting multiplication iteration " +str(i))
        
    #     kf10 = KFold(n_splits=5, shuffle=True)
    #     k = 0
    #     for train_index, test_index in kf10.split(rn):
            
    #         train_Inputs = Inputs[train_index]
    #         train_Outputs = retrieved_Outputs[train_index,None]
            
    #         test_Inputs = Inputs[test_index]
    #         test_Outputs = retrieved_Outputs[test_index,None]
            
    #         for indx1,kernel_name1 in enumerate(Kernel_names):
    #             for indx2,kernel_name2 in enumerate(Kernel_names):
    #                 kernel_name = kernel_name1+str("_x_")+kernel_name2
    #                 kernel_type = Kernel_list[indx1]*Kernel_list[indx2]
                    
    #                 sema.acquire()       
    #                 p = multiprocess.Process(target=GP_model, args=(train_Inputs,train_Outputs,test_Inputs,test_Outputs,kernel_type,kernel_name,i,k,sema))
    #                 jobs.append(p)
    #                 p.start()
            
    #         k += 1
            
    # for i in range(n_sims):
    #     print("Starting addition iteration " +str(i))
        
    #     kf10 = KFold(n_splits=5, shuffle=True)
    #     k = 0
    #     for train_index, test_index in kf10.split(rn):
            
    #         train_Inputs = Inputs[train_index]
    #         train_Outputs = retrieved_Outputs[train_index,None]
            
    #         test_Inputs = Inputs[test_index]
    #         test_Outputs = retrieved_Outputs[test_index,None]
            
    #         for indx1,kernel_name1 in enumerate(Kernel_names):
    #             for indx2,kernel_name2 in enumerate(Kernel_names):
    #                 kernel_name = kernel_name1+str("_+_")+kernel_name2
    #                 kernel_type = Kernel_list[indx1] + Kernel_list[indx2]
                    
    #                 sema.acquire()       
    #                 p = multiprocess.Process(target=GP_model, args=(train_Inputs,train_Outputs,test_Inputs,test_Outputs,kernel_type,kernel_name,i,k,sema))
    #                 jobs.append(p)
    #                 p.start()
            
            #k += 1
            
    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
        
        
# from tqdm import tqdm
# RMSE = np.zeros((10,10))
# i = 0
# for n_dim_manifold in tqdm(range(10)):
    
#     #RMSE = []
#     j = 0
#     for n_hidden in range(1,10):
        
#         #n_dim_manifold = 2
#         #n_hidden = 3
#         architecture=((n_features, n_hidden, n_dim_manifold),)
#         kernel_nn = C(1.0, (1e-10, 100)) \
#             * ManifoldKernel.construct(base_kernel=RBF(0.1, (1.0, 1000.0)),
#                                         architecture=architecture,
#                                         transfer_fct="tanh", max_nn_weight=10.0) \
#             + WhiteKernel(1e-3, (1e-10, 1e-1))
#         gp_nn = GaussianProcessRegressor(kernel=kernel_nn, alpha=0,
#                                           n_restarts_optimizer=3)

#         # Fit GPs
#         gp_nn.fit(X_tr_scaled, Y_tr_scaled)
        
#         y_mean_nn, y_std_nn = gp_nn.predict(X_ts_scaled, return_std=True)
#         RMSE[i,j] = rmse(y_mean_nn, Y_ts)
#         #print('# RMSE:' + str(rmse(y_mean_nn, Y_ts)), "_n_hidden_",str(n_hidden)+ "_manifold", str(n_dim_manifold))
#         j += 1
#     i += 1
# print(RMSE)

# import matplotlib.pyplot as plt
# for i in range(10):
#     plt.plot(RMSE[i,0:8], label = i)
# plt.legend()
# plt.show()
# # Best is 3 dim manifold with 4 hidden layers
from tqdm import tqdm
def run_NonStationary_kernels_comparison(n_sims = 30):
    #Inputs = pickle.load(open("ISR3D_data\InputList.pkl", "rb")).to_numpy()
    #data = np.load('ISR3D_data\LumenData.npy')
    #n_sims = 30
    
    Inputs = np.load('ISR3D_data\input_list_numberseq.npy') #pickle.load(open("ISR3D_data\InputList.pkl", "rb")).to_numpy()
    #data = np.load('ISR3D_data\LumenData.npy')
    MinLumenArea = np.zeros((128))
    for i in range(128):
        MinLumenArea[i] = genfromtxt('ISR3D_data\MinAreaData\d'+str(i)+'.csv', delimiter=',')[1:,0].min()*(0.03125**2) 
    retrieved_Outputs = MinLumenArea
    
    #sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    #sema = multiprocess.Semaphore(int(8))
    
    # Convert to Areas
    #retrieved_Outputs = data[:,-1,:].mean(axis = 1)
    
    rn = range(1,128)
    #jobs = []
    
    #n = 4
    for run_id in tqdm(range(1,n_sims)):
        print("Starting multiplication iteration " +str(run_id))
        
        kf10 = KFold(n_splits=5, shuffle=True)
        kk = 0
        for train_index, test_index in kf10.split(rn):
            
            train_Inputs = Inputs[train_index]
            train_Outputs = retrieved_Outputs[train_index,None]
            
            test_Inputs = Inputs[test_index]
            test_Outputs = retrieved_Outputs[test_index,None]
            
            # kernel_lls = RBF(np.ones(4), (1.0, 1000.0))*LocalLengthScalesKernel.construct(train_Inputs, isotropic = False,l_isotropic = False, l_samples=10) + WhiteKernel(noise_level_bounds=(1e-1,1e2))
            # gp_lls = GaussianProcessRegressor(kernel=kernel_lls, n_restarts_optimizer = 10)
            
            # gp_lls.fit(train_Inputs, train_Outputs)
            # y_mean_lls, y_std_lls = gp_lls.predict(test_Inputs, return_std=True)
            
            # #GP_means, GP_vars = model.predict(test_inputs)
                  
            # # Get RMSE of Validation data
            # RMSE = mean_squared_error(y_mean_lls, test_Outputs, squared=False) 
              
            # # Get Max Error
            # MAX = max_error(y_mean_lls, test_Outputs) 
              
            # # R2 Score
            # r2 = r2_score(test_Outputs,y_mean_lls) 
            # k_name = "NonStationary_lls"
            # with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_GP_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
            #   pickle.dump(gp_lls, f)        
            # with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_RMSE_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
            #     pickle.dump(RMSE, f) 
            # with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_MAX_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
            #     pickle.dump(MAX, f) 
            # with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_r2_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
            #     pickle.dump(r2, f)  
            # with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_test_inputs_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
            #     pickle.dump(test_Inputs, f)  
            # with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_test_outputs_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
            #     pickle.dump(test_Outputs, f)  
                
            n_features = 4
            n_hidden = 4
            n_dim_manifold = 3
            architecture=((n_features, n_hidden, n_dim_manifold),)
            kernel_nn = C(1.0, (1e-10, 100)) * ManifoldKernel.construct(base_kernel=RBF(0.1, (1.0, 1000.0)),architecture=architecture,
                                        transfer_fct="tanh", max_nn_weight=15.0) #+ WhiteKernel(1e-3, (1e-10, 1e2))
            gp_nn = GaussianProcessRegressor(kernel=kernel_nn, alpha=0, n_restarts_optimizer=10)

            # Fit GPs
            gp_nn.fit(train_Inputs, train_Outputs)
            
            y_mean_nn, y_std_nn = gp_nn.predict(test_Inputs, return_std=True)  
            
            # Get RMSE of Validation data
            RMSE = mean_squared_error(y_mean_nn, test_Outputs, squared=False) 
              
            # Get Max Error
            MAX = max_error(y_mean_nn, test_Outputs) 
              
            # R2 Score
            r2 = r2_score(test_Outputs,y_mean_nn) 
            k_name = "NonStationary_mnn_NoWhiteKernel"
            with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_GP_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
              pickle.dump(gp_nn, f)        
            with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_RMSE_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
                pickle.dump(RMSE, f) 
            with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_MAX_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
                pickle.dump(MAX, f) 
            with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_r2_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
                pickle.dump(r2, f)  
            with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_test_inputs_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
                pickle.dump(test_Inputs, f)  
            with open("ISR3D_MinAreas_ComparingKernels\k_"+str(k_name)+ "_test_outputs_rep_"+str(run_id)+"_k_" +str(kk) +".dump" , "wb") as f:
                pickle.dump(test_Outputs, f)  
                
                
            kk += 1

            
            # # Pass the predictions through inverse scaling 
            #Y_pred    = scalerY.inverse_transform(y_mean_lls)
            # Y_pred_s  = scalerY.inverse_transform(y_mean_nn)
            #print('# RMSE lls : ' + str(rmse(Y_pred, Y_ts)))

"""
Several Occurrences of: 
    
ConvergenceWarning:The optimal value found for dimension 19 
of parameter k1__k2__w is close to the specified upper bound 10.0.
Increasing the bound and calling fit again may find a better value.
 
ConvergenceWarning:The optimal value found for dimension 19 
of parameter k1__k2__w is close to the specified upper bound -10.0.
Increasing the bound and calling fit again may find a better value.

(for various dimensions )

ConvergenceWarning:The optimal value found for dimension 0 of 
parameter k2__noise_level is close to the specified upper bound 
0.1. Increasing the bound and calling fit again may
find a better value.
 
More rare:
ConvergenceWarning:The optimal value found for dimension 55 of parameter
k1__k2__w is close to the specified lower bound 0.0. Decreasing the bound 
and calling fit again may find a better value.

"""

if __name__ == '__main__':
    #run_sims_for_ISR3D_Retrieved_slice_Data()
    #run_sims_for_ISR3D_Retrieved_Volume_Data()
    run_stationary_kernels_comparison()