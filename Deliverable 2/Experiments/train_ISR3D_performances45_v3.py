# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:06:48 2021

@author: Cillian
"""
import sys
import argparse
import GPy
import AL_strategies
#from utils import *
import numpy as np
#import Gray_Scott
#from Sobol_indices import get_sobol_indices__via_saltelli
import GP_model_ISR3D_performances45_v3 as GP_model
#import GP_model_ISR3D_performances45_v2 as GP_model
import pickle
#import time
from sklearn.metrics import mean_squared_error,max_error,r2_score,mean_absolute_percentage_error
from sklearn.model_selection import KFold
import multiprocess
import time

# from sklearn import *
# def mean_absolute_percentage_error(y_true, y_pred,
#                                    sample_weight=None,
#                                    multioutput='uniform_average'):
#     """Mean absolute percentage error regression loss.
#     Note here that we do not represent the output as a percentage in range
#     [0, 100]. Instead, we represent it in range [0, 1/eps]. Read more in the
#     :ref:`User Guide <mean_absolute_percentage_error>`.
#     .. versionadded:: 0.24
#     Parameters
#     ----------
#     y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
#         Ground truth (correct) target values.
#     y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
#         Estimated target values.
#     sample_weight : array-like of shape (n_samples,), default=None
#         Sample weights.
#     multioutput : {'raw_values', 'uniform_average'} or array-like
#         Defines aggregating of multiple output values.
#         Array-like value defines weights used to average errors.
#         If input is list then the shape must be (n_outputs,).
#         'raw_values' :
#             Returns a full set of errors in case of multioutput input.
#         'uniform_average' :
#             Errors of all outputs are averaged with uniform weight.
#     Returns
#     -------
#     loss : float or ndarray of floats in the range [0, 1/eps]
#         If multioutput is 'raw_values', then mean absolute percentage error
#         is returned for each output separately.
#         If multioutput is 'uniform_average' or an ndarray of weights, then the
#         weighted average of all output errors is returned.
#         MAPE output is non-negative floating point. The best value is 0.0.
#         But note the fact that bad predictions can lead to arbitarily large
#         MAPE values, especially if some y_true values are very close to zero.
#         Note that we return a large value instead of `inf` when y_true is zero.
#     Examples
#     --------
#     >>> from sklearn.metrics import mean_absolute_percentage_error
#     >>> y_true = [3, -0.5, 2, 7]
#     >>> y_pred = [2.5, 0.0, 2, 8]
#     >>> mean_absolute_percentage_error(y_true, y_pred)
#     0.3273...
#     >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
#     >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
#     >>> mean_absolute_percentage_error(y_true, y_pred)
#     0.5515...
#     >>> mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
#     0.6198...
#     """
#     y_type, y_true, y_pred, multioutput = _check_reg_targets(
#         y_true, y_pred, multioutput)
#     check_consistent_length(y_true, y_pred, sample_weight)
#     epsilon = np.finfo(np.float64).eps
#     mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
#     output_errors = np.average(mape,
#                                weights=sample_weight, axis=0)
#     if isinstance(multioutput, str):
#         if multioutput == 'raw_values':
#             return output_errors
#         elif multioutput == 'uniform_average':
#             # pass None as weights to np.average: uniform mean
#             multioutput = None

#     return np.average(output_errors, weights=multioutput)

# class NonStationaryCorrelation(object):
#     """ Non-stationary correlation model with predetermined length-scales."""

#     def fit(self, X, nugget=10. * MACHINE_EPSILON):
#         self.X = X
#         self.nugget = nugget
#         self.n_samples = self.X.shape[0]

#         # Calculate array with shape (n_eval, n_features) giving the
#         # componentwise distances between locations x and x' at which the
#         # correlation model should be evaluated.
#         self.D, self.ij = l1_cross_differences(self.X)

#         # Calculate length scales
#         self.l_train = length_scale(self.X)

#     def __call__(self, theta, X=None):
#         # Prepare distances and length scale information for any pair of
#         # datapoints, whose correlation shall be computed
#         if X is not None:
#             # Get pairwise componentwise L1-differences to the input training
#             # set
#             d = X[:, np.newaxis, :] - self.X[np.newaxis, :, :]
#             d = d.reshape((-1, X.shape[1]))
#             # Calculate length scales
#             l_query = length_scale(X)
#             l = np.transpose([np.tile(self.l_train, len(l_query)),
#                               np.repeat(l_query, len(self.l_train))])
#         else:
#             # No external datapoints given; auto-correlation of training set
#             # is used instead
#             d = self.D
#             l = self.l_train[self.ij]

#         # Compute general Matern kernel for nu=1.5
#         nu = 1.5
#         if d.ndim > 1 and theta.size == d.ndim:
#             activation = np.sum(theta.reshape(1, d.ndim) * d ** 2, axis=1)
#         else:
#             activation = theta[0] * np.sum(d ** 2, axis=1)
#         tmp = 0.5*(l**2).sum(1)
#         tmp2 = 2*np.sqrt(nu * activation / tmp)
#         r = np.sqrt(l[:, 0]) * np.sqrt(l[:, 1]) / (gamma(nu) * 2**(nu - 1))
#         r /= np.sqrt(tmp)
#         r *= tmp2**nu * kv(nu, tmp2)

#         # Convert correlations to 2d matrix
#         if X is not None:
#             return r.reshape(-1, self.n_samples)
#         else:  # exploit symmetry of auto-correlation
#             R = np.eye(self.n_samples) * (1. + self.nugget)
#             R[self.ij[:, 0], self.ij[:, 1]] = r
#             R[self.ij[:, 1], self.ij[:, 0]] = r
#             return R

#     def log_prior(self, theta):
#         # Just use a flat prior
#         return 0

def train_model(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps, run_id,s, sema = None):
    try:
        pickle.load(open("GP_data_ISR3D\GP_"+str(AL)+ "_slice_"+str(s)+"_Inputs_rep_"+str(run_id) +".dump", "rb"))
        if sema:
            sema.release()
    
    except:
        try:
            GP, GP_Inputs = GP_model.GP_model(retrieved_Inputs,retrieved_Outputs,AL_strategy,parameter_ranges=ps)
            with open("GP_data_ISR3D\GP_" + str(AL)+ "_slice_"+str(s)+"_GP_rep_"+str(run_id) +".dump" , "wb") as f:
              pickle.dump(GP, f) 
              
            with open("GP_data_ISR3D\GP_"+str(AL)+ "_slice_"+str(s)+"_Inputs_rep_"+str(run_id) +".dump" , "wb") as f:
              pickle.dump(GP_Inputs, f) 
                    
            if sema:
                sema.release()
        except:
            trained_flag = 0
            attempt_nr = 0
            while trained_flag == 0:
                try:
                    GP, GP_Inputs = GP_model.GP_model(retrieved_Inputs,retrieved_Outputs,AL_strategy,parameter_ranges=ps)
                    with open("GP_data_ISR3D\GP_" + str(AL)+ "_slice_"+str(s)+"_GP_rep_"+str(run_id) +".dump" , "wb") as f:
                      pickle.dump(GP, f) 
                      
                    with open("GP_data_ISR3D\GP_"+str(AL)+ "_slice_"+str(s)+"_Inputs_rep_"+str(run_id) +".dump" , "wb") as f:
                      pickle.dump(GP_Inputs, f) 
                
                    trained_flag = 1
                    if sema:
                        sema.release()
                    
                except:
                    attempt_nr += 1
                    print("Still not training.... having an error somewhere. On attempt number"+str(attempt_nr))
                    
def train_validate_model(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps, run_id,s, sema = None):
    try:

        pickle.load(open(str(save_location)+str(AL)+ "NO_slice_"+str(s)+"_Inputs_rep_"+str(run_id)+"_k_" +str(k) +".dump", "rb"))

        if sema:
            sema.release()
    
    except:
        try:

            GP, GP_Inputs, GP_Outputs, RMSE, MAX,r2,MAPE  = GP_model.GP_model_batchClustering(batch_size,retrieved_Inputs,retrieved_Outputs,AL_strategy, test_Inputs,test_Outputs,parameter_ranges=ps)

            print("I'm fully trained! Now time to be tested!!")
            with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_MAPE_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
              pickle.dump(MAPE, f) 
            
            with open(str(save_location)+ str(AL)+ "_slice_"+str(s)+"_GP_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
              pickle.dump(GP, f) 
              
            with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_Inputs_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
              pickle.dump(GP_Inputs, f) 
            with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_Outputs_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
              pickle.dump(GP_Outputs, f)        
            with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_RMSE_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
              pickle.dump(RMSE, f) 
            with open(str(save_location)++str(AL)+ "_slice_"+str(s)+"_MAX_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
              pickle.dump(MAX, f) 
            with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_r2_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
              pickle.dump(r2, f)               
              
            with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_ValidationInputData_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
              pickle.dump(test_Inputs, f)      
              
            with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_ValidationOutputData_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:

              pickle.dump(test_Outputs, f)  
            print("I'm fully trained and tested!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
            print(RMSE)
            if sema:
                sema.release()
        except:
            trained_flag = 0
            attempt_nr = 0
            while trained_flag == 0:
                try:

                    GP, GP_Inputs, GP_Outputs, RMSE, MAX,r2,MAPE  = GP_model.GP_model(batch_size,retrieved_Inputs,retrieved_Outputs,AL_strategy, test_Inputs,test_Outputs,parameter_ranges=ps)
        
                    print("I'm fully trained! Now time to be tested!!")
                    with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_MAPE_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
                        pickle.dump(MAPE, f) 
                    
                    with open(str(save_location)+ str(AL)+ "_slice_"+str(s)+"_GP_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
                      pickle.dump(GP, f) 
                      
                    with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_Inputs_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
                      pickle.dump(GP_Inputs, f) 
                    with open(str(save_location)+ str(AL)+ "_slice_"+str(s)+"_Outputs_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
                      pickle.dump(GP_Outputs, f)        
                    with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_RMSE_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
                      pickle.dump(RMSE, f) 
                    with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_MAX_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
                      pickle.dump(MAX, f) 
                    with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_r2_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
                      pickle.dump(r2, f)               
                      
                    with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_ValidationInputData_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
                      pickle.dump(test_Inputs, f)      
                      
                    with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_ValidationOutputData_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:

                      pickle.dump(test_Outputs, f)  
                    print("I'm fully trained and tested!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
              
                    trained_flag = 1
                    if sema:
                        sema.release()
                    
                except:
                    attempt_nr += 1
                    print("Still not training.... having an error somewhere. On attempt number"+str(attempt_nr))
  
    #except:
    #    if sema:
    #        sema.release()
        
def run_sims_for_ISR3D_Retrieved_slice_Data(n_sims = 30):
    
    jobs = []
    
    Inputs = pickle.load(open("ISR3D_data\InputList.pkl", "rb"))
    data = np.load('ISR3D_data\LumenData.npy')
    
    retrieved_Inputs = Inputs.to_numpy()
    
    # Set parameter ranges
    ps = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    
    #sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    sema = multiprocess.Semaphore(int(5))
    
    # For different slices
    for s in range(10,128,10):
        retrieved_Outputs = data[:,-1,s]
    

    
        
        
        # AL_strategy = AL_strategies.ALM
        # AL = "ALM"
        # for i in range(n_sims):
            
        #     sema.acquire()
            
        #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
        #     jobs.append(p)
        #     p.start()
            
        # AL_strategy = AL_strategies.ALC
        # AL = "ALC"
        # for i in range(n_sims):
            
        #     sema.acquire()
            
        #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
        #     jobs.append(p)
        #     p.start()
            
        # AL_strategy = AL_strategies.SLRGP
        # AL = "SLRGP"
        # for i in range(n_sims):
            
        #     sema.acquire()
            
        #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
        #     jobs.append(p)
        #     p.start()
            
        AL_strategy = AL_strategies.SLRGP_Batch
        AL = "SLRGP_Batch_5"
        for i in range(n_sims):
            
            sema.acquire()
            
            p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
            jobs.append(p)
            p.start()
        
            
        # AL_strategy = AL_strategies.IMSE
        # AL = "IMSE"
        # for i in range(n_sims):
            
        #     sema.acquire()
            
        #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
        #     jobs.append(p)
        #     p.start()
            
        # AL_strategy = AL_strategies.MMSE
        # AL = "MMSE"
        # for i in range(n_sims):
            
        #     sema.acquire()
            
        #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
        #     jobs.append(p)
        #     p.start()
            
        # AL_strategy = AL_strategies.SLRV
        # AL = "SLRV"
        # for i in range(n_sims):
            
        #     sema.acquire()
            
        #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
        #     jobs.append(p)
        #     p.start()
        
    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
        

def run_sims_for_ISR3D_Retrieved_Volume_Data(n_sims = 100):
    
    jobs = []
    
    # For naming conventions
    s = "_All_Volume_"
    
    Inputs = pickle.load(open("ISR3D_data\InputList.pkl", "rb"))
    data = np.load('ISR3D_data\LumenData.npy')
    
    retrieved_Inputs = Inputs.to_numpy()
    
    # Set parameter ranges
    ps = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    
    sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    
    # Convert to Areas
    retrieved_Outputs = data[:,-1,:].sum(axis = 1)*0.03125
    
    # For different slices
    #for s in range(10,128,10):
    #    retrieved_Outputs = data[:,-1,s]
    

    
        
        
    # AL_strategy = AL_strategies.ALM
    # AL = "ALM"
    # for i in range(n_sims):
        
    #     sema.acquire()
        
    #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
    #     jobs.append(p)
    #     p.start()
        
    # AL_strategy = AL_strategies.ALC
    # AL = "ALC"
    # for i in range(n_sims):
        
    #     sema.acquire()
        
    #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
    #     jobs.append(p)
    #     p.start()
            
    AL_strategy = AL_strategies.SLRGP
    AL = "SLRGP"
    for i in range(13,n_sims):
        
        sema.acquire()
        
        p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
        jobs.append(p)
        p.start()
        
            
    # AL_strategy = AL_strategies.IMSE
    # AL = "IMSE"
    # for i in range(n_sims):
        
    #     sema.acquire()
        
    #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
    #     jobs.append(p)
    #     p.start()
        
    # AL_strategy = AL_strategies.MMSE
    # AL = "MMSE"
    # for i in range(n_sims):
        
    #     sema.acquire()
        
    #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
    #     jobs.append(p)
    #     p.start()
            
        # AL_strategy = AL_strategies.SLRV
        # AL = "SLRV"
        # for i in range(n_sims):
            
        #     sema.acquire()
            
        #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
        #     jobs.append(p)
        #     p.start()
        
    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
   
from sklearn.model_selection import cross_validate
scoring = ["neg_root_mean_squared_error", "r2", "max_error"]

def run_sims_for_Volume_with_CrossValidation(n_sims = 100):
    
    jobs = []
    
    # For naming conventions
    s = "_All_Volume_"
    
    Inputs = pickle.load(open("ISR3D_data\InputList.pkl", "rb"))
    data = np.load('ISR3D_data\LumenData.npy')
    
    retrieved_Inputs = Inputs.to_numpy()
    
    # Set parameter ranges
    ps = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    
    sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    
    # Convert to Areas
    Length_of_centerline = 16.80000426024675
    retrieved_Outputs = data[:,-1,:].sum(axis = 1)*0.03125/Length_of_centerline
    
    # For different slices
    #for s in range(10,128,10):
    #    retrieved_Outputs = data[:,-1,s]
    

    
        
        
    # AL_strategy = AL_strategies.ALM
    # AL = "ALM"
    # for i in range(n_sims):
        
    #     sema.acquire()
        
    #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
    #     jobs.append(p)
    #     p.start()
        
    # AL_strategy = AL_strategies.ALC
    # AL = "ALC"
    # for i in range(n_sims):
        
    #     sema.acquire()
        
    #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
    #     jobs.append(p)
    #     p.start()
            
    AL_strategy = AL_strategies.SLRGP
    AL = "SLRGP"
    for i in range(13,n_sims):
        
        sema.acquire()
        
        p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
        jobs.append(p)
        p.start()
        
            
    # AL_strategy = AL_strategies.IMSE
    # AL = "IMSE"
    # for i in range(n_sims):
        
    #     sema.acquire()
        
    #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
    #     jobs.append(p)
    #     p.start()
        
    # AL_strategy = AL_strategies.MMSE
    # AL = "MMSE"
    # for i in range(n_sims):
        
    #     sema.acquire()
        
    #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
    #     jobs.append(p)
    #     p.start()
            
        # AL_strategy = AL_strategies.SLRV
        # AL = "SLRV"
        # for i in range(n_sims):
            
        #     sema.acquire()
            
        #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
        #     jobs.append(p)
        #     p.start()
        
    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
        
from numpy import genfromtxt
import time

def run_sims_for_Volume_with_CrossValidation(iteration):
    # if int(iteration) == 1:
    #     n_min = 6
    # if int(iteration) == 2:
    #     n_min = 8
    # if int(iteration) == 3:
    #     n_min = 9
    # if int(iteration) == 4:
    #     n_min = 18
    # if int(iteration) == 5:
    #     n_min = 19

    
    jobs = []
    
    # For naming conventions
    s = "_All_Volume_"
    
    Inputs = np.load('ISR3D_data\input_list_numberseq.npy') #pickle.load(open("ISR3D_data\InputList.pkl", "rb")).to_numpy()
    #data = np.load('ISR3D_data\LumenData.npy')

    #MinLumenArea = np.zeros((128))
    #for i in range(128):
    #    MinLumenArea[i] = genfromtxt('ISR3D_data\MinAreaData\d'+str(i)+'.csv', delimiter=',')[1:,0].min()*(0.03125**2) 
    #
    #np.load('PercentLost_LumenArea.npy')
    
    # Save Location
    save_location = "ISR3D_SLRGP_Batches4\SD_"
    
    #retrieved_Outputs = np.load('PercentLost_LumenArea.npy')[:,250]# One Slice

    #retrieved_Inputs = Inputs.to_numpy()
    

    #retrieved_Outputs = np.load('ISR3D_data\Max_PercentLost_LumenArea.npy') 
    #retrieved_Outputs = np.load('ISR3D_data\Mean_PercentLost_LumenArea.npy') 
    retrieved_Outputs = np.load('ISR3D_data\SD_PercentLost_LumenArea.npy') 
    
    # Set parameter ranges
    ps = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    
    #sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    sema = multiprocess.Semaphore(int(3))
    
    # Convert to Areas
    #retrieved_Outputs = data[:,-1,:].mean(axis = 1)
    
    #Length_of_centerline = 16.80000426024675
    #retrieved_Outputs = data[:,-1,:].sum(axis = 1)*0.03125/Length_of_centerline
    
    #retrieved_Outputs = data[:,-1,:].sum(axis = 1)*0.03125
    
    # For different slices
    #for s in range(10,128,10):
    #    retrieved_Outputs = data[:,-1,s]
    
    # Create range 1 to 128
    rn = range(1,128)
    start = time.time()
    kkkkk = int(iteration)

    for i in range(int(iteration)-1,int(iteration)):

        start = time.time()
        print("Starting iteration " +str(i))
        
        kf10 = KFold(n_splits=5, shuffle=True)
        k = 0
        for train_index, test_index in kf10.split(rn):
            
            retrieved_Inputs = Inputs[train_index]
            Outputs = retrieved_Outputs[train_index]
            
            test_Inputs = Inputs[test_index]
            test_Outputs = retrieved_Outputs[test_index]
            

            # AL_strategy = AL_strategies.ALM
            # AL = "ALM_NoiseFixRelease"                
            # sema.acquire()

            
            # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            
            # jobs.append(p)
            # p.start()
                
            # AL_strategy = AL_strategies.ALC
            # AL = "AverageCSArea_ALC"
            # #for i in range(n_sims):
                
            # sema.acquire()
            

            # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            
            # jobs.append(p)
            # p.start()
            
            # AL_strategy = AL_strategies.SLRGP_Batch
            # batch_size = 1
            # AL = "SLRGP"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()
            
            # AL_strategy = AL_strategies.SLRGP_Batch_Group0_2
            # batch_size = 5
            # AL = "SLRGP_Batch_Distances0_5fac2"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()
            
            # AL_strategy = AL_strategies.SLRGP_Batch_Group0_3
            # batch_size = 5
            # AL = "SLRGP_Batch_Distances0_5fac3"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()

            # AL_strategy = AL_strategies.SLRGP_Batch_Group0_4
            # batch_size = 5
            # AL = "SLRGP_Batch_Distances0_5fac4"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()

            # AL_strategy = AL_strategies.SLRGP_Batch_Group0_5
            # batch_size = 5
            # AL = "SLRGP_Batch_Distances0_5fac5"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()  
            
            # AL_strategy = AL_strategies.SLRGP_Batch_Group0_6
            # batch_size = 5
            # AL = "SLRGP_Batch_Distances0_5fac6"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()     
            
            # AL_strategy = AL_strategies.SLRGP_Batch_Group1_2
            # batch_size = 5
            # AL = "SLRGP_Batch_Distances1_5fac2"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()
            
            # AL_strategy = AL_strategies.SLRGP_Batch_Group1_3
            # batch_size = 5
            # AL = "SLRGP_Batch_Distances1_5fac3"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()

            # AL_strategy = AL_strategies.SLRGP_Batch_Group1_4
            # batch_size = 5
            # AL = "SLRGP_Batch_Distances1_5fac4"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()

            # AL_strategy = AL_strategies.SLRGP_Batch_Group1_5
            # batch_size = 5
            # AL = "SLRGP_Batch_Distances1_5fac5"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()  
            
            # AL_strategy = AL_strategies.SLRGP_Batch_Group1_6
            # batch_size = 5
            # AL = "SLRGP_Batch_Distances1_5fac6"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()   
                    
            # AL_strategy = AL_strategies.SLRGP_Batch
            # batch_size = 2
            # AL = "SLRGP_Batch2_NewRevalGP"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()
            
            AL_strategy = AL_strategies.SLRGP_Batch
            batch_size = 5
            AL = "SLRGP_Batch5_NewRevalGP"
            #for i in range(n_sims):
            sema.acquire()
            p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            jobs.append(p)
            p.start()            

            # AL_strategy = AL_strategies.SLRGP_Batch_Clustering2
            # batch_size = 5
            # AL = "SLRGP_Batch5_2Clustering"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()         
            
            # AL_strategy = AL_strategies.SLRGP_Batch_Clustering3
            # batch_size = 5
            # AL = "SLRGP_Batch5_3Clustering"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()     

            # AL_strategy = AL_strategies.SLRGP_Batch_Clustering4
            # batch_size = 5
            # AL = "SLRGP_Batch5_4Clustering"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()        
            
            # AL_strategy = AL_strategies.SLRGP_Batch_Clustering5
            # batch_size = 5
            # AL = "SLRGP_Batch5_5Clustering"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()   
            
            # AL_strategy = AL_strategies.SLRGP_Batch_Clustering6
            # batch_size = 5
            # AL = "SLRGP_Batch5_6Clustering"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start() 
                       
            # AL_strategy = AL_strategies.IMSE
            # AL = "AverageCSArea_IMSE"
            # #for i in range(n_sims):
                
            # sema.acquire()
            

            # # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            
            # jobs.append(p)
            # p.start()
                
            # AL_strategy = AL_strategies.MMSE
            # AL = "AverageCSArea_MMSE"
            # #for i in range(n_sims):
                
            # sema.acquire()
            

            # # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            
            # jobs.append(p)
            # p.start()
            
            # AL_strategy = AL_strategies.SLRGP_Batch
            # AL = "SLRGP_Batch_2"
            # #for i in range(n_sims):
                
            # sema.acquire()
            
            # p = multiprocess.Process(target=train_validate_model, args=(test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
              
            # jobs.append(p)
            # p.start()
        

            #end = time.time()
            #print("....Time taken for one iteration:", end - start)

            
            

            k += 1
                
            # AL_strategy = AL_strategies.SLRV
            # AL = "SLRV"
            # for i in range(n_sims):
                
            #     sema.acquire()
                
            #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
                
            #     jobs.append(p)
            #     p.start()
    # penalties =[AL_strategies.SLRGP_modified_penalty] 
    # # penalties = [AL_strategies.SLRGP_modified_penalty2_1,AL_strategies.SLRGP_modified_penalty2_2,
    # #           AL_strategies.SLRGP_modified_penalty2_3,AL_strategies.SLRGP_modified_penalty2_4B,
    # #           AL_strategies.SLRGP_modified_penalty2_5,AL_strategies.SLRGP_modified_penalty2_6,
    # #           AL_strategies.SLRGP_modified_penalty2_7,AL_strategies.SLRGP_modified_penalty2_8,
    # #           AL_strategies.SLRGP_modified_penalty2_9] 
    # batch_size = 1
    #for _,AL_strategy in enumerate(penalties):
    #for i in range(min_sim,max_sims):
    # i = int(iteration) 
    # for i in range(1,10):
    #     for _,AL_strategy in enumerate(penalties):
    
    #         #start = time.time()
    #         print("Starting iteration " +str(i))
            
    #         kf10 = KFold(n_splits=5, shuffle=True)
    #         k = 0
    #         for train_index, test_index in kf10.split(rn):
                
    #             retrieved_Inputs = Inputs[train_index]
    #             Outputs = retrieved_Outputs[train_index]
                
    #             test_Inputs = Inputs[test_index]
    #             test_Outputs = retrieved_Outputs[test_index]
                
    #             #AL_strategy = AL_strategies.SLRGP_Batch_Group1_6
                
    #             AL = "SLRGP_ModifiedPenlty1" #"2_"+str(_+1)
    #             #for i in range(n_sims):
    #             sema.acquire()
    #             p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
    #             jobs.append(p)
    #             p.start()   
                
    #             k += 1
            
            
    # retrieved_Outputs = np.load('ISR3D_data\Max_PercentLost_LumenArea.npy') 
    # save_location = "ISR3D_PercentLost_LumenArea\Max_SingleGP_"
    # rn = range(1,128)
    # start = time.time()

    # for i in range(29,30):

    #     start = time.time()
    #     print("Starting iteration " +str(i))
        
    #     kf10 = KFold(n_splits=5, shuffle=True)
    #     k = 0
    #     for train_index, test_index in kf10.split(rn):
            
    #         retrieved_Inputs = Inputs[train_index]
    #         Outputs = retrieved_Outputs[train_index]
            
    #         test_Inputs = Inputs[test_index]
    #         test_Outputs = retrieved_Outputs[test_index]
            
    #         AL_strategy = AL_strategies.SLRGP_Batch
    #         batch_size = 1
    #         AL = "SLRGP"
    #         #for i in range(n_sims):
    #         sema.acquire()
    #         p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
    #         jobs.append(p)
    #         p.start()

    
    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
        
    end = time.time()
    print("Time taken for 1 iteration", end - start)
        

def run_GPyMOGP_sims(min_sim = 0,max_sims = 10):
    
    jobs = []
    
    # For naming conventions
    s = "_Area_stats_"
    
    Inputs = np.load('ISR3D_data\input_list_numberseq.npy') #pickle.load(open("ISR3D_data\InputList.pkl", "rb")).to_numpy()

    
    # Save Location
    save_location = "ISR3D_PercentLost_LumenArea\GPyMOGP_"
    
    #retrieved_Outputs = np.load('PercentLost_LumenArea.npy')[:,250]# One Slice
    #retrieved_Inputs = Inputs.to_numpy()
    

    #retrieved_Outputs = np.load('ISR3D_data\Max_PercentLost_LumenArea.npy') 
    #retrieved_Outputs = np.load('ISR3D_data\Mean_PercentLost_LumenArea.npy') 
    #retrieved_Outputs = np.load('ISR3D_data\SD_PercentLost_LumenArea.npy') 
    
    # Set parameter ranges
    ps = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    
    #sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    sema = multiprocess.Semaphore(int(5))
    
    P = 3
    Max_D30 = np.load('ISR3D_data\Max_PercentLost_LumenArea.npy') 
    Mean_D30 = np.load('ISR3D_data\Mean_PercentLost_LumenArea.npy') 
    SD_D30 = np.load('ISR3D_data\SD_PercentLost_LumenArea.npy') 
    Data = [Max_D30,Mean_D30,SD_D30]

    
    # Create range 1 to 128
    rn = range(1,128)
    start = time.time()
    for i in range(min_sim,max_sims):
        start = time.time()
        print("Starting iteration " +str(i))
        
        kf10 = KFold(n_splits=5, shuffle=True)
        k = 0
        for train_index, test_index in kf10.split(rn):
            
            train_inputs = Inputs[train_index]
            train_outputs = [ Data[_][train_index] for _ in range(P) ]
            
            test_inputs = Inputs[test_index]
            test_outputs = [ Data[_][test_index] for _ in range(P) ]
            

            # AL_strategy = SLRGP
            # batch_size = 1
            # AL = "SLRGP"
            # #for i in range(n_sims):
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            # jobs.append(p)
            # p.start()            
                       
            # k += 1

        
    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
        
    end = time.time()
    print("Time taken for 1 iteration", end - start)

import numpy as np
from sklearn.model_selection import LeaveOneOut

def LOOCV_GP():
    
    
    jobs = []
    
    # For naming conventions
    s = "_All_Volume_"
    
    Inputs = np.load('ISR3D_data\input_list_numberseq.npy') 
    
    # Save Location
    save_location = "ISR3D_LOOCV\SD_SingleGP_"
    
    #retrieved_Outputs = np.load('ISR3D_data\Max_PercentLost_LumenArea.npy') 
    #retrieved_Outputs = np.load('ISR3D_data\Mean_PercentLost_LumenArea.npy') 
    retrieved_Outputs = np.load('ISR3D_data\SD_PercentLost_LumenArea.npy') 
    
    # Set parameter ranges
    ps = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    
    #sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    sema = multiprocess.Semaphore(int(10))
    
    loo = LeaveOneOut()
    loo.get_n_splits(Inputs)
    
    print(loo)
    LeaveOneOut()
    it = 0
    
    AL_strategy = AL_strategies.SLRGP_Batch
    batch_size = 5
    AL = "None"
    #for i in range(n_sims):
    k = 0
    i = 0
            
    for train_index, test_index in loo.split(Inputs):
         print("Starting iteration ",it)
         X_train, X_test = Inputs[train_index], Inputs[test_index]
         y_train, y_test = retrieved_Outputs[train_index], retrieved_Outputs[test_index]
         
         sema.acquire()
         p = multiprocess.Process(target=train_full_model, args=(batch_size,save_location,X_test,y_test,k,X_train,y_train,AL_strategy,AL,ps,it,s, sema))
         jobs.append(p)
         p.start()
         
         it += 1
         
def train_full_model(batch_size,save_location,test_Inputs,test_Outputs,k,retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps, run_id,s, sema = None):

        GP, GP_Inputs, GP_Outputs, RMSE, MAX,r2,MAPE  = GP_model.GP_model_FULL(batch_size,retrieved_Inputs,retrieved_Outputs,AL_strategy, test_Inputs,test_Outputs,parameter_ranges=ps)

        print("I'm fully trained! Now time to be tested!!")
        with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_MAPE_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(MAPE, f) 
        
        with open(str(save_location)+ str(AL)+ "_slice_"+str(s)+"_GP_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(GP, f)        
        with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_RMSE_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(RMSE, f) 
        with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_MAX_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(MAX, f) 
        with open(str(save_location)+str(AL)+ "_slice_"+str(s)+"_r2_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(r2, f)                
        print("I'm fully trained and tested!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        if sema:
            sema.release()
         
        
# if __name__ == '__main__':
#     #run_sims_for_ISR3D_Retrieved_slice_Data()
#     #run_sims_for_ISR3D_Retrieved_Volume_Data()
#     run_sims_for_Volume_with_CrossValidation()
#     #run_GPyMOGP_sims()
#     #LOOCV_GP()
    
    
if __name__ == '__main__':
    
    #Setup the argument parser class
    parser = argparse.ArgumentParser(prog='Experiments program',
                                     description='''\
            Performs K-fold validation

             ''')
    #We use the optional switch -- otherwise it is mandatory
    parser.add_argument('iteration', action='store', help='Run id', default=1)
    #Run the argument parser
    args = parser.parse_args()
    #Extract our value or default
    iteration = args.iteration

    run_sims_for_Volume_with_CrossValidation(iteration)
    #sema.release()