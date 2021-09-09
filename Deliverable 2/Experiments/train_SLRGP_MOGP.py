# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:28:28 2021

@author: Cillian
"""
import GPy
import AL_strategies
#from utils import *
import numpy as np
#import Gray_Scott
#from Sobol_indices import get_sobol_indices__via_saltelli
import GP_model_ISR3D as GP_model
import pickle
#import time
from sklearn.metrics import mean_squared_error,max_error,r2_score
from sklearn.model_selection import KFold
import multiprocess

def train_validate_multioutput_model(Outputs,train_index, test_index,test_Inputs,test_Outputs,K,retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps, run_id,s, sema = None):
    # try:
    #     pickle.load(open("GP_data_ISR3D_val\GP_"+str(AL)+ "_slice_"+str(s)+"_Inputs_rep_"+str(run_id)+"_k_" +str(k) +".dump", "rb"))
    #     if sema:
    #         sema.release()
    
    # except:
    Volumes_d10 = Outputs[train_index,241,:].sum(axis = 1)*0.03125
    Volumes_d20 = Outputs[train_index,481,:].sum(axis = 1)*0.03125
    Volumes_d30 = Outputs[train_index,-1,:].sum(axis = 1)*0.03125
    
    test_outputs_d10 = Outputs[test_index,241,:].sum(axis = 1)*0.03125
    test_outputs_d20 = Outputs[test_index,481,:].sum(axis = 1)*0.03125
    test_outputs_d30 = Outputs[test_index,-1,:].sum(axis = 1)*0.03125

    GP, GP_Inputs = GP_model.GP_model(retrieved_Inputs,Volumes_d10,AL_strategy,parameter_ranges=ps)

    # Outputs should have been saved!
    Inputs_indx = []
    for i in range(GP_Inputs.shape[0]):
        Inputs_indx.append( np.where(retrieved_Inputs==GP_Inputs[i])[0][0]  )
        
    # Get outputs to use for for other local models 
    Outputs_d10 = np.zeros((GP_Inputs.shape[0],1))
    Outputs_d20 = np.zeros((GP_Inputs.shape[0],1))
    Outputs_d30 = np.zeros((GP_Inputs.shape[0],1))
    for _,j in enumerate(Inputs_indx):
        Outputs_d10[_] = Volumes_d10[j]
        Outputs_d20[_] = Volumes_d20[j]
        Outputs_d30[_] = Volumes_d30[j]
        
    # Number of parameters
    n = 4
    
    # Number of output dimensions
    #P = 3
    
    #kern_list = [GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True) + GPy.kern.Bias(input_dim=n) for _ in range(P)]
        
    #GP_models = [GPy.models.GPRegression(GP_Inputs,Outputs_d10,kernel = kern_list[_] ) for _ in range(P) ]    
    # Define kernel
    kg = GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True)
    kb = GPy.kern.Bias(input_dim=n)
    k = kg + kb  
    # Train regression function
    GP_model10 = GPy.models.GPRegression(GP_Inputs,Outputs_d10,kernel = k)
    
    # optimise parameters
    GP_model10.optimize()
    GP_model10.optimize_restarts(num_restarts = 10, verbose=False)
    
    # Use the model to make predictions for unevaluated points
    GP_means_10, _ = GP_model10.predict(test_Inputs)
    
    # Define kernel
    kg = GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True)
    kb = GPy.kern.Bias(input_dim=n)
    k = kg + kb
    
    # Train regression function
    GP_model20 = GPy.models.GPRegression(GP_Inputs,Outputs_d20,kernel = k)
    
    # optimise parameters
    GP_model20.optimize()
    GP_model20.optimize_restarts(num_restarts = 10, verbose=False)
    
    # Use the model to make predictions for unevaluated points
    GP_means_20, _ = GP_model20.predict(test_Inputs)
    
    # Define kernel
    kg = GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True)
    kb = GPy.kern.Bias(input_dim=n)
    k = kg + kb
    
    # Train regression function
    GP_model30 = GPy.models.GPRegression(GP_Inputs,Outputs_d30,kernel = k)
    
    # optimise parameters
    GP_model30.optimize()
    GP_model30.optimize_restarts(num_restarts = 10, verbose=False)
    
    # Use the model to make predictions for unevaluated points
    GP_means_30, _ = GP_model30.predict(test_Inputs)
    
    print("I'm fully trained! Now time to be tested!!")
    # Predict validation data
    #GP_means, GP_vars = GP.predict(test_Inputs)
      
    # Get RMSE of Validation data
    RMSE = [mean_squared_error(GP_means_10, test_outputs_d10, squared=False),
            mean_squared_error(GP_means_20, test_outputs_d20, squared=False),
            mean_squared_error(GP_means_30, test_outputs_d30, squared=False)]
      
    # Get Max Error
    MAX = [max_error(GP_means_10, test_outputs_d10),max_error(GP_means_20, test_outputs_d20),
           max_error(GP_means_30, test_outputs_d30)]
      
    # R2 Score
    r2 = [r2_score(test_outputs_d10,GP_means_10),r2_score(test_outputs_d20,GP_means_20),
          r2_score(test_outputs_d30,GP_means_30)]
    
    #scores = [RMSE,MAX,r2]
    GPs = [GP_model10,GP_model20,GP_model30]
    
    with open("GP_data_ISR3D_val\GP_" + str(AL)+ "_slice_"+str(s)+"_GP_rep_"+str(run_id)+"_k_" +str(K) +".dump" , "wb") as f:
      pickle.dump(GPs, f) 
      
    with open("GP_data_ISR3D_val\GP_"+str(AL)+ "_slice_"+str(s)+"_Inputs_rep_"+str(run_id)+"_k_" +str(K) +".dump" , "wb") as f:
      pickle.dump(GP_Inputs, f) 

    with open("GP_data_ISR3D_val\GP_"+str(AL)+ "_slice_"+str(s)+"_RMSE_rep_"+str(run_id)+"_k_" +str(K) +".dump" , "wb") as f:
      pickle.dump(RMSE, f) 
    with open("GP_data_ISR3D_val\GP_"+str(AL)+ "_slice_"+str(s)+"_MAX_rep_"+str(run_id)+"_k_" +str(K) +".dump" , "wb") as f:
      pickle.dump(MAX, f) 
    with open("GP_data_ISR3D_val\GP_"+str(AL)+ "_slice_"+str(s)+"_r2_rep_"+str(run_id)+"_k_" +str(K) +".dump" , "wb") as f:
      pickle.dump(r2, f)               
    print("I'm fully trained and tested!")
    
    if sema:
        sema.release()



def run_sims(n_sims = 30):
    
    jobs = []
    
    # For naming conventions
    s = "_All_Volume_"
    
    Inputs = pickle.load(open("ISR3D_data\InputList.pkl", "rb")).to_numpy()
    data = np.load('ISR3D_data\LumenData.npy')
        
    # Set parameter ranges
    ps = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    
    #sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    sema = multiprocess.Semaphore(int(4))
    
    # Convert to Areas
    retrieved_Outputs = data[:,-1,:].sum(axis = 1)*0.03125
    
    # For different slices
    #for s in range(10,128,10):
    #    retrieved_Outputs = data[:,-1,s]
    
    # Create range 1 to 128
    rn = range(1,128)
    
    for i in range(19,n_sims):
        print("Starting iteration " +str(i))
        
        kf10 = KFold(n_splits=5, shuffle=True)
        k = 0
        for train_index, test_index in kf10.split(rn):
            
            retrieved_Inputs = Inputs[train_index]
            Outputs = retrieved_Outputs[train_index]
            
            test_Inputs = Inputs[test_index]
            test_Outputs = retrieved_Outputs[test_index]
            
            AL_strategy = AL_strategies.SLRGP
            AL = "local_MOGP_SLRGP"
            #for i in range(n_sims):
                
            sema.acquire()
            
            p = multiprocess.Process(target=train_validate_multioutput_model, args=(data,train_index,test_index,  test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            jobs.append(p)
            p.start()
                
            k += 1
            
        
    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
        

        
if __name__ == '__main__':
    run_sims()