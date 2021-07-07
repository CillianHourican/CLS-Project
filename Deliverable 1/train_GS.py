# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 00:34:20 2021

@author: Cillian
"""
import GPy
import AL_strategies
from utils import *
import numpy as np
import Gray_Scott
from Sobol_indices import get_sobol_indices__via_saltelli
import GP_model_GS as GP_model
import pickle
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error,max_error,r2_score

import multiprocess
    
def train_model(model,AL_strategy,AL,ps, run_id, sema = None):
    
    GP, GP_Inputs = GP_model.GP_model(model,AL_strategy,parameter_ranges = ps, M = 5, max_eval = 20, lhs_samples=20, lhs_stratrgy='center')
    
    with open("GP_data_GS_testing\GP_" + str(AL)+"_GP_rep_"+str(run_id) +".dump" , "wb") as f:
      pickle.dump(GP, f) 
      
    with open("GP_data_GS_testing\GP_"+str(AL)+"_Inputs_rep_"+str(run_id) +".dump" , "wb") as f:
      pickle.dump(GP_Inputs, f) 
    
    if sema:
        sema.release()
        
# def train_validate_model(model,AL_strategy,AL,ps, run_id, sema = None):
    
#     GP, GP_Inputs = GP_model.GP_model(model,AL_strategy,parameter_ranges = ps, M = 5, max_eval = 20, lhs_samples=20, lhs_stratrgy='center')
    
#     # Get set of Input Values
#     Grid_points = get_grid_points(5)
    
#     # Compute Ground truth for
    
#     GP_means, GP_vars = GP.predict(Grid_points)
              
#     # Get RMSE of Validation data
#     RMSE = mean_squared_error(GP_means, test_Outputs, squared=False)
      
#     # Get Max Error
#     MAX = max_error(GP_means, test_Outputs)
      
#     # R2 Score
#     r2 = r2_score(test_Outputs,GP_means)
    
#     scores = [RMSE,MAX,r2]
    
#     with open("GP_data_GS_val\GP_" + str(AL)+"_GP_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
#       pickle.dump(GP, f) 
      
#     with open("GP_data_GS_val\GP_"+str(AL)+"_Inputs_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
#       pickle.dump(GP_Inputs, f) 
      
#     with open("GP_data_GS_val\GP_"+str(AL)+"_Val_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
#       pickle.dump(scores, f) 

#     #print(str(AL)+"_GP_rep_"+str(run_id) +".dump")
#     #print(str(AL)+"_Inputs_rep_"+str(run_id) +".dump")
    
#     if sema:
#         sema.release()
    

def run_sims_for_toy_model_GP_dynamics(n_sims = 10):
    
    jobs = []
    
    # Set parameter ranges
    ps = [[0.01,0.002], [0.001,0.0001],[0.1,0.01],[0.2,0.1]]
    model = Gray_Scott.Y
    
    sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-3))
    
    AL_strategy = AL_strategies.ALM
    AL = "ALM"
    for i in range(n_sims):
        
        sema.acquire()
        
        p = multiprocess.Process(target=train_model, args=(model,AL_strategy,AL,ps,i, sema))
        
        jobs.append(p)
        p.start()
        
    AL_strategy = AL_strategies.ALC
    AL = "ALC"
    for i in range(n_sims):
        
        sema.acquire()
        
        p = multiprocess.Process(target=train_model, args=(model,AL_strategy,AL,ps,i, sema))
        
        jobs.append(p)
        p.start()
        
    AL_strategy = AL_strategies.SLRGP
    AL = "SLRGP"
    for i in range(n_sims):
        
        sema.acquire()
        
        p = multiprocess.Process(target=train_model, args=(model,AL_strategy,AL,ps,i, sema))
        
        jobs.append(p)
        p.start()
        
    AL_strategy = AL_strategies.SLRGP_Batch
    AL = "SLRGP_Batch"
    for i in range(3,n_sims):
        
        sema.acquire()
        
        p = multiprocess.Process(target=train_model, args=(model,AL_strategy,AL,ps,i, sema))
        
        jobs.append(p)
        p.start()
    
        
    AL_strategy = AL_strategies.IMSE
    AL = "IMSE"
    for i in range(13,n_sims):
        
        sema.acquire()
        
        p = multiprocess.Process(target=train_model, args=(model,AL_strategy,AL,ps,i, sema))
        
        jobs.append(p)
        p.start()
        
    AL_strategy = AL_strategies.MMSE
    AL = "MMSE"
    for i in range(n_sims):
        
        sema.acquire()
        
        p = multiprocess.Process(target=train_model, args=(model,AL_strategy,AL,ps,i, sema))
        
        jobs.append(p)
        p.start()
        

    
    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
        
if __name__ == '__main__':
    run_sims_for_toy_model_GP_dynamics()